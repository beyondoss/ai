//! The billing facts beyond token counts, read on every wire that reports them.

use super::vendor::usd_to_e10;
use super::*;

#[test]
fn e10_round_trips_exactly_as_a_decimal() {
    assert_eq!(e10_to_usd(0), "0");
    assert_eq!(e10_to_usd(12_345), "0.0000012345");
    assert_eq!(e10_to_usd(10_000_000_000), "1");
    assert_eq!(e10_to_usd(15_000_000_000), "1.5");
    assert_eq!(usd_to_e10(0.0012), Some(12_000_000));
    assert_eq!(usd_to_e10(-1.0), None);
    assert_eq!(usd_to_e10(f64::NAN), None);
    assert_eq!(usd_to_e10(f64::INFINITY), None);
}

/// Anthropic Messages, non-stream: speed, geography, every server-tool counter, the message id
/// and the container (https://platform.claude.com/docs/en/api/messages/create).
#[test]
fn anthropic_body_reads_speed_geo_tools_id_and_container() {
    let body = br#"{"id":"msg_01abc","type":"message","container":{"id":"container_011x","expires_at":"2026-10-10T00:00:00Z"},
        "usage":{"input_tokens":10,"output_tokens":5,"speed":"fast","inference_geo":"us","service_tier":"standard",
        "server_tool_use":{"web_search_requests":3,"web_fetch_requests":2,"code_execution_requests":1}}}"#;
    let u = anthropic_body(body).unwrap();
    assert_eq!(u.speed.as_deref(), Some("fast"));
    assert_eq!(u.inference_geo.as_deref(), Some("us"));
    assert_eq!(u.server_tool_calls, 3, "the old field keeps its meaning");
    assert_eq!(
        (
            u.server_tools.web_search,
            u.server_tools.web_fetch,
            u.server_tools.code_execution
        ),
        (3, 2, 1)
    );
    assert_eq!(u.upstream.generation_id.as_deref(), Some("msg_01abc"));
    assert_eq!(u.upstream.container_id.as_deref(), Some("container_011x"));
    assert_eq!(
        u.server_tools.to_row().as_deref(),
        Some("web_search=3,web_fetch=2,code_execution=1")
    );
}

/// Anthropic streams `usage.speed` (and `inference_geo`) on `message_start` only: the final
/// `message_delta` leaves them out (measured live on claude-opus-4-8 in fast mode), so a delta
/// without them must not erase them. The container rides on `message_delta.delta`.
#[test]
fn anthropic_stream_keeps_start_only_speed_and_reads_the_delta_container() {
    let sse = b"event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_9\",\"usage\":{\"input_tokens\":7,\"output_tokens\":1,\"speed\":\"fast\",\"inference_geo\":\"global\"}}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"container\":{\"id\":\"container_s\"}},\"usage\":{\"output_tokens\":40,\"server_tool_use\":{\"web_search_requests\":2,\"web_fetch_requests\":1}}}\n\n";
    let u = anthropic_stream(sse).unwrap();
    assert_eq!(u.speed.as_deref(), Some("fast"));
    assert_eq!(u.inference_geo.as_deref(), Some("global"));
    assert_eq!(u.upstream.generation_id.as_deref(), Some("msg_9"));
    assert_eq!(u.upstream.container_id.as_deref(), Some("container_s"));
    assert_eq!(
        (u.server_tools.web_search, u.server_tools.web_fetch),
        (2, 1)
    );
    assert_eq!(u.server_tool_calls, 2);
    assert_eq!(u.output_tokens, 40);
}

/// OpenRouter Chat Completions: `usage.cost` (always sent now), `is_byok`, `cost_details`,
/// `server_tool_use_details`, the root `provider` and `id`
/// (https://openrouter.ai/docs/use-cases/usage-accounting, https://openrouter.ai/openapi.json).
#[test]
fn openrouter_chat_reads_cost_byok_tools_host_and_id_body_and_stream() {
    let body = br#"{"id":"gen-123-abc","provider":"Amazon Bedrock","model":"anthropic/claude-opus-4.8",
        "usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":0.0012,"is_byok":false,
        "cost_details":{"upstream_inference_cost":null,"upstream_inference_prompt_cost":0.0008,"upstream_inference_completions_cost":0.0004,"server_tool_cost":0.007},
        "server_tool_use_details":{"tool_calls_executed":2,"tool_calls_requested":2,"web_search_requests":1}}}"#;
    let u = openai_body(body).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(12_000_000));
    assert_eq!(
        u.upstream.inference_cost_e10, None,
        "null is absent, not zero"
    );
    assert_eq!(u.upstream.tool_cost_e10, Some(70_000_000));
    assert_eq!(u.upstream.byok, Some(false));
    assert_eq!(u.upstream.served_by.as_deref(), Some("Amazon Bedrock"));
    assert_eq!(u.upstream.generation_id.as_deref(), Some("gen-123-abc"));
    assert_eq!(
        (u.server_tools.tool_calls, u.server_tools.web_search),
        (2, 1)
    );

    let sse = b"data: {\"id\":\"gen-9\",\"provider\":\"Anthropic\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
data: {\"id\":\"gen-9\",\"provider\":\"Anthropic\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4,\"cost\":0.5,\"is_byok\":true,\"cost_details\":{\"upstream_inference_cost\":0.4},\"server_tool_use\":{\"web_search_requests\":4}}}\n\n\
data: [DONE]\n\n";
    let u = openai_stream(sse).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(5_000_000_000));
    assert_eq!(u.upstream.inference_cost_e10, Some(4_000_000_000));
    assert_eq!(u.upstream.byok, Some(true));
    assert_eq!(u.upstream.served_by.as_deref(), Some("Anthropic"));
    assert_eq!(u.upstream.generation_id.as_deref(), Some("gen-9"));
    assert_eq!(u.server_tools.web_search, 4, "the overview docs' spelling");
}

/// OpenRouter Messages: Anthropic usage plus `cost` and `server_tool_use.tool_calls_executed`.
#[test]
fn openrouter_messages_reads_cost_and_executed_tool_calls() {
    let body = br#"{"id":"gen-m1","type":"message","usage":{"input_tokens":5,"output_tokens":2,"cost":0.25,"is_byok":false,"speed":"fast","service_tier":"priority",
        "server_tool_use":{"web_search_requests":1,"tool_calls_executed":3}}}"#;
    let u = anthropic_body(body).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(2_500_000_000));
    assert_eq!(
        (u.server_tools.web_search, u.server_tools.tool_calls),
        (1, 3)
    );
    assert_eq!(u.speed.as_deref(), Some("fast"));
    assert_eq!(u.service_tier.as_deref(), Some("priority"));
}

/// OpenRouter Responses streams its cost inside `response.completed.response.usage`, and the
/// response id beside it.
#[test]
fn openrouter_responses_stream_reads_cost_and_id() {
    let sse = b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"gen-r1\",\"service_tier\":\"default\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4,\"cost\":0.01,\"is_byok\":false}}}\n\n";
    let u = openai_stream(sse).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(100_000_000));
    assert_eq!(u.upstream.generation_id.as_deref(), Some("gen-r1"));
}

/// xAI reports its own exact cost in 1e-10 USD ticks and every server-side tool by kind, on
/// Chat Completions and Responses (https://docs.x.ai/developers/cost-tracking,
/// https://docs.x.ai/developers/tools/tool-usage-details).
#[test]
fn xai_reads_ticks_sources_and_tool_details_on_both_apis() {
    let chat = br#"{"id":"x-1","service_tier":"priority","usage":{"prompt_tokens":185,"completion_tokens":5,"total_tokens":190,"num_sources_used":7,"cost_in_usd_ticks":2187000,
        "server_side_tool_usage_details":{"web_search_calls":2,"x_search_calls":1,"x_posts_fetched":12,"x_users_fetched":3,"code_interpreter_calls":1,"file_search_calls":4,"mcp_calls":5,"document_search_calls":6,"image_generation_calls":7}}}"#;
    let u = openai_body(chat).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(2_187_000));
    assert_eq!(u.service_tier.as_deref(), Some("priority"));
    let want = ServerTools {
        web_search: 2,
        x_search: 1,
        x_posts: 12,
        x_users: 3,
        code_execution: 1,
        file_search: 4,
        mcp: 5,
        document_search: 6,
        image_generation: 7,
        sources: 7,
        ..ServerTools::default()
    };
    assert_eq!(u.server_tools, want);
    assert_eq!(u.upstream.generation_id.as_deref(), Some("x-1"));

    let responses = b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_x\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4,\"cost_in_usd_ticks\":2337000,\"num_server_side_tools_used\":1,\"server_side_tool_usage_details\":{\"web_search_calls\":1}}}}\n\n";
    let u = openai_stream(responses).unwrap();
    assert_eq!(u.upstream.cost_e10, Some(2_337_000));
    assert_eq!(u.server_tools.web_search, 1);
    assert_eq!(u.upstream.generation_id.as_deref(), Some("resp_x"));
}

/// A malformed vendor member costs that member, never the token counts.
#[test]
fn malformed_vendor_members_never_cost_the_usage() {
    for bad in [
        r#""cost":"0.1""#,
        r#""cost":{"x":1}"#,
        r#""is_byok":"yes""#,
        r#""cost_in_usd_ticks":-5"#,
        r#""server_side_tool_usage_details":{"web_search_calls":-1}"#,
        r#""server_side_tool_usage_details":{"web_search_calls":1.5}"#,
        r#""server_tool_use":{"web_search_requests":"3"}"#,
        r#""num_sources_used":null"#,
        r#""speed":"FAST!""#,
        r#""inference_geo":["us"]"#,
    ] {
        let body = format!(
            r#"{{"id":"bad id with spaces","usage":{{"prompt_tokens":3,"completion_tokens":1,{bad}}}}}"#
        );
        let u = openai_body(body.as_bytes()).unwrap_or_else(|| panic!("{bad}"));
        assert_eq!((u.input_tokens, u.output_tokens), (3, 1), "{bad}");
        assert!(!u.server_tools.any(), "{bad}");
        assert_eq!(u.upstream.cost_e10, None, "{bad}");
        assert_eq!(
            u.upstream.generation_id, None,
            "an id outside the charset is dropped"
        );
        assert_eq!(u.speed, None, "{bad}");
        assert_eq!(u.inference_geo, None, "{bad}");
    }
}

/// A cache entry keeps what the client is billed for and drops the facts of the call that
/// filled it.
#[test]
fn a_cache_entry_does_not_replay_the_upstream_call() {
    let body = br#"{"id":"gen-1","usage":{"prompt_tokens":3,"completion_tokens":1,"cost":0.5}}"#;
    let u = openai_body(body).unwrap().for_cache();
    assert_eq!(u.upstream, Upstream::default());
    assert_eq!(u.input_tokens, 3);
}

/// The typed facts the gateway prices from ([`Usage::pricer_row`]) price exactly as the row's
/// logged text did when the pricer read that text back: `server_tools` through
/// [`ServerTools::to_row`] and `ToolCounts::parse`, the reported costs through [`e10_to_usd`] and
/// `parse_usd_atto`, the tiers from their logged strings. Across every catalog row and candidate,
/// both wires, and a corpus of every tool kind (alone, at one and at its largest count, all at
/// once), costs from zero to `u64::MAX`, and every tier, speed and geo spelling, known and not,
/// the two give the same `Result`, refusal reason included. (`providers::pricing`'s
/// `typed_tiers_decide_as_the_text_did` holds the typed tier readings to the old text matching.)
#[test]
fn pricer_row_prices_as_the_logged_text() {
    use providers::pricing::{Geo, Speed, Tier, ToolCounts, UsageRow, parse_usd_atto, price};

    const SPELLINGS: &[&str] = &[
        "default",
        "standard",
        "on_demand",
        "auto",
        "priority",
        "fast",
        "ultrafast",
        "flex",
        "scale",
        "global",
        "us",
        "not_available",
        "eu",
    ];
    let tier = |s: &str| Some(tier_from(s).unwrap());

    let mut facts = vec![Usage::default()];
    let mut every = ServerTools::default();
    for (i, (k, _)) in ServerTools::default().entries().into_iter().enumerate() {
        for n in [1, u32::MAX] {
            let mut u = Usage::default();
            set_kind(&mut u.server_tools, k, n);
            facts.push(u);
        }
        set_kind(&mut every, k, u32::try_from(i).unwrap() + 1);
    }
    facts.push(Usage {
        server_tools: every,
        ..Usage::default()
    });
    // OpenRouter's web searches beside its every-kind `tool_calls`, the pair its pricing compares.
    for (w, c) in [(2, 2), (1, 2), (3, 0)] {
        let mut u = Usage::default();
        u.server_tools.web_search = w;
        u.server_tools.tool_calls = c;
        facts.push(u);
    }
    let costs = [
        None,
        Some(0),
        Some(1),
        Some(12_345),
        Some(10_000_000_000),
        Some(15_000_000_000),
        Some(123_456_789_012),
        Some(u64::MAX),
    ];
    for cost_e10 in costs {
        for tool_cost_e10 in costs {
            let mut u = Usage::default();
            u.upstream.cost_e10 = cost_e10;
            u.upstream.tool_cost_e10 = tool_cost_e10;
            facts.push(u);
        }
    }
    let spelled = || std::iter::once(None).chain(SPELLINGS.iter().map(|s| tier(s)));
    for service_tier in spelled() {
        for speed in spelled() {
            facts.push(Usage {
                service_tier,
                speed,
                ..Usage::default()
            });
        }
        facts.push(Usage {
            inference_geo: service_tier,
            ..Usage::default()
        });
    }
    // Several facts on one row, and a named OpenRouter host.
    for (t, s, g, c) in [
        ("priority", "fast", "us", Some(4_200_000)),
        ("flex", "standard", "global", None),
        ("standard", "fast", "not_available", Some(0)),
    ] {
        let mut u = Usage {
            service_tier: tier(t),
            speed: tier(s),
            inference_geo: tier(g),
            ..Usage::default()
        };
        u.server_tools.web_search = 2;
        u.upstream.cost_e10 = c;
        u.upstream.served_by = host_from("Anthropic");
        facts.push(u);
    }

    let (mut rows, mut priced) = (0usize, 0usize);
    let mut reasons = std::collections::BTreeSet::new();
    for rr in providers::rates::ROW_RATES {
        for cc in rr.cost {
            let provider = providers::by_id(cc.provider).name;
            for wire in [Dialect::Anthropic, Dialect::OpenAi] {
                for f in &facts {
                    let u = Usage {
                        input_tokens: 100_000,
                        output_tokens: 777,
                        cache_read_tokens: 40_000,
                        cache_write_tokens: 5_000,
                        cache_write_1h_tokens: 2_000,
                        ..*f
                    };
                    let typed = price(&UsageRow {
                        price_model: Some(rr.model),
                        provider: Some(provider),
                        unix_secs: 1_791_806_400,
                        ..u.pricer_row(wire)
                    });
                    // The old path: the row's text exactly as `logging` writes it, read back.
                    let tools_text = u.server_tools.to_row();
                    let cost_text = u.upstream.cost_e10.map(e10_to_usd);
                    let tool_cost_text = u.upstream.tool_cost_e10.map(e10_to_usd);
                    let text = ToolCounts::parse(tools_text.as_deref().unwrap_or("")).and_then(
                        |server_tools| {
                            price(&UsageRow {
                                price_model: Some(rr.model),
                                provider: Some(provider),
                                usage_wire: u.wire.unwrap_or(wire),
                                input_tokens: u.input_tokens,
                                output_tokens: u.output_tokens,
                                cache_read_tokens: u.cache_read_tokens,
                                cache_write_tokens: u.cache_write_tokens,
                                cache_write_1h_tokens: u.cache_write_1h_tokens,
                                gateway_cache_write_tokens: u.gateway_cache_write_tokens,
                                server_tools,
                                service_tier: u.service_tier.as_deref().map(Tier::parse),
                                speed: u.speed.as_deref().map(Speed::parse),
                                inference_geo: u.inference_geo.as_deref().map(Geo::parse),
                                upstream_cost_usd: cost_text.as_deref().map(parse_usd_atto),
                                upstream_tool_cost_usd: tool_cost_text
                                    .as_deref()
                                    .map(parse_usd_atto),
                                served_by: u.upstream.served_by.as_deref(),
                                unix_secs: 1_791_806_400,
                                ..UsageRow::default()
                            })
                        },
                    );
                    assert_eq!(typed, text, "{} on {provider} ({wire:?}): {u:?}", rr.model);
                    rows += 1;
                    match typed {
                        Ok(_) => priced += 1,
                        Err(e) => {
                            reasons.insert(e.as_str());
                        }
                    }
                }
            }
        }
    }
    // Not vacuous: rows price, and the refusals the corpus is built to reach are reached.
    assert!(priced * 10 > rows, "{priced} of {rows} priced");
    for want in ["unknown_class", "unknown_geo", "no_tool_fee", "no_rate"] {
        assert!(reasons.contains(want), "{want} never reached: {reasons:?}");
    }
}

/// The two halves of the typed path, held to the text they replace directly, below the price
/// (where two kinds with the same fee would agree even if swapped): each kind's typed count is
/// what its logged `kind=count` text parses to, alone and with every kind at once, and an e10
/// cost's attodollars are what its logged decimal parses to.
#[test]
fn typed_tools_and_costs_equal_their_text() {
    use providers::pricing::{ToolCounts, parse_usd_atto};
    let same = |t: &ServerTools| {
        let text = t.to_row();
        assert_eq!(
            ToolCounts::from(t),
            ToolCounts::parse(text.as_deref().unwrap_or("")).unwrap(),
            "{text:?}"
        );
    };
    same(&ServerTools::default());
    let mut every = ServerTools::default();
    for (i, (k, _)) in ServerTools::default().entries().into_iter().enumerate() {
        for n in [1, 7, u32::MAX] {
            let mut t = ServerTools::default();
            set_kind(&mut t, k, n);
            same(&t);
        }
        set_kind(&mut every, k, u32::try_from(i).unwrap() + 1);
    }
    same(&every);
    for v in [
        0,
        1,
        9,
        12_345,
        10_000_000_000,
        15_000_000_000,
        u64::MAX - 1,
        u64::MAX,
    ] {
        let mut u = Usage::default();
        u.upstream.cost_e10 = Some(v);
        u.upstream.tool_cost_e10 = Some(v);
        let r = u.pricer_row(Dialect::Anthropic);
        let text = Some(parse_usd_atto(&e10_to_usd(v)));
        assert_eq!(
            (r.upstream_cost_usd, r.upstream_tool_cost_usd),
            (text, text),
            "{v}"
        );
    }
}

/// Set one [`ServerTools`] kind by its row key, so a corpus built from [`ServerTools::entries`]
/// names every kind the row can; a kind added there and not here panics.
fn set_kind(t: &mut ServerTools, k: &str, n: u32) {
    *match k {
        "web_search" => &mut t.web_search,
        "web_search_preview" => &mut t.web_search_preview,
        "web_search_page" => &mut t.web_search_page,
        "web_fetch" => &mut t.web_fetch,
        "code_execution" => &mut t.code_execution,
        "file_search" => &mut t.file_search,
        "image_generation" => &mut t.image_generation,
        "computer_use" => &mut t.computer_use,
        "mcp" => &mut t.mcp,
        "shell" => &mut t.shell,
        "tool_search" => &mut t.tool_search,
        "x_search" => &mut t.x_search,
        "x_posts" => &mut t.x_posts,
        "x_users" => &mut t.x_users,
        "document_search" => &mut t.document_search,
        "sources" => &mut t.sources,
        "tool_calls" => &mut t.tool_calls,
        k => panic!("unmapped kind {k}"),
    } = n;
}
