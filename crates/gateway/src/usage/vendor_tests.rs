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
