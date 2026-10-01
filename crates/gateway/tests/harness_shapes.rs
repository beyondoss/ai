//! Request shapes real coding-agent harnesses (Codex, pi) send through the gateway, and what the
//! upstream and the harness see. Each test is a defect a live harness cell found: the body here is
//! the captured shape, cut to the fields that decided the outcome.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use serde_json::{Value, json};

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

async fn post(gw: &Gateway, sk: &ed25519_dalek::SigningKey, path: &str, body: &Value) -> Value {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(sk)))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    serde_json::from_str(&text).unwrap_or(Value::String(text))
}

fn captured(mock: &MockUpstream) -> (Captured, Value) {
    let cap = mock.captured().expect("the request reached the upstream");
    let body = serde_json::from_slice(&cap.body).unwrap();
    (cap, body)
}

const APPLY_PATCH_GRAMMAR: &str = "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n";

/// Codex's tool list, as 0.159 sends it to a GPT-5 model: a plain function, `apply_patch` as a
/// `custom` tool with a Lark grammar, and its multi-agent tools grouped in a `namespace`.
fn codex_tools() -> Value {
    json!([
        {"type": "function", "name": "shell", "description": "Run a command.", "strict": false,
         "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}},
        {"type": "custom", "name": "apply_patch", "description": "Edit files with a patch.",
         "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_GRAMMAR}},
        {"type": "namespace", "name": "multi_agent_v1", "description": "Spawn and manage sub-agents.", "tools": [
            {"type": "function", "name": "spawn_agent", "description": "Spawn a sub-agent.",
             "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}},
            {"type": "function", "name": "wait", "parameters": {"type": "object", "properties": {}}}
        ]}
    ])
}

/// A Codex turn: `store: false` and encrypted reasoning asked back, on every request.
fn codex_body(model: &str, input: Value) -> Value {
    json!({
        "model": model,
        "instructions": "You are Codex.",
        "store": false,
        "stream": false,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "tools": codex_tools(),
        "input": input,
    })
}

/// Codex sends `store: false` on every turn. On a GPT row that has a Responses arm but lists Chat
/// Completions first (gpt-5-mini, gpt-5.5), that used to count as "no session state, so translate
/// onto Chat Completions", where Codex's `namespace` tools and `custom` grammars have no shape and
/// OpenAI 400s. The row's own Responses arm takes the body as it came (less the reasoning items the
/// gateway minted, D50).
/// claim: W2, E3, CAT-9
/// defect: D74
#[tokio::test]
async fn codex_on_a_chat_first_gpt_row_is_relayed_to_its_responses_arm() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    for model in ["gpt-5-mini", "gpt-5.5"] {
        let body = codex_body(
            model,
            json!([
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the test"}]},
                {"type": "reasoning", "id": "rs_gw18f2c3a4b5d60001", "encrypted_content": "rs_gw:EqQBsig==", "summary": []},
                {"type": "function_call", "call_id": "call_1", "namespace": "multi_agent_v1", "name": "wait", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ]),
        );
        post(&gw, &sk, "/v1/responses", &body).await;

        let (cap, got) = captured(&mock);
        assert_eq!(
            cap.path, "/v1/responses",
            "{model}: served by the Responses arm"
        );
        assert_eq!(
            got["tools"],
            codex_tools(),
            "{model}: tools relayed as sent"
        );
        assert_eq!(got["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(got["store"], false);
        assert_eq!(got["input"][1]["namespace"], "multi_agent_v1", "{got}");
        assert!(
            !got.to_string().contains("rs_gw"),
            "{model}: a gateway reasoning item reached OpenAI: {got}"
        );
    }
}

/// Claude's answer to a Codex turn: a call into the flattened namespace, and `apply_patch` as the
/// object-wrapped free-form input the gateway asked for.
const CLAUDE_CODEX_TOOL_JSON: &str = r#"{"id":"msg_mock","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"tool_use","id":"toolu_1","name":"multi_agent_v1__spawn_agent","input":{"task":"fix"}},{"type":"tool_use","id":"toolu_2","name":"apply_patch","input":{"input":"*** Begin Patch\n*** End Patch\n"}}],"stop_reason":"tool_use","usage":{"input_tokens":13,"output_tokens":7}}"#;

/// The same answer from a Chat Completions host.
const CHAT_CODEX_TOOL_JSON: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","model":"anthropic/claude-sonnet-4","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{\"task\":\"fix\"}"}},{"id":"call_2","type":"custom","custom":{"name":"apply_patch","input":"*** Begin Patch\n*** End Patch\n"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

/// What Codex must get back either way: the namespace call under its own name and namespace, and
/// `apply_patch` as a `custom_tool_call` with raw input.
fn assert_codex_calls(resp: &Value) {
    let out = resp["output"].as_array().expect("output");
    let call = out
        .iter()
        .find(|i| i["type"] == "function_call")
        .expect("a function call");
    assert_eq!(call["name"], "spawn_agent", "{resp}");
    assert_eq!(call["namespace"], "multi_agent_v1", "{resp}");
    assert_eq!(
        serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap(),
        json!({"task": "fix"})
    );
    let custom = out
        .iter()
        .find(|i| i["type"] == "custom_tool_call")
        .unwrap_or_else(|| panic!("a custom tool call: {resp}"));
    assert_eq!(custom["name"], "apply_patch");
    assert_eq!(custom["input"], "*** Begin Patch\n*** End Patch\n");
}

/// On every row without a Responses arm, Codex's request is translated, and its tools must survive
/// it: a `namespace` flattens into its member tools (`multi_agent_v1__spawn_agent`), mapped back
/// to `{name, namespace}` on the calls Codex receives, and a `custom` tool keeps its grammar
/// (Chat Completions nests it under `grammar`; Messages has no free-form tool, so the tool takes one
/// `input` string and its description carries the grammar). Forwarded as they came, Anthropic 400ed
/// the `namespace` tag and OpenAI the unnested grammar.
/// claim: W2
/// defect: D75
#[tokio::test]
async fn codex_tools_survive_translation_and_calls_map_back() {
    let history = json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the test"}]},
        {"type": "function_call", "call_id": "toolu_0", "namespace": "multi_agent_v1", "name": "wait", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "toolu_0", "output": "done"},
    ]);

    // Messages: a Claude row on Anthropic.
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock =
        MockUpstream::start(Mode::Raw(200, "application/json", CLAUDE_CODEX_TOOL_JSON)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    let resp = post(
        &gw,
        &sk,
        "/v1/responses",
        &codex_body("claude-opus-4-8", history.clone()),
    )
    .await;
    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/messages");
    let tools = got["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(
        names,
        [
            "shell",
            "apply_patch",
            "multi_agent_v1__spawn_agent",
            "multi_agent_v1__wait"
        ],
        "{got}"
    );
    assert!(
        tools.iter().all(|t| t.get("type").is_none()),
        "every tool is a plain client tool: {got}"
    );
    let patch = &tools[1];
    assert_eq!(
        patch["input_schema"]["properties"]["input"]["type"], "string",
        "{patch}"
    );
    assert_eq!(patch["input_schema"]["required"], json!(["input"]));
    let desc = patch["description"].as_str().unwrap();
    assert!(desc.starts_with("Edit files with a patch."), "{desc}");
    assert!(
        desc.contains("lark") && desc.contains(APPLY_PATCH_GRAMMAR),
        "{desc}"
    );
    let spawn = &tools[2];
    assert_eq!(spawn["input_schema"]["required"], json!(["task"]));
    assert!(
        spawn["description"]
            .as_str()
            .unwrap()
            .contains("Spawn a sub-agent."),
        "{spawn}"
    );
    assert_eq!(
        got["messages"][1]["content"][0]["name"], "multi_agent_v1__wait",
        "a replayed namespaced call keeps the flat name: {got}"
    );
    assert_codex_calls(&resp);

    // Chat Completions: a Claude row OpenRouter serves.
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", CHAT_CODEX_TOOL_JSON)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    let resp = post(
        &gw,
        &sk,
        "/v1/responses",
        &codex_body("claude-sonnet-4", history),
    )
    .await;
    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/api/v1/chat/completions");
    let tools = got["tools"].as_array().unwrap();
    assert!(
        tools.iter().all(|t| t["type"] != "namespace"),
        "no namespace reaches Chat Completions: {got}"
    );
    assert_eq!(
        tools[1],
        json!({"type": "custom", "custom": {"name": "apply_patch", "description": "Edit files with a patch.",
            "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": APPLY_PATCH_GRAMMAR}}}})
    );
    assert_eq!(tools[2]["function"]["name"], "multi_agent_v1__spawn_agent");
    assert_eq!(tools[3]["function"]["name"], "multi_agent_v1__wait");
    assert_eq!(
        got["messages"][2]["tool_calls"][0]["function"]["name"], "multi_agent_v1__wait",
        "{got}"
    );
    assert_codex_calls(&resp);
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})
}

/// Turn 2 of a thinking + tool loop, as pi sends it to a Claude model OpenRouter serves: the tool
/// call comes back, its reasoning as a plain unsigned `reasoning` string, and no
/// `reasoning_details`. OpenRouter forwards `reasoning_effort` to Anthropic as enabled thinking,
/// and Anthropic 400s a final assistant turn that does not open on a signed thinking block. The
/// request goes without reasoning instead, as D14 does on Anthropic itself; a turn that does carry
/// signed `reasoning_details` keeps it.
/// claim: TRN-7, W4
/// defect: D77
#[tokio::test]
#[ignore = "D77 reproduced: reasoning_effort reaches OpenRouter on a Claude tool turn without replayable reasoning"]
async fn a_claude_tool_turn_without_reasoning_details_goes_without_reasoning_on_openrouter() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &mock.authority())
        .start()
        .await;

    let call = json!([{"id": "toolu_01", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]);
    let turn2 = |model: &str, assistant: Value| {
        json!({
            "model": model,
            "reasoning_effort": "medium",
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                assistant,
                {"role": "tool", "tool_call_id": "toolu_01", "content": "sunny"},
            ],
        })
    };
    let pi = json!({"role": "assistant", "content": "", "reasoning": "Need weather.", "tool_calls": call});
    let signed = json!({"role": "assistant", "content": "", "tool_calls": call, "reasoning_details": [
        {"type": "reasoning.text", "text": "Need weather.", "signature": "EqQBsig==",
         "format": "anthropic-claude-v1", "index": 0}]});

    let mut bad = Vec::new();
    // Same wire (claude-sonnet-4 is OpenRouter-only), and a Claude row's OpenRouter failover.
    for model in ["claude-sonnet-4", "claude-sonnet-4-5"] {
        post(&gw, &sk, "/v1/chat/completions", &turn2(model, pi.clone())).await;
        let (cap, got) = captured(&mock);
        assert_eq!(cap.path, "/api/v1/chat/completions");
        if got.get("reasoning_effort").is_some() || got.get("reasoning").is_some() {
            bad.push(format!("{model}, pi's turn 2: {got}"));
        }

        // Control: replayable reasoning keeps the request's reasoning.
        post(
            &gw,
            &sk,
            "/v1/chat/completions",
            &turn2(model, signed.clone()),
        )
        .await;
        let (_, got) = captured(&mock);
        assert_eq!(got["reasoning_effort"], "medium", "{model}: {got}");
    }

    // Translated onto OpenRouter: a Responses client that drops its reasoning items.
    post(
        &gw,
        &sk,
        "/v1/responses",
        &json!({"model": "claude-sonnet-4", "store": false, "reasoning": {"effort": "medium"},
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
            "input": [
                {"role": "user", "content": "weather in Paris?"},
                {"type": "function_call", "call_id": "toolu_01", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "toolu_01", "output": "sunny"},
            ]}),
    )
    .await;
    let (_, got) = captured(&mock);
    if got.get("reasoning_effort").is_some() || got.get("reasoning").is_some() {
        bad.push(format!("Responses → OpenRouter Chat: {got}"));
    }

    // Turn 1 has no tool turn to replay: its reasoning stays.
    post(
        &gw,
        &sk,
        "/v1/chat/completions",
        &json!({"model": "claude-sonnet-4", "reasoning_effort": "medium", "tools": [weather_tool()],
            "messages": [{"role": "user", "content": "weather in Paris?"}]}),
    )
    .await;
    let (_, got) = captured(&mock);
    assert_eq!(got["reasoning_effort"], "medium", "{got}");

    assert!(bad.is_empty(), "{bad:#?}");
}
