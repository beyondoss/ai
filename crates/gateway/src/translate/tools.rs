//! Tool and tool-choice mappings between the three wires.

use super::*;

/// How a Responses client's tools were reshaped on the way to a Chat Completions or Messages
/// upstream, so the calls that come back can be named the way the client offered them.
///
/// - A `namespace` tool (Codex groups its multi-agent and MCP tools in one) has no shape off
///   Responses: its member tools are sent flat, as [`namespaced_name`] (`multi_agent__spawn`),
///   and a call to one comes back as `{name, namespace}`.
/// - A `custom` (free-form input) tool has no Messages shape: it is sent as a tool taking one
///   `input` string, and a call to it comes back as a `custom_tool_call` with that string as its
///   raw input.
///
/// Empty (no allocation) for every other request.
#[derive(Debug, Default, Clone)]
pub struct ToolNames {
    /// `(flat name, namespace, member name)` for each namespaced member tool.
    namespaced: Vec<(String, String, String)>,
    /// Flat names of the `custom` tools sent to Messages as a one-string-field tool.
    wrapped: Vec<String>,
}

impl ToolNames {
    pub(super) fn of_responses(v: &Value, wrap_custom: bool) -> Self {
        let mut names = Self::default();
        for t in v
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            match t.get("type").and_then(Value::as_str) {
                Some("namespace") => {
                    for m in namespace_members(t) {
                        if wrap_custom
                            && m.tool.get("type").and_then(Value::as_str) == Some("custom")
                        {
                            names.wrapped.push(m.flat.clone());
                        }
                        names
                            .namespaced
                            .push((m.flat, m.namespace.to_owned(), m.name.to_owned()));
                    }
                }
                Some("custom") if wrap_custom => {
                    if let Some(name) = t.get("name").and_then(Value::as_str) {
                        names.wrapped.push(name.to_owned());
                    }
                }
                _ => {}
            }
        }
        names
    }

    /// The name a call to `flat` had in the client's request, and its namespace if it had one.
    pub(super) fn restore<'a>(&'a self, flat: &'a str) -> (&'a str, Option<&'a str>) {
        self.namespaced
            .iter()
            .find(|(f, _, _)| f == flat)
            .map_or((flat, None), |(_, ns, name)| {
                (name.as_str(), Some(ns.as_str()))
            })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.namespaced.is_empty() && self.wrapped.is_empty()
    }

    /// Whether `flat` is a `custom` tool sent to Messages wrapped in an `input` object.
    pub(super) fn wrapped(&self, flat: &str) -> bool {
        self.wrapped.iter().any(|w| w == flat)
    }
}

/// A namespace member's flat name: `{namespace}__{name}`, or `{namespace}{name}` when the
/// namespace already ends in `_` (Codex's MCP namespaces, `mcp__server__`, then read as the
/// `mcp__server__tool` names Codex itself gives MCP tools elsewhere). Both fit every provider's
/// tool-name alphabet.
pub(super) fn namespaced_name(namespace: &str, name: &str) -> String {
    if namespace.is_empty() {
        name.to_owned()
    } else if namespace.ends_with('_') {
        format!("{namespace}{name}")
    } else {
        format!("{namespace}__{name}")
    }
}

/// One named member of a `namespace` tool.
pub(super) struct NamespaceMember<'a> {
    pub namespace: &'a str,
    pub name: &'a str,
    /// [`namespaced_name`] of the two.
    pub flat: String,
    pub tool: &'a Value,
}

/// The members of a `namespace` tool, skipping nameless ones: the one walk of a namespace.
pub(super) fn namespace_members(t: &Value) -> impl Iterator<Item = NamespaceMember<'_>> {
    let namespace = t.get("name").and_then(Value::as_str).unwrap_or("");
    t.get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(move |tool| {
            let name = tool.get("name").and_then(Value::as_str)?;
            Some(NamespaceMember {
                namespace,
                name,
                flat: namespaced_name(namespace, name),
                tool,
            })
        })
}

pub(super) fn openai_tool_to_anthropic(t: &Value) -> Option<Value> {
    // `custom` (free-form input) and hosted tools have no Messages equivalent. Forwarded for the
    // provider to reject by name: dropping one would let the model answer without a tool the
    // client offered.
    if t.get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty != "function")
    {
        return Some(t.clone());
    }
    let func = t.get("function").unwrap_or(t);
    let name = func.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("name".into(), json!(name));
    if let Some(d) = func.get("description") {
        m.insert("description".into(), d.clone());
    }
    let schema = func
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    m.insert("input_schema".into(), schema);
    if func.get("strict").and_then(Value::as_bool) == Some(true) {
        m.insert("strict".into(), json!(true));
    }
    copy_cache_control(&mut m, t);
    if !m.contains_key("cache_control") {
        copy_cache_control(&mut m, func);
    }
    Some(Value::Object(m))
}

pub(super) fn openai_tool_choice_to_anthropic(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "none" => json!({ "type": "none" }),
            "required" => json!({ "type": "any" }),
            _ => json!({ "type": "auto" }),
        },
        Value::Object(o) => {
            // `allowed_tools` narrows which tools may be called this turn. Messages has no subset
            // (and trimming `tools` would rewrite the cached, thinking-bound prefix), so the mode
            // carries: `required` over one tool forces that tool, over several forces any tool.
            if let Some(allowed) = o.get("allowed_tools") {
                let tools = allowed.get("tools").and_then(Value::as_array);
                if allowed.get("mode").and_then(Value::as_str) != Some("required") {
                    return json!({ "type": "auto" });
                }
                return match tools.map(Vec::as_slice) {
                    Some([only]) => openai_tool_choice_to_anthropic(only),
                    _ => json!({ "type": "any" }),
                };
            }
            if let Some(name) = o
                .get("function")
                .or_else(|| o.get("custom"))
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .or_else(|| o.get("name").and_then(Value::as_str))
            {
                json!({ "type": "tool", "name": name })
            } else {
                json!({ "type": "auto" })
            }
        }
        _ => json!({ "type": "auto" }),
    }
}

/// Names of the deferred tools a `tool_result`'s `tool_reference` blocks loaded, anywhere in the
/// conversation.
pub(super) fn anthropic_tool_references(v: &Value) -> std::collections::HashSet<&str> {
    let mut names = std::collections::HashSet::new();
    let messages = v.get("messages").and_then(Value::as_array);
    for block in messages
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        .filter_map(|b| b.get("content").and_then(Value::as_array))
        .flatten()
    {
        if block.get("type").and_then(Value::as_str) == Some("tool_reference")
            && let Some(name) = block.get("tool_name").and_then(Value::as_str)
        {
            names.insert(name);
        }
    }
    names
}

pub(super) fn anthropic_tool_to_openai(t: &Value) -> Option<Value> {
    // Server tools (`web_search_…`, `code_execution_…`) and Anthropic-defined client tools
    // (`bash_…`, `text_editor_…`, `computer_…`) run a contract no Chat Completions function
    // carries: as an empty-schema function the model would call it and nothing would run it.
    // Forwarded as-is for the provider to reject by name.
    if t.get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty != "custom")
    {
        return Some(t.clone());
    }
    let name = t.get("name").and_then(Value::as_str)?;
    let mut func = Map::new();
    func.insert("name".into(), json!(name));
    if let Some(d) = t.get("description") {
        func.insert("description".into(), d.clone());
    }
    let params = t
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    // OpenAI's strict mode needs a stricter schema than Anthropic's (see `openai_strict_schema`).
    if t.get("strict").and_then(Value::as_bool) == Some(true) {
        func.insert("strict".into(), json!(openai_strict_schema(&params)));
    }
    func.insert("parameters".into(), params);
    let mut tool = json!({ "type": "function", "function": Value::Object(func) });
    if let Some(obj) = tool.as_object_mut() {
        copy_cache_control(obj, t);
    }
    Some(tool)
}

/// Whether OpenAI accepts `schema` under `strict: true`: every object sets
/// `additionalProperties: false` and lists all of its properties in `required`. Anthropic's
/// structured outputs allow optional properties, so a schema valid there is a 400 here with
/// `strict: true`; sent with `strict: false` it still shapes the output, without the guarantee.
pub(super) fn openai_strict_schema(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return true;
    };
    let is_object =
        obj.get("type").and_then(Value::as_str) == Some("object") || obj.contains_key("properties");
    if is_object {
        if obj.get("additionalProperties") != Some(&Value::Bool(false)) {
            return false;
        }
        let required: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if let Some(props) = obj.get("properties").and_then(Value::as_object)
            && !props.keys().all(|k| required.contains(&k.as_str()))
        {
            return false;
        }
    }
    let nested = |key: &str| -> Vec<&Value> {
        match obj.get(key) {
            Some(Value::Object(m)) if matches!(key, "properties" | "$defs" | "definitions") => {
                m.values().collect()
            }
            Some(Value::Array(a)) => a.iter().collect(),
            Some(v @ Value::Object(_)) => vec![v],
            _ => Vec::new(),
        }
    };
    [
        "properties",
        "$defs",
        "definitions",
        "items",
        "prefixItems",
        "anyOf",
        "oneOf",
        "allOf",
    ]
    .into_iter()
    .flat_map(nested)
    .all(openai_strict_schema)
}

pub(super) fn anthropic_tool_choice_to_openai(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("none") => json!("none"),
        Some("any") => json!("required"),
        Some("tool") => {
            let name = choice.get("name").and_then(Value::as_str).unwrap_or("");
            json!({ "type": "function", "function": { "name": name } })
        }
        _ => json!("auto"),
    }
}

/// OpenAI's hosted search (`web_search`, `web_search_preview`, and their dated spellings). Codex
/// offers it on every turn, by default over OpenAI's cached index (`external_web_access: false`),
/// which no other provider has. It is the model's to use or not and the client runs nothing for it,
/// so leaving Responses it is dropped (the model answers without searching) instead of forwarded
/// for the provider to reject, unless the `tool_choice` names it. Anthropic's server `web_search`
/// is not a faithful stand-in: it always fetches the live web (what Codex's default mode opts out
/// of), it is billed per search, and a Claude row's OpenRouter and Bedrock candidates would each
/// need their own spelling of it, so a failover would change what the tool does.
pub(super) fn is_hosted_web_search(t: &Value) -> bool {
    t.get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty.starts_with("web_search"))
}

/// Whether a Responses `tool_choice` names the hosted search: forced (`{"type": "web_search"}`)
/// or listed in `allowed_tools`.
pub(super) fn names_hosted_web_search(choice: &Value) -> bool {
    is_hosted_web_search(choice)
        || choice
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|ts| ts.iter().any(is_hosted_web_search))
}

/// A Responses tool → Chat Completions. `custom` (free-form input) has a Chat Completions shape;
/// hosted tools (`file_search`, `mcp`, …) do not, and are forwarded as-is for the provider to
/// reject by name. Dropping one would let the model answer without a tool the client offered. (The
/// hosted search never gets here unless chosen: see [`is_hosted_web_search`].)
pub(super) fn responses_tool_to_openai(t: &Value, wrap_custom: bool) -> Value {
    if t.get("function").is_some() || t.get("custom").is_some() {
        return t.clone();
    }
    let typ = t.get("type").and_then(Value::as_str).unwrap_or("function");
    let Some(name) = t.get("name").and_then(Value::as_str) else {
        return t.clone();
    };
    if typ == "custom" && wrap_custom {
        return wrapped_custom_tool(t, name);
    }
    let keys: &[&str] = match typ {
        "function" => &["description", "parameters", "strict"],
        "custom" => &["description"],
        _ => return t.clone(),
    };
    let mut inner = Map::new();
    inner.insert("name".into(), json!(name));
    for key in keys {
        copy_if(&mut inner, t, key);
    }
    if typ == "custom"
        && let Some(format) = t.get("format").filter(|f| !f.is_null())
    {
        inner.insert("format".into(), custom_format_to_chat(format));
    }
    let mut out = Map::new();
    out.insert("type".into(), json!(typ));
    out.insert(typ.into(), Value::Object(inner));
    copy_cache_control(&mut out, t);
    Value::Object(out)
}

/// A Chat Completions tool → Responses (flat). `cache_control` is Anthropic's and goes.
pub(super) fn openai_tool_to_responses(t: &Value) -> Option<Value> {
    let typ = t.get("type").and_then(Value::as_str).unwrap_or("function");
    let (inner, keys): (&Value, &[&str]) = match typ {
        "function" => (
            t.get("function").unwrap_or(t),
            &["description", "parameters", "strict"],
        ),
        "custom" => (t.get("custom").unwrap_or(t), &["description"]),
        // Hosted and Anthropic server tools: forwarded for the provider to reject by name.
        _ => return Some(t.clone()),
    };
    let name = inner.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("type".into(), json!(typ));
    m.insert("name".into(), json!(name));
    for key in keys {
        copy_if(&mut m, inner, key);
    }
    if typ == "custom"
        && let Some(format) = inner.get("format").filter(|f| !f.is_null())
    {
        m.insert("format".into(), custom_format_to_responses(format));
    }
    Some(Value::Object(m))
}

/// A `custom` tool's grammar format, Responses (`{type, syntax, definition}`) → Chat Completions
/// (`{type, grammar: {syntax, definition}}`). Copied flat, OpenAI 400s it ("Missing required
/// parameter: 'tools[3].custom.format.grammar'"). `text`, and a format already nested, pass.
fn custom_format_to_chat(format: &Value) -> Value {
    if format.get("type").and_then(Value::as_str) != Some("grammar")
        || format.get("grammar").is_some()
    {
        return format.clone();
    }
    let mut grammar = Map::new();
    copy_if(&mut grammar, format, "syntax");
    copy_if(&mut grammar, format, "definition");
    json!({ "type": "grammar", "grammar": grammar })
}

/// The inverse of [`custom_format_to_chat`].
fn custom_format_to_responses(format: &Value) -> Value {
    let Some(grammar) = format
        .get("grammar")
        .filter(|_| format.get("type").and_then(Value::as_str) == Some("grammar"))
    else {
        return format.clone();
    };
    let mut out = Map::new();
    out.insert("type".into(), json!("grammar"));
    copy_if(&mut out, grammar, "syntax");
    copy_if(&mut out, grammar, "definition");
    Value::Object(out)
}

/// A Responses `custom` tool bound for Messages, which has no free-form tool: a Chat Completions
/// function taking one `input` string, its description saying what the string must be (the
/// grammar, when the tool has one). A call to it comes back as a `custom_tool_call` (see
/// [`ToolNames`]); a replayed one already crosses as `tool_use.input: {input}`.
pub(super) fn wrapped_custom_tool(t: &Value, name: &str) -> Value {
    let mut desc = non_empty_str(t, "description").unwrap_or("").to_owned();
    if !desc.is_empty() {
        desc.push_str("\n\n");
    }
    let grammar = t
        .get("format")
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("grammar"))
        .map(|f| f.get("grammar").unwrap_or(f));
    match grammar.and_then(|g| non_empty_str(g, "definition")) {
        Some(definition) => {
            let syntax = grammar
                .and_then(|g| non_empty_str(g, "syntax"))
                .unwrap_or("lark");
            desc.push_str(&format!(
                "`input` is this tool's raw input text, not JSON. It must match this {syntax} \
                 grammar:\n```{syntax}\n{definition}\n```"
            ));
        }
        None => desc.push_str("`input` is this tool's raw, free-form input text."),
    }
    let mut out = json!({
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": { "input": { "type": "string" } },
                "required": ["input"],
            },
        },
    });
    if let Some(m) = out.as_object_mut() {
        copy_cache_control(m, t);
    }
    out
}

/// A tool reference inside `tool_choice` / `allowed_tools`, Responses (flat) → Chat (nested).
fn responses_tool_ref_to_openai(r: &Value) -> Value {
    match (
        r.get("type").and_then(Value::as_str),
        r.get("name").and_then(Value::as_str),
    ) {
        (Some(typ @ ("function" | "custom")), Some(name)) => {
            json!({ "type": typ, typ: { "name": name } })
        }
        _ => r.clone(),
    }
}

/// A tool reference, Chat (nested) → Responses (flat).
fn openai_tool_ref_to_responses(r: &Value) -> Value {
    match r.get("type").and_then(Value::as_str) {
        Some(typ @ ("function" | "custom")) => {
            match r
                .get(typ)
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            {
                Some(name) => json!({ "type": typ, "name": name }),
                None => r.clone(),
            }
        }
        _ => r.clone(),
    }
}

/// Responses `tool_choice` → Chat Completions. Strings are the same; a named tool nests its name,
/// and `allowed_tools` nests its mode and list. A hosted-tool choice (`{"type": "web_search"}`)
/// is forwarded for the provider to reject.
pub(super) fn responses_tool_choice_to_openai(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("allowed_tools") if choice.get("allowed_tools").is_none() => {
            let tools: Vec<Value> = choice
                .get("tools")
                .and_then(Value::as_array)
                .map(|ts| ts.iter().map(responses_tool_ref_to_openai).collect())
                .unwrap_or_default();
            json!({
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": choice.get("mode").cloned().unwrap_or(json!("auto")),
                    "tools": tools,
                },
            })
        }
        Some("function" | "custom") => responses_tool_ref_to_openai(choice),
        _ => choice.clone(),
    }
}

/// Chat Completions `tool_choice` → Responses: the inverse of [`responses_tool_choice_to_openai`].
pub(super) fn openai_tool_choice_to_responses(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("allowed_tools") => {
            let allowed = choice.get("allowed_tools").unwrap_or(choice);
            let tools: Vec<Value> = allowed
                .get("tools")
                .and_then(Value::as_array)
                .map(|ts| ts.iter().map(openai_tool_ref_to_responses).collect())
                .unwrap_or_default();
            json!({
                "type": "allowed_tools",
                "mode": allowed.get("mode").cloned().unwrap_or(json!("auto")),
                "tools": tools,
            })
        }
        Some("function" | "custom") => openai_tool_ref_to_responses(choice),
        _ => choice.clone(),
    }
}
