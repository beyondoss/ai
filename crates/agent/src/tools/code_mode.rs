//! Code Mode — one confined JS `execute` tool over deferred MCP tools.
//!
//! OpenCode/Cloudflare-style: the model writes a small JavaScript program; a QuickJS-NG runtime
//! (`rquickjs`) runs it against an explicit tree of schema-described tools. Built-in coding tools
//! (`read`/`edit`/`bash`/…) stay direct. MCP tools are the deferred catalog: they are not advertised
//! as `mcp__server__tool` entries, and are reachable only as `tools.<server>.<method>(input)` inside
//! the program.
//!
//! Authority stays with the host [`Tool`] implementations. The JS context has no filesystem, network,
//! process, or module loader — only the injected `tools` tree (and `tools.search`). Nested calls still
//! honor `--exclude-tools` / `--deny-tool` via [`select_deferred_tools`].
//!
//! Off by default (`--code-mode` / `AI_AGENT_CODE_MODE`).

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use agent_core::tool::{Tool, ToolProgress};
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use rquickjs::function::Opt;
use rquickjs::prelude::Async;
use rquickjs::promise::Promise;
use rquickjs::{AsyncContext, AsyncRuntime, CatchResultExt, Ctx, Function};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// Advertised / registered name of the Code Mode tool.
pub const NAME: &str = "execute";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_CALLS: usize = 64;
const DEFAULT_MAX_OUTPUT: usize = 256 * 1024;
const DEFAULT_MEMORY_LIMIT: usize = 32 * 1024 * 1024;
const DEFAULT_STACK_SIZE: usize = 512 * 1024;

/// One deferred tool as the JS catalog sees it.
#[derive(Debug, Clone, serde::Serialize)]
struct CatalogEntry {
    /// Registered host name (`mcp__github__list_issues`).
    name: String,
    /// JS path segments (`["github", "list_issues"]`).
    path: Vec<String>,
    description: String,
    signature: String,
}

/// Budgets for one `execute` run. Production uses [`Limits::production`]; tests tighten them.
#[derive(Debug, Clone)]
pub struct Limits {
    pub timeout: Duration,
    pub max_calls: usize,
    pub max_output: usize,
    pub memory_limit: usize,
}

impl Limits {
    fn production() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            max_calls: DEFAULT_MAX_CALLS,
            max_output: DEFAULT_MAX_OUTPUT,
            memory_limit: DEFAULT_MEMORY_LIMIT,
        }
    }
}

/// The model-facing Code Mode tool.
pub struct Execute {
    tools: Arc<BTreeMap<String, Arc<dyn Tool>>>,
    catalog: Arc<Vec<CatalogEntry>>,
    description: String,
    limits: Limits,
}

impl Execute {
    /// Build from the already-filtered deferred catalog (typically MCP tools).
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self::with_limits(tools, Limits::production())
    }

    fn with_limits(tools: Vec<Arc<dyn Tool>>, limits: Limits) -> Self {
        let mut map = BTreeMap::new();
        let mut catalog = Vec::new();
        for tool in tools {
            let name = tool.name().to_string();
            let path = js_path(&name);
            catalog.push(CatalogEntry {
                name: name.clone(),
                path,
                description: tool.description().to_string(),
                signature: compact_signature(&tool.input_schema()),
            });
            map.insert(name, tool);
        }
        let description = build_description(&catalog);
        Self {
            tools: Arc::new(map),
            catalog: Arc::new(catalog),
            description,
            limits,
        }
    }

    async fn execute(&self, code: &str, progress: &ToolProgress) -> Result<String, ToolError> {
        if code.trim().is_empty() {
            return Err(ToolError::InvalidInput("`code` must not be empty".into()));
        }
        let timeout = self.limits.timeout;
        let deadline = std::time::Instant::now() + timeout;
        let timed_out = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));

        let tools = self.tools.clone();
        let catalog = self.catalog.clone();
        let max_calls = self.limits.max_calls;
        let max_output = self.limits.max_output;
        let memory_limit = self.limits.memory_limit;
        let code = code.to_string();
        let progress = progress.clone();

        let run = run_js(
            tools,
            catalog,
            code,
            calls,
            max_calls,
            memory_limit,
            deadline,
            timed_out.clone(),
            cancelled.clone(),
            progress.clone(),
        );

        tokio::select! {
            biased;
            () = progress.cancelled() => {
                cancelled.store(true, Ordering::Relaxed);
                Err(ToolError::Execution("code mode cancelled".into()))
            }
            result = run => {
                if timed_out.load(Ordering::Relaxed) {
                    Err(ToolError::Execution(format!(
                        "code mode timed out after {}ms",
                        timeout.as_millis()
                    )))
                } else {
                    result
                }
            }
        }
        .map(|text| bound_output(text, max_output))
    }
}

/// Drop MCP tools the operator excluded or denied, so they are unreachable through `execute` too.
pub fn select_deferred_tools(
    mcp_tools: &[Arc<dyn Tool>],
    exclude: &[String],
    deny: &[String],
) -> Vec<Arc<dyn Tool>> {
    let drop: HashSet<&str> = exclude.iter().chain(deny).map(String::as_str).collect();
    mcp_tools
        .iter()
        .filter(|t| !drop.contains(t.name()))
        .cloned()
        .collect()
}

/// After `--tools` filtering, put `execute` back unless the operator excluded it or asked for no tools.
///
/// An allow-list of built-ins (`--tools read,bash`) would otherwise drop `execute` and leave the
/// deferred MCP catalog unreachable — the whole point of `--code-mode`.
pub fn restore_execute(
    registry: &mut agent_core::ToolRegistry,
    execute: Arc<dyn Tool>,
    exclude: Option<&[String]>,
    no_tools: bool,
) {
    if no_tools {
        return;
    }
    if exclude.is_some_and(|e| e.iter().any(|n| n == NAME)) {
        return;
    }
    if registry.get(NAME).is_none() {
        registry.register(execute);
    }
}

#[async_trait]
impl Tool for Execute {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript program. `tools.<server>.<method>(input)` calls a deferred MCP tool and returns a Promise. Independent calls can run together via Promise.all. Return the value the model should see."
                }
            },
            "required": ["code"],
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let (tx, _rx) = futures::channel::mpsc::unbounded();
        let progress =
            ToolProgress::new(tx, "execute".into(), NAME.into(), CancellationToken::new());
        self.run_streaming(input, &progress).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let code = input
            .get("code")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `code` string".into()))?;
        let text = self.execute(code, progress).await?;
        Ok(ToolOutput::text(text))
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_js(
    tools: Arc<BTreeMap<String, Arc<dyn Tool>>>,
    catalog: Arc<Vec<CatalogEntry>>,
    code: String,
    calls: Arc<AtomicUsize>,
    max_calls: usize,
    memory_limit: usize,
    deadline: std::time::Instant,
    timed_out: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    progress: ToolProgress,
) -> Result<String, ToolError> {
    let catalog_json = serde_json::to_string(catalog.as_ref())
        .map_err(|e| ToolError::Execution(format!("code mode catalog serialize failed: {e}")))?;
    let script = wrap_program(&code);

    let rt = AsyncRuntime::new().map_err(js_err)?;
    rt.set_memory_limit(memory_limit).await;
    rt.set_max_stack_size(DEFAULT_STACK_SIZE).await;
    rt.set_interrupt_handler(Some(Box::new({
        let timed_out = timed_out.clone();
        let cancelled = cancelled.clone();
        move || {
            if cancelled.load(Ordering::Relaxed) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                timed_out.store(true, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
    })))
    .await;

    let ctx = AsyncContext::full(&rt).await.map_err(js_err)?;

    ctx.async_with(async move |ctx| -> Result<String, ToolError> {
        install_bridge(&ctx, tools, calls, max_calls, progress)?;
        ctx.globals()
            .set("__catalogJson", catalog_json)
            .map_err(js_err)?;
        let promise: Promise<'_> = ctx.eval(script).map_err(js_err)?;
        promise
            .into_future::<String>()
            .await
            .catch(&ctx)
            .map_err(|e| ToolError::Execution(format!("code mode: {e}")))
    })
    .await
}

fn install_bridge<'js>(
    ctx: &Ctx<'js>,
    tools: Arc<BTreeMap<String, Arc<dyn Tool>>>,
    calls: Arc<AtomicUsize>,
    max_calls: usize,
    progress: ToolProgress,
) -> Result<(), ToolError> {
    let func = Function::new(
        ctx.clone(),
        Async(move |name: String, input: Opt<String>| {
            let tools = tools.clone();
            let calls = calls.clone();
            let progress = progress.clone();
            async move {
                Ok::<String, rquickjs::Error>(
                    invoke_nested(
                        tools,
                        calls,
                        max_calls,
                        name,
                        input.0.unwrap_or_else(|| "{}".into()),
                        progress,
                    )
                    .await,
                )
            }
        }),
    )
    .map_err(js_err)?;
    ctx.globals().set("__callTool", func).map_err(js_err)?;
    Ok(())
}

async fn invoke_nested(
    tools: Arc<BTreeMap<String, Arc<dyn Tool>>>,
    calls: Arc<AtomicUsize>,
    max_calls: usize,
    name: String,
    input_json: String,
    progress: ToolProgress,
) -> String {
    if progress.is_cancelled() {
        return envelope("code mode cancelled", true);
    }
    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
    if n > max_calls {
        return envelope(
            format!("code mode exceeded max nested tool calls ({max_calls})"),
            true,
        );
    }
    let Some(tool) = tools.get(&name).cloned() else {
        return envelope(format!("unknown tool: {name}"), true);
    };
    let input = serde_json::from_str(&input_json).unwrap_or_else(|_| json!({}));
    progress.emit(format!("{name} ({n})"), None);
    match tool.run(input).await {
        Ok(out) => envelope(out.text, false),
        Err(e) => envelope(e.to_string(), true),
    }
}

fn envelope(text: impl Into<String>, is_error: bool) -> String {
    serde_json::to_string(&json!({ "text": text.into(), "isError": is_error }))
        .unwrap_or_else(|_| "{\"text\":\"internal json error\",\"isError\":true}".to_string())
}

fn wrap_program(user_code: &str) -> String {
    // User code is the body of an async function. A syntax error surfaces as a Code Mode failure.
    format!(
        r#"(async () => {{
  const catalog = JSON.parse(globalThis.__catalogJson);
  function call(name, input) {{
    return globalThis.__callTool(name, JSON.stringify(input ?? {{}})).then((raw) => {{
      const parsed = JSON.parse(raw);
      if (parsed.isError) {{
        const err = new Error(parsed.text);
        err.name = "ToolError";
        throw err;
      }}
      const text = parsed.text;
      if (typeof text !== "string") return text;
      try {{ return JSON.parse(text); }} catch {{ return text; }}
    }});
  }}
  function makeTools(entries) {{
    const root = Object.create(null);
    for (const e of entries) {{
      let obj = root;
      for (let i = 0; i < e.path.length - 1; i++) {{
        const p = e.path[i];
        if (obj[p] == null) obj[p] = Object.create(null);
        obj = obj[p];
      }}
      obj[e.path[e.path.length - 1]] = (input) => call(e.name, input);
    }}
    root.search = (q) => {{
      const needle = String(q ?? "").toLowerCase();
      return entries
        .filter((e) =>
          e.name.toLowerCase().includes(needle)
          || e.path.join(".").toLowerCase().includes(needle)
          || String(e.description || "").toLowerCase().includes(needle)
        )
        .map((e) => ({{
          path: e.path.join("."),
          name: e.name,
          description: e.description,
          signature: e.signature,
        }}));
    }};
    return root;
  }}
  const tools = makeTools(catalog);
  const __fn = async () => {{
{user_code}
  }};
  const __result = await __fn();
  if (__result === undefined || __result === null) return "";
  return typeof __result === "string" ? __result : JSON.stringify(__result);
}})()
"#
    )
}

fn build_description(catalog: &[CatalogEntry]) -> String {
    let mut lines = Vec::new();
    lines.push(
        "Run a JavaScript program that composes deferred MCP tools. Built-in tools (read, write, edit, bash, …) stay direct — call those normally. Inside `code`, tools are `tools.<server>.<method>(input)` Promises. Use Promise.all for independent calls. Filter/aggregate in the program and return only what you need. tools.search(query) lists matching deferred tools."
            .into(),
    );
    if catalog.is_empty() {
        lines.push("No deferred tools are currently available.".into());
        return lines.join("\n");
    }
    lines.push("Deferred catalog:".into());
    const BUDGET: usize = 4_000;
    let mut used = lines.iter().map(String::len).sum::<usize>();
    let mut shown = 0usize;
    for entry in catalog {
        let line = format!(
            "- tools.{}({}) — {}",
            entry.path.join("."),
            entry.signature,
            truncate(&entry.description, 120)
        );
        if used + line.len() + 1 > BUDGET {
            break;
        }
        used += line.len() + 1;
        shown += 1;
        lines.push(line);
    }
    if shown < catalog.len() {
        lines.push(format!(
            "… {} more; call tools.search(query) for the rest.",
            catalog.len() - shown
        ));
    }
    lines.join("\n")
}

fn compact_signature(schema: &Value) -> String {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return String::new();
    };
    let required: HashSet<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut parts: Vec<String> = props
        .keys()
        .map(|k| {
            if required.contains(k.as_str()) {
                k.clone()
            } else {
                format!("{k}?")
            }
        })
        .collect();
    parts.sort();
    parts.join(", ")
}

/// `mcp__github__list_issues` → `["github", "list_issues"]`.
fn js_path(registered: &str) -> Vec<String> {
    let rest = registered.strip_prefix("mcp__").unwrap_or(registered);
    let parts: Vec<String> = rest
        .split("__")
        .filter(|p| !p.is_empty())
        .map(js_ident)
        .collect();
    if parts.is_empty() {
        vec![js_ident(registered)]
    } else {
        parts
    }
}

fn js_ident(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if i == 0 && c.is_ascii_digit() {
                out.push('_');
            }
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return "_".into();
    }
    match out.as_str() {
        "search" | "constructor" | "__proto__" | "prototype" | "tools" => format!("_{out}"),
        _ => out,
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(c);
    }
    out
}

fn bound_output(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let mut cut = max;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}\n… [truncated, {} bytes total]",
        &text[..cut],
        text.len()
    )
}

fn js_err<E: std::fmt::Display>(e: E) -> ToolError {
    ToolError::Execution(format!("code mode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fake {
        name: &'static str,
        desc: &'static str,
        handler: fn(Value) -> Result<ToolOutput, ToolError>,
    }

    #[async_trait]
    impl Tool for Fake {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.desc
        }
        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "q": { "type": "string" }, "n": { "type": "number" } },
                "required": ["q"],
            })
        }
        async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
            (self.handler)(input)
        }
    }

    fn exec(tools: Vec<Arc<dyn Tool>>) -> Execute {
        Execute::with_limits(
            tools,
            Limits {
                timeout: Duration::from_secs(5),
                max_calls: 16,
                max_output: 16 * 1024,
                memory_limit: 8 * 1024 * 1024,
            },
        )
    }

    async fn run_code(tool: &Execute, code: &str) -> Result<String, ToolError> {
        Ok(tool.run(json!({ "code": code })).await?.text)
    }

    #[test]
    fn js_path_splits_mcp_names() {
        assert_eq!(
            js_path("mcp__github__list_issues"),
            vec!["github", "list_issues"]
        );
        assert_eq!(
            js_path("mcp__fs__resource__notes"),
            vec!["fs", "resource", "notes"]
        );
        assert_eq!(js_path("echo"), vec!["echo"]);
    }

    #[test]
    fn js_ident_sanitizes() {
        assert_eq!(js_ident("list-issues"), "list_issues");
        assert_eq!(js_ident("search"), "_search");
        assert_eq!(js_ident("2fa"), "_2fa");
    }

    #[test]
    fn select_deferred_tools_drops_excluded_and_denied() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(Fake {
                name: "mcp__a__one",
                desc: "one",
                handler: |_| Ok("1".into()),
            }),
            Arc::new(Fake {
                name: "mcp__a__two",
                desc: "two",
                handler: |_| Ok("2".into()),
            }),
        ];
        let kept = select_deferred_tools(&tools, &["mcp__a__one".into()], &["mcp__a__two".into()]);
        assert!(kept.is_empty());
        let kept = select_deferred_tools(&tools, &["mcp__a__one".into()], &[]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].name(), "mcp__a__two");
    }

    #[tokio::test]
    async fn execute_calls_a_nested_tool() {
        let tool = exec(vec![Arc::new(Fake {
            name: "mcp__echo__shout",
            desc: "shout",
            handler: |input| {
                let q = input.get("q").and_then(Value::as_str).unwrap_or("");
                Ok(q.to_uppercase().into())
            },
        })]);
        let out = run_code(&tool, r#"return await tools.echo.shout({q: "hi"});"#)
            .await
            .unwrap();
        assert_eq!(out, "HI");
    }

    #[tokio::test]
    async fn execute_promise_all_and_filter() {
        let tool = exec(vec![Arc::new(Fake {
            name: "mcp__nums__get",
            desc: "get",
            handler: |input| {
                let n = input.get("n").and_then(Value::as_u64).unwrap_or(0);
                Ok(serde_json::to_string(&json!({"n": n})).unwrap().into())
            },
        })]);
        let out = run_code(
            &tool,
            r#"
            const rows = await Promise.all([1, 2, 3, 4].map((n) => tools.nums.get({q: "x", n})));
            return rows.map((r) => r.n).filter((n) => n % 2 === 0);
            "#,
        )
        .await
        .unwrap();
        assert_eq!(out, "[2,4]");
    }

    #[tokio::test]
    async fn execute_unknown_tool_is_a_tool_error() {
        let tool = exec(vec![Arc::new(Fake {
            name: "mcp__echo__shout",
            desc: "shout",
            handler: |_| Ok("x".into()),
        })]);
        let err = run_code(&tool, r#"return await tools.nope.missing({});"#)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("TypeError") || msg.contains("undefined") || msg.contains("code mode"),
            "{msg}"
        );
    }

    #[tokio::test]
    async fn execute_search_lists_catalog() {
        let tool = exec(vec![Arc::new(Fake {
            name: "mcp__github__list_issues",
            desc: "list github issues",
            handler: |_| Ok("[]".into()),
        })]);
        let out = run_code(&tool, r#"return tools.search("github");"#)
            .await
            .unwrap();
        assert!(out.contains("github.list_issues"), "{out}");
        assert!(out.contains("mcp__github__list_issues"), "{out}");
    }

    #[tokio::test]
    async fn execute_does_not_expose_host_builtins() {
        let tool = exec(Vec::new());
        let err = run_code(&tool, r#"return typeof os;"#).await.unwrap();
        assert_eq!(err, "undefined");
        let err = run_code(&tool, r#"return typeof std;"#).await.unwrap();
        assert_eq!(err, "undefined");
    }

    #[tokio::test]
    async fn execute_times_out_an_infinite_loop() {
        let tool = Execute::with_limits(
            Vec::new(),
            Limits {
                timeout: Duration::from_millis(200),
                max_calls: 1,
                max_output: 1024,
                memory_limit: 8 * 1024 * 1024,
            },
        );
        let err = run_code(&tool, "while (true) { await Promise.resolve(); }")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{}", err.to_string());
    }

    #[tokio::test]
    async fn execute_caps_nested_calls() {
        let tool = Execute::with_limits(
            vec![Arc::new(Fake {
                name: "mcp__echo__shout",
                desc: "shout",
                handler: |_| Ok("ok".into()),
            })],
            Limits {
                timeout: Duration::from_secs(5),
                max_calls: 2,
                max_output: 1024,
                memory_limit: 8 * 1024 * 1024,
            },
        );
        let err = run_code(
            &tool,
            r#"
            await tools.echo.shout({q: "a"});
            await tools.echo.shout({q: "b"});
            return await tools.echo.shout({q: "c"});
            "#,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("max nested"),
            "{}",
            err.to_string()
        );
    }

    #[test]
    fn restore_execute_keeps_the_tool_unless_excluded() {
        let mut reg = agent_core::ToolRegistry::new();
        let execute: Arc<dyn Tool> = Arc::new(exec(Vec::new()));
        restore_execute(&mut reg, execute.clone(), None, false);
        assert!(reg.get(NAME).is_some());
        let mut reg = agent_core::ToolRegistry::new();
        restore_execute(&mut reg, execute.clone(), Some(&["execute".into()]), false);
        assert!(reg.get(NAME).is_none());
        let mut reg = agent_core::ToolRegistry::new();
        restore_execute(&mut reg, execute, None, true);
        assert!(reg.get(NAME).is_none());
    }
}
