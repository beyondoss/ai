//! Startup checks for the tool-selection flags (`--tools`, `--exclude-tools`, `--deny-tool`,
//! `--no-tools`), shared by `run` and `serve`.
//!
//! A filter list is matched by exact name ([`super::filter_allows`]), so a typo matches nothing:
//! `--tools raed,bash` silently ran without `read`, and `--exclude-tools bsh` excluded nothing. Every
//! name is checked against the built-in tools here, before anything runs.
//!
//! MCP tools are the exception: they exist only once their server connects, after these checks. Their
//! names always carry the `mcp__<server>__` prefix ([`super::mcp`]), so any name with the `mcp__`
//! prefix is accepted as is. A typo there filters nothing, as before; a typo anywhere else is
//! refused with the closest built-in name.

/// Every built-in tool a selection flag may name: the default set plus the ones registered only in
/// some configurations (`web` is opt-in, `memory` needs a memory store, `subagent` and
/// `structured_output` are added per run, `tool_search` fronts deferred MCP tools, `execute` is Code
/// Mode's). A test keeps this in step with the default registry.
pub const BUILTIN_TOOLS: &[&str] = &[
    "read",
    "write",
    "edit",
    "bash",
    super::bash::JOB_TOOL,
    "ls",
    "grep",
    "find",
    super::todo::NAME,
    super::web::NAME,
    super::memory::NAME,
    super::subagent::NAME,
    super::structured_output::NAME,
    super::tool_search::NAME,
    super::code_mode::NAME,
];

/// Refuse a contradictory pair of tool flags, or a name in any list that is neither a built-in tool
/// nor an MCP tool name. The error names the flag, the unknown name and the closest built-in.
pub fn check_tool_flags(
    tools: Option<&[String]>,
    exclude: Option<&[String]>,
    deny: &[String],
    no_tools: bool,
) -> Result<(), String> {
    if no_tools {
        if tools.is_some() {
            return Err(
                "--no-tools and --tools contradict each other: --no-tools registers no tools, \
                 --tools picks which ones to register"
                    .into(),
            );
        }
        if exclude.is_some() {
            return Err(
                "--no-tools and --exclude-tools contradict each other: with --no-tools there is \
                 nothing to exclude"
                    .into(),
            );
        }
    }
    check_names("--tools", tools.unwrap_or_default())?;
    check_names("--exclude-tools", exclude.unwrap_or_default())?;
    check_names("--deny-tool", deny)
}

fn check_names(flag: &str, names: &[String]) -> Result<(), String> {
    for name in names {
        if name.starts_with("mcp__") || BUILTIN_TOOLS.contains(&name.as_str()) {
            continue;
        }
        let hint = closest(name, BUILTIN_TOOLS)
            .map(|c| format!(" (did you mean `{c}`?)"))
            .unwrap_or_default();
        return Err(format!(
            "{flag}: unknown tool `{name}`{hint}. Built-in tools: {}. MCP tools are named \
             mcp__<server>__<tool>.",
            BUILTIN_TOOLS.join(", ")
        ));
    }
    Ok(())
}

/// The candidate closest to `name` by edit distance, if it is close enough to be a likely typo
/// (at most a third of the name's length, and at least 1).
pub fn closest<'a>(name: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).max(1);
    candidates
        .iter()
        .map(|c| (edit_distance(name, c), *c))
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// Optimal-string-alignment distance: insertions, deletions, substitutions and adjacent
/// transpositions (`raed` → `read` is one edit).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_typo_is_refused_with_the_closest_built_in() {
        let err = check_tool_flags(Some(&s(&["raed", "bash"])), None, &[], false).unwrap_err();
        assert!(
            err.starts_with("--tools: unknown tool `raed` (did you mean `read`?)"),
            "{err}"
        );
        let err = check_tool_flags(None, Some(&s(&["bsh"])), &[], false).unwrap_err();
        assert!(
            err.contains("--exclude-tools: unknown tool `bsh` (did you mean `bash`?)"),
            "{err}"
        );
        let err = check_tool_flags(None, None, &s(&["wirte"]), false).unwrap_err();
        assert!(
            err.contains("--deny-tool: unknown tool `wirte` (did you mean `write`?)"),
            "{err}"
        );
        let err = check_tool_flags(Some(&s(&["frobnicate"])), None, &[], false).unwrap_err();
        assert!(
            !err.contains("did you mean"),
            "no close match, no guess: {err}"
        );
    }

    #[test]
    fn built_ins_and_mcp_names_are_accepted() {
        check_tool_flags(
            Some(&s(&[
                "read",
                "web",
                "bash_job",
                "tool_search",
                "mcp__github__get_issue",
            ])),
            Some(&s(&["write"])),
            &s(&["mcp__fs__write_file", "bash"]),
            false,
        )
        .unwrap();
        check_tool_flags(None, None, &[], true).unwrap();
    }

    #[test]
    fn no_tools_with_a_tool_list_is_refused() {
        let err = check_tool_flags(Some(&s(&["read"])), None, &[], true).unwrap_err();
        assert!(err.contains("--no-tools and --tools"), "{err}");
        let err = check_tool_flags(None, Some(&s(&["bash"])), &[], true).unwrap_err();
        assert!(err.contains("--no-tools and --exclude-tools"), "{err}");
    }

    #[test]
    fn every_default_tool_is_a_known_built_in() {
        let reg = super::super::default_registry();
        for name in reg.names() {
            assert!(
                BUILTIN_TOOLS.contains(&name.as_str()),
                "{name} is registered by default but missing from BUILTIN_TOOLS"
            );
        }
    }
}
