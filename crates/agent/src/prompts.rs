//! Prompt templates — reusable `/name args` expansions.
//!
//! A template is a Markdown file under `~/.claude/prompts` (user) or `<cwd>/.claude/prompts`
//! (project). When a prompt message begins with `/name ...`, the matching template's body is expanded
//! with bash-style argument substitution and sent to the model in place of the slash line.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::skills::Collision;

/// Cap on a single prompt template file's own size, checked (via `fs::metadata`, not a read) before
/// [`load_file`] ever calls `fs::read_to_string` on it — same reasoning as `skills.rs`'s own
/// `MAX_SKILL_FILE_LEN`: a legitimate template is a short YAML frontmatter block plus at most a few KB
/// of body text, so a cap this far above that only ever catches a pathological file (e.g. a data dump
/// misnamed `.md`), not a verbose but legitimate one.
const MAX_PROMPT_TEMPLATE_FILE_LEN: u64 = 1024 * 1024; // 1 MiB

/// A discovered prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    /// Invoke name (the file stem), used as `/name`.
    pub name: String,
    /// One-line hint for arguments, from `argument-hint:` frontmatter (if any).
    pub argument_hint: Option<String>,
    /// One-line summary for autocomplete: `description:` frontmatter, else the first non-empty body
    /// line (truncated). Always populated so a caller can show *something* per command.
    pub description: String,
    /// The template body (frontmatter stripped).
    pub body: String,
    /// Absolute path to the template's own `.md` file. Surfaced via `serve`'s `get_commands` (Task #39
    /// pi-parity fix: previously omitted entirely, even though `discover_with_diagnostics_impl` already
    /// tracked it internally for collision reporting).
    pub path: PathBuf,
    /// Which discovery root this template actually came from — `"user"` (`~/.claude/prompts`),
    /// `"project"` (`<cwd>/.claude/prompts`), or `"temporary"` (an operator-supplied
    /// `--prompt-template` extra root) — matching pi's own `SourceScope` (`source-info.ts`) and
    /// [`crate::skills::Skill::scope`]'s identical convention. Set once per discovery-root group in
    /// `discover_with_diagnostics_impl`, not by `load_file`/`load_dir` themselves (neither has any
    /// notion of which root it was reached from) — every constructor of a fresh `PromptTemplate` picks
    /// *some* placeholder value, always overwritten immediately after by that per-root pass.
    pub scope: &'static str,
}

/// Discover prompt templates under the user root, plus the project root when `project_trusted`;
/// project shadows user by name.
///
/// The user root is **never** gated on `project_trusted`, mirroring [`crate::skills::discover`]: it's
/// the operator's own machine-wide directory, not something the current (possibly untrusted) project
/// checkout controls. `project_trusted` is a required parameter (not a bool a call site can forget to
/// pass, or default away) precisely because it's load-bearing for the trust-injection fix this module
/// shares with `skills.rs`: a project's own `.claude/prompts/*.md` is untrusted repo content, and its
/// body is injected into the model's context the moment `/name` is invoked (see `expand_if_slash`) —
/// skipping this gate would silently reopen exactly the prompt-injection vector already closed for
/// skills.
pub fn discover(cwd: &Path, project_trusted: bool, extra_roots: &[String]) -> Vec<PromptTemplate> {
    discover_with_diagnostics(cwd, project_trusted, extra_roots).0
}

/// Like [`discover`], but also reports name collisions — the same `/name` defined by more than one
/// template file, silently shadowed by `discover` (the later standard root wins) — as structured
/// [`Collision`]s naming both paths, for `get_commands` to surface as a diagnostic rather than a client
/// having no way to notice a template was shadowed.
///
/// `extra_roots` are additional, ad-hoc discovery roots beyond the two standard ones — pi's own
/// `--prompt-template <path>` (repeatable), which accepts either a directory (walked like a standard
/// root) or a single standalone `.md` file (one template, no directory of its own — pi's other
/// `--prompt-template` shape, `prompt-templates.ts`'s `stats.isFile() && resolvedPath.endsWith(".md")`);
/// see `skills::discover_with_diagnostics`'s identical parameter for the missing-path-diagnostic and
/// standard-root-always-wins-over-an-extra-one shadow ordering, which applies here unchanged.
pub fn discover_with_diagnostics(
    cwd: &Path,
    project_trusted: bool,
    extra_roots: &[String],
) -> (Vec<PromptTemplate>, Vec<Collision>) {
    discover_with_diagnostics_impl(cwd, project_trusted, extra_roots, true)
}

/// Like [`discover_with_diagnostics`], but skips *both* standard roots (`~/.claude/prompts` and
/// `<cwd>/.claude/prompts`) entirely, keeping only `extra_roots` — what `--no-prompt-templates` needs,
/// mirroring [`crate::skills::discover_extra_only`]'s identical reasoning: pi's own `noPromptTemplates`
/// still honors an explicit `--prompt-template` path passed alongside it (pi-parity fix, M2).
pub fn discover_extra_only(extra_roots: &[String]) -> (Vec<PromptTemplate>, Vec<Collision>) {
    discover_with_diagnostics_impl(Path::new(""), false, extra_roots, false)
}

fn discover_with_diagnostics_impl(
    cwd: &Path,
    project_trusted: bool,
    extra_roots: &[String],
    include_standard_roots: bool,
) -> (Vec<PromptTemplate>, Vec<Collision>) {
    let mut found: Vec<PromptTemplate> = Vec::new();
    let mut origins: HashMap<String, PathBuf> = HashMap::new();
    let mut collisions: Vec<Collision> = Vec::new();
    // Paired with the `scope` (Task #39 — `"user"`/`"project"`, matching pi's own `SourceScope`) each
    // root's own templates get tagged with — `load_file`/`load_dir` have no notion of which root they
    // were reached from, so it's set once per root group below instead.
    let mut standard_roots: Vec<(PathBuf, &'static str)> = Vec::new();
    // Canonical form of every root added so far, across `standard_roots` and the extra-root loop
    // below — a root reached by two different paths (a symlink, a relative-vs-absolute spelling, `cwd`
    // itself equaling `~/.claude/prompts` in some deployment) must only be walked once, or its
    // templates would double-count and self-collide against a phantom duplicate rather than a genuine
    // collision. See `path_utils::push_unique_scoped_root`'s own doc comment.
    let mut seen_roots: HashSet<PathBuf> = HashSet::new();
    use crate::path_utils::push_unique_scoped_root as push_scoped;
    if include_standard_roots {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            push_scoped(
                &mut standard_roots,
                &mut seen_roots,
                home.join(".claude/prompts"),
                "user",
            );
        }
        if project_trusted {
            push_scoped(
                &mut standard_roots,
                &mut seen_roots,
                cwd.join(".claude/prompts"),
                "project",
            );
        }
    }

    // Both silent-drop paths — an oversized file (checked via `fs::metadata` and skipped *before* any
    // read, see `MAX_PROMPT_TEMPLATE_FILE_LEN`) and an unreadable one — are reported through
    // `diagnostics` (folded into the `Vec<Collision>` `discover_with_diagnostics` surfaces to a caller,
    // wrapped as a message-only `Collision`) and `tracing::warn!`-logged at the point of detection,
    // matching `skills::parse_skill`'s identical handling of the same two drop paths. Previously this
    // was a bare `fs::read_to_string(path).ok()?` — a read failure (permissions, non-UTF8, …) vanished
    // with zero diagnostic, unlike the extra-single-file-root branch below, which already warns.
    fn load_file(
        path: &Path,
        diagnostics: &mut Vec<String>,
    ) -> Option<(String, PathBuf, PromptTemplate)> {
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned())?;
        if let Ok(meta) = fs::metadata(path) {
            if meta.len() > MAX_PROMPT_TEMPLATE_FILE_LEN {
                let message = format!(
                    "prompt template {} exceeds {MAX_PROMPT_TEMPLATE_FILE_LEN} bytes ({} bytes) — \
                     skipped without reading",
                    path.display(),
                    meta.len()
                );
                tracing::warn!("{message}");
                diagnostics.push(message);
                return None;
            }
        }
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) => {
                let message = format!("failed to read prompt template {}: {err}", path.display());
                tracing::warn!("{message}");
                diagnostics.push(message);
                return None;
            }
        };
        let (hint, description, body) = parse(&text);
        Some((
            name.clone(),
            path.to_path_buf(),
            PromptTemplate {
                name,
                argument_hint: hint,
                description,
                body,
                path: path.to_path_buf(),
                // Always overwritten by the caller — see `PromptTemplate::scope`'s own doc comment.
                scope: "temporary",
            },
        ))
    }

    // Gitignore-aware, matching `skills.rs::walk` — pi's own `collectAutoPromptEntries` applies its
    // `IGNORE_FILE_NAMES` (`.gitignore`/`.ignore`/`.fdignore`) here too via `addIgnoreRules`; this had
    // no `ignore`-crate usage at all before, so a `.claude/prompts/.gitignore` entry that pi would skip
    // (e.g. `draft-*.md` hiding a work-in-progress template) was silently still loaded and exposed as
    // `/name` here. `max_depth(Some(1))` keeps this the same single-directory scan pi's own
    // `readdirSync` does (prompt templates, unlike skills, are never nested in a subdirectory of their
    // own) — only `root`'s immediate children are considered, not an arbitrary-depth walk.
    // `git_global`/`git_exclude` are disabled for the same reason `skills.rs` disables them: pi never
    // consults a global excludes file or `.git/info/exclude`, only the three per-directory ignore files.
    fn load_dir(
        root: &Path,
        diagnostics: &mut Vec<String>,
    ) -> Vec<(String, PathBuf, PromptTemplate)> {
        let mut out = Vec::new();
        let paths: Vec<PathBuf> = ignore::WalkBuilder::new(root)
            .max_depth(Some(1))
            .follow_links(true)
            .add_custom_ignore_filename(".fdignore")
            .require_git(false)
            .git_global(false)
            .git_exclude(false)
            .filter_entry(|entry| entry.file_name() != "node_modules")
            .build()
            .flatten() // missing/unreadable/inaccessible entries are the normal case, not an error
            .filter(|entry| entry.depth() > 0 && entry.file_type().is_some_and(|t| t.is_file()))
            .map(ignore::DirEntry::into_path)
            .collect();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if let Some(entry) = load_file(&path, diagnostics) {
                out.push(entry);
            }
        }
        out
    }

    // Each extra root's own templates, kept as separate groups (rather than one flat Vec) so root
    // order is still visible below for the "later extra wins over an earlier extra" tie-break.
    let mut extra_entries: Vec<Vec<(String, PathBuf, PromptTemplate)>> = Vec::new();
    for extra in extra_roots {
        // pi-parity fix (L8): pi's own `resolveCliPaths`→`resolvePath` expands a leading `~` on a
        // `--prompt-template` path; ours previously took it verbatim — usually masked by the shell
        // expanding it first, but not for a quoted argument or one built programmatically.
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let expanded = crate::tools::expand_tilde(extra, home.as_deref().and_then(|p| p.to_str()));
        let root = PathBuf::from(expanded);
        if root.is_dir() {
            // Already scanned via a standard root, or an earlier `--prompt-template`, under a
            // different (possibly symlinked or relative) path — walking it again would double-count
            // its templates and self-collide every name in it against a phantom duplicate.
            if !seen_roots.insert(crate::path_utils::resolved_path(&root)) {
                continue;
            }
            let mut dir_diagnostics = Vec::new();
            extra_entries.push(load_dir(&root, &mut dir_diagnostics));
            collisions.extend(
                dir_diagnostics
                    .into_iter()
                    .map(|m| Collision::message_only("prompt", m)),
            );
        } else if root.extension().and_then(|e| e.to_str()) == Some("md") {
            // Same reasoning as the directory case above, for a standalone `.md` file.
            if !seen_roots.insert(crate::path_utils::resolved_path(&root)) {
                continue;
            }
            // pi-parity fix (M3): pi's other `--prompt-template` shape — a single standalone `.md`
            // file, one template, no directory of its own resources.
            // A discarded diagnostics sink: this call site already reports a `None` with its own, more
            // specific "--prompt-template file" framing below, so `load_file`'s own diagnostic (which
            // would otherwise duplicate it) is intentionally not folded into `collisions` — matching
            // `skills::discover_with_diagnostics_impl`'s identical `--skill` single-file branch.
            match load_file(&root, &mut Vec::new()) {
                Some(entry) => extra_entries.push(vec![entry]),
                None => {
                    let message = format!("--prompt-template file could not be read: {extra}");
                    tracing::warn!("{message}");
                    collisions.push(Collision::message_only("prompt", message));
                }
            }
        } else {
            let message = format!(
                "--prompt-template path does not exist, or is not a directory or a .md file: \
                 {extra}"
            );
            tracing::warn!("{message}");
            collisions.push(Collision::message_only("prompt", message));
        }
    }

    for (root, scope) in &standard_roots {
        let mut dir_diagnostics = Vec::new();
        for (name, path, mut template) in load_dir(root, &mut dir_diagnostics) {
            template.scope = *scope;
            // Later standard roots (project) win over earlier (user) on name collisions.
            if let Some(existing_path) = origins.get(&name) {
                let message = format!(
                    "prompt template \"{name}\" defined at both {} and {} — the latter wins",
                    existing_path.display(),
                    path.display()
                );
                tracing::warn!("{message}");
                collisions.push(Collision {
                    resource_type: "prompt",
                    name: name.clone(),
                    winner_path: Some(path.clone()),
                    loser_path: Some(existing_path.clone()),
                    winner_source: None,
                    loser_source: None,
                    message,
                });
            }
            origins.insert(name.clone(), path);
            if let Some(existing) = found.iter_mut().find(|t| t.name == name) {
                *existing = template;
            } else {
                found.push(template);
            }
        }
        collisions.extend(
            dir_diagnostics
                .into_iter()
                .map(|m| Collision::message_only("prompt", m)),
        );
    }
    // Names claimed by a standard root — snapshotted *before* extra roots are processed, so a
    // standard-root template always wins over an extra one, but two extra roots can still shadow each
    // other (later wins), matching this function's doc comment.
    let standard_names: std::collections::HashSet<String> = origins.keys().cloned().collect();
    for entries in extra_entries {
        // `template.scope` needs no further tagging here — `load_file`'s own placeholder ("temporary")
        // already is the correct final value for every entry reached through this extra-roots path.
        for (name, path, template) in entries {
            if let Some(existing_path) = standard_names
                .contains(&name)
                .then(|| origins.get(&name))
                .flatten()
            {
                let message = format!(
                    "prompt template \"{name}\" defined at both {} (standard root) and {} \
                     (--prompt-template) — the standard root wins",
                    existing_path.display(),
                    path.display()
                );
                tracing::warn!("{message}");
                collisions.push(Collision {
                    resource_type: "prompt",
                    name: name.clone(),
                    winner_path: Some(existing_path.clone()),
                    loser_path: Some(path.clone()),
                    winner_source: Some("standard root"),
                    loser_source: Some("--prompt-template"),
                    message,
                });
                continue;
            }
            if let Some(existing_path) = origins.get(&name) {
                let message = format!(
                    "prompt template \"{name}\" defined at both {} and {} — the latter wins",
                    existing_path.display(),
                    path.display()
                );
                tracing::warn!("{message}");
                collisions.push(Collision {
                    resource_type: "prompt",
                    name: name.clone(),
                    winner_path: Some(path.clone()),
                    loser_path: Some(existing_path.clone()),
                    winner_source: None,
                    loser_source: None,
                    message,
                });
            }
            origins.insert(name.clone(), path);
            if let Some(existing) = found.iter_mut().find(|t| t.name == name) {
                *existing = template;
            } else {
                found.push(template);
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    (found, collisions)
}

/// Split optional `---` frontmatter (reading `argument-hint:` and `description:`) from the body, and
/// derive a `description` (frontmatter wins; otherwise the first non-empty body line, truncated like
/// pi to 60 chars). Returns `(argument_hint, description, body)`.
///
/// Frontmatter parsing itself is [`crate::skills::parse_frontmatter`] — shared with `skills.rs` rather
/// than reimplemented here, so a long `description:`/`argument-hint:` written as a YAML block scalar
/// (`key: |` / `key: >`, its value spanning the following more-indented lines) is understood the same
/// way a skill's frontmatter already is, instead of silently collapsing to the literal `|`/`>`
/// character with its continuation lines dropped (LOW pi-parity gap, fixed).
fn parse(text: &str) -> (Option<String>, String, String) {
    let (frontmatter, body) = crate::skills::parse_frontmatter(text);
    // pi-parity fix (Task #42): pi's own `extractFrontmatter` does a full `.trim()` on the body
    // (`normalized.slice(endIndex + 4).trim()`) — this used to only `trim_end()`, so a blank line left
    // after the closing `---` fence (a common formatting habit) leaked a leading `"\n"` into the body
    // that pi would strip. `skills::expand_if_skill_invocation`'s own equivalent call site already does
    // the full trim correctly; only prompt templates had this gap.
    let body = body.trim().to_string();
    let hint = frontmatter
        .get("argument-hint")
        .cloned()
        .filter(|h| !h.is_empty());
    let description = frontmatter
        .get("description")
        .cloned()
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| first_line_summary(&body));
    (hint, description, body)
}

/// A short summary from the first non-empty body line — pi truncates to 60 chars and appends `...`.
fn first_line_summary(body: &str) -> String {
    let Some(line) = body.lines().find(|l| !l.trim().is_empty()) else {
        return String::new();
    };
    // Count by chars so a multi-byte boundary never splits mid-codepoint.
    if line.chars().count() > 60 {
        let truncated: String = line.chars().take(60).collect();
        format!("{truncated}...")
    } else {
        line.to_string()
    }
}

/// If `message` is a `/name ...` invocation of a known template, expand and return it; otherwise
/// return the message unchanged.
pub fn expand_if_slash(message: &str, templates: &[PromptTemplate]) -> String {
    let Some(rest) = message.strip_prefix('/') else {
        return message.to_string();
    };
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    match templates.iter().find(|t| t.name == name) {
        Some(t) => substitute(&t.body, &parse_command_args(args)),
        None => message.to_string(),
    }
}

/// Split an argument string into fields, honoring single and double quotes so `"a b" c` is two args
/// (`a b`, `c`) rather than three. Mirrors pi's `parseCommandArgs`: a quote starts a span that runs to
/// the matching quote, and the quote characters themselves are dropped. Any unquoted Unicode whitespace
/// character is a field separator, same as pi's own `/\s/.test(char)` — not just space/tab/`\n`, but
/// also a bare `\r`, vertical tab, form feed, and Unicode space separators; a multi-line pasted argument
/// (e.g. `/name label\n\ndescription text`) splits on every line break, not just the trailing ones. A
/// quoted span still preserves an embedded whitespace character (including a newline) verbatim.
///
/// A field is only ever pushed when non-empty — matching pi's own `if (current) args.push(current)`
/// (JS truthiness: only `""` is falsy for a string, so `" "` still counts and is kept). This means a
/// truly empty quoted argument (`""`, with nothing between the quotes) is silently dropped rather than
/// becoming an empty-string token — pi-parity fix (L4): ours previously tracked field-in-progress via a
/// separate `started` flag set the instant a quote opened, so `""` produced an empty token that shifted
/// every later `$N` index by one, unlike pi.
pub fn parse_command_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in input.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                } else {
                    current.push(ch);
                }
            }
            None => match ch {
                '"' | '\'' => {
                    quote = Some(ch);
                }
                // pi-parity fix (Task #44): pi's own `parseCommandArgs` (prompt-templates.ts:40) tests
                // `/\s/.test(char)`, matching any Unicode whitespace character — a bare `\r` (a
                // stray carriage return not part of a `\r\n` pair), vertical tab, form feed, or a
                // Unicode space separator all split fields there too, not just space/tab/`\n`.
                // `char::is_whitespace()` is Rust's Unicode-whitespace-category equivalent, closely
                // matching JS's `\s` (the one known gap: JS's `\s` also matches U+FEFF BOM, which isn't
                // in Unicode's White_Space property and so isn't matched by `is_whitespace()` — a
                // vanishingly unlikely character to find embedded in a slash-command's arguments).
                c if c.is_whitespace() => {
                    if !current.is_empty() {
                        args.push(std::mem::take(&mut current));
                    }
                }
                _ => {
                    current.push(ch);
                }
            },
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

/// Bash-style argument substitution over a template body. Supports:
/// - `$ARGUMENTS` / `$@` → all args joined by a space.
/// - `$N` (any positive N, not just 1–9) → the Nth positional (1-based); out of range → empty.
/// - `${@:N}` / `${@:N:L}` → bash array slice: args from index N, optionally L of them, space-joined.
/// - `${N:-default}` → the Nth positional if present and non-empty, else `default`.
///
/// Done in a single left-to-right scan so a substituted value is never itself re-scanned (a plain
/// `replace` pass could rewrite a `$1` that came *out* of an argument).
fn substitute(body: &str, args: &[String]) -> String {
    let all = args.join(" ");
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            // Copy the current UTF-8 char verbatim (indices stay on char boundaries because we only
            // ever advance past whole `$...` tokens or single ASCII bytes that are < 0x80).
            let ch_len = utf8_len(bytes[i]);
            out.push_str(&body[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        // We're at `$`. Try each form in turn; fall back to a literal `$` if none match.
        let after = &body[i + 1..];
        if let Some(stripped) = after.strip_prefix("ARGUMENTS") {
            out.push_str(&all);
            i = body.len() - stripped.len();
        } else if after.starts_with('@') {
            out.push_str(&all);
            i += 2; // `$@`
        } else if after.starts_with('{') {
            if let Some(end) = after.find('}') {
                let inner = &after[1..end];
                out.push_str(&expand_brace(inner, args, &all));
                i += 2 + end; // `$` + `{` … `}`
            } else {
                out.push('$');
                i += 1;
            }
        } else if let Some(digits) = leading_digits(after) {
            out.push_str(positional(args, digits));
            i += 1 + digits.len();
        } else {
            out.push('$');
            i += 1;
        }
    }
    out
}

/// Expand the contents of a `${...}` placeholder. `inner` is the text between the braces.
///
/// Mirrors pi's own substitution regex (`prompt-templates.ts:73`,
/// `/\$\{(\d+):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)/g`) exactly: a `${...}` body
/// only ever means something when it's `@:N`/`@:N:L` (a slice, `N`/`L` pure digits), `N:-default` (`N`
/// pure digits), or `@`/`ARGUMENTS` alone. Anything else — a non-digit slice/positional index, or a body
/// pi's regex simply doesn't match at all (e.g. `${HOME}` in a template documenting shell/env-var
/// syntax) — isn't a match for `.replace()` either, so pi leaves it untouched verbatim. pi-parity fix
/// (Task #41): this used to fall through to `String::new()` for any unrecognized body, silently deleting
/// arbitrary template text that merely looked like a placeholder instead of leaving it literal.
fn expand_brace(inner: &str, args: &[String], all: &str) -> String {
    // Reconstructs the original `${inner}` text verbatim — what every "doesn't match pi's regex"
    // fallback below returns instead of deleting the placeholder.
    let literal = || format!("${{{inner}}}");

    if let Some(spec) = inner.strip_prefix("@:") {
        // `@:N` or `@:N:L` — a bash array slice. Both `N` and (if present) `L` must be pure digits to
        // match pi's `\$\{@:(\d+)(?::(\d+))?\}` — e.g. `${@:abc}` or `${@:1:x}` isn't a match at all.
        let (start_str, len_str) = match spec.split_once(':') {
            Some((s, l)) => (s, Some(l)),
            None => (spec, None),
        };
        let Some(start) = parse_slice_start(start_str) else {
            return literal();
        };
        let len = match len_str {
            Some(l) => match parse_usize(l) {
                Some(len) => Some(len),
                None => return literal(),
            },
            None => None,
        };
        let slice: &[String] = match len {
            // `start`/`len` are parsed straight from a `${@:N:L}` placeholder in a project's own
            // `.claude/prompts/*.md` file — untrusted repo content. `saturating_add` so a
            // pathological `N` near `usize::MAX` can't overflow before `.min()` clamps it.
            Some(len) => args
                .get(start..start.saturating_add(len).min(args.len()))
                .unwrap_or(&[]),
            None => args.get(start..).unwrap_or(&[]),
        };
        return slice.join(" ");
    }
    if let Some((num, default)) = inner.split_once(":-") {
        // `${N:-default}` — default when the positional is missing or empty. `num` must be pure,
        // non-empty digits to match pi's `\$\{(\d+):-([^}]*)\}` — e.g. `${1x:-default}` isn't a match at
        // all and must stay literal. A pure-digit `num` that computes to an invalid index (`0`, since
        // this positional scheme is 1-based; or a value so large it overflows `usize`) is still a regex
        // match, though — pi's own `args[index]` on an out-of-range index is just `undefined`, which
        // resolves to the default exactly like a missing positional does, not "no match at all".
        if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
            return literal();
        }
        return match parse_index(num).and_then(|idx| args.get(idx)) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => default.to_string(),
        };
    }
    if inner == "@" || inner == "ARGUMENTS" {
        return all.to_string();
    }
    // pi-parity fix (Task #41, minor superset, aligned): a bare `${N}` (no `:-default`) used to be
    // treated as positional substitution, same as unbraced `$N` — but pi's regex has no alternative for
    // a braced bare digit at all (only the unbraced `\$(\d+)` form matches that), so pi leaves `${1}`
    // literal. Falls through to the `literal()` fallback below instead, matching pi exactly; unbraced
    // `$1` (handled in `substitute`, not here) is unaffected.
    literal()
}

/// The 1-based positional `digits` as a `&str` slice into `args`, or `""` if missing/unparsable.
fn positional<'a>(args: &'a [String], digits: &str) -> &'a str {
    match parse_index(digits).and_then(|i| args.get(i)) {
        Some(v) => v.as_str(),
        None => "",
    }
}

/// Parse a 1-based positional number into a 0-based index (`"1"` → `0`). `"0"`/empty/overflow → None.
fn parse_index(s: &str) -> Option<usize> {
    let n: usize = s.parse().ok()?;
    n.checked_sub(1)
}

/// Parse a `${@:N}` slice-start number into a 0-based index, matching pi's own bash-convention clamp:
/// `N=0` means "from the start" (same as `N=1`), not "one before the first argument" — unlike
/// [`parse_index`] (used for `$N`/`${N:-default}`, where a positional `$0` has no meaning and stays
/// empty), a slice start is always a valid index, so `saturating_sub` clamps instead of failing.
fn parse_slice_start(s: &str) -> Option<usize> {
    let n: usize = s.parse().ok()?;
    Some(n.saturating_sub(1))
}

fn parse_usize(s: &str) -> Option<usize> {
    s.parse().ok()
}

/// The run of leading ASCII digits in `s`, or `None` if it doesn't start with a digit.
fn leading_digits(s: &str) -> Option<&str> {
    let end = s.bytes().take_while(u8::is_ascii_digit).count();
    (end > 0).then(|| &s[..end])
}

/// Length in bytes of the UTF-8 sequence whose lead byte is `b`.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_project_never_sees_project_prompt_templates() {
        // `project_trusted` is a required parameter precisely so a call site can't forget it (see
        // `discover`'s doc comment) — an untrusted project's own `.claude/prompts/*.md` must never be
        // discovered (its body is injected into the model's context the moment `/name` is invoked),
        // mirroring the fix already structural in `skills::discover`.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(
            pdir.join("untrusted-only.md"),
            "This body must never reach an untrusted project's model context.",
        )
        .unwrap();

        let trusted = discover(tmp.path(), true, &[]);
        assert!(
            trusted.iter().any(|t| t.name == "untrusted-only"),
            "sanity check: the template must be discoverable when trusted"
        );

        let untrusted = discover(tmp.path(), false, &[]);
        assert!(
            !untrusted.iter().any(|t| t.name == "untrusted-only"),
            "an untrusted project's own prompt templates must not be discovered: {untrusted:?}"
        );

        // Expansion must fall through unchanged too — nothing to expand if it was never discovered.
        assert_eq!(
            expand_if_slash("/untrusted-only", &untrusted),
            "/untrusted-only"
        );
    }

    #[test]
    fn discovers_and_expands_template() {
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(
            pdir.join("fix.md"),
            "---\nargument-hint: <file>\n---\nFix the bug in $1 and explain: $ARGUMENTS",
        )
        .unwrap();

        // `discover` also scans the developer's real `~/.claude/prompts`; assert on our template by
        // name rather than the total count.
        let templates = discover(tmp.path(), true, &[]);
        let fix = templates
            .iter()
            .find(|t| t.name == "fix")
            .expect("fix template");
        assert_eq!(fix.argument_hint.as_deref(), Some("<file>"));

        let expanded = expand_if_slash("/fix foo.rs urgently", &templates);
        assert_eq!(
            expanded,
            "Fix the bug in foo.rs and explain: foo.rs urgently"
        );
    }

    #[test]
    fn an_oversized_prompt_template_is_skipped_before_being_read() {
        // A pathological template file (or thousands of small ones, in a checked-out repo) must not be
        // fully read into memory before any validation runs — `MAX_PROMPT_TEMPLATE_FILE_LEN` is checked
        // via `fs::metadata` *before* `fs::read_to_string` ever runs, mirroring
        // `skills::MAX_SKILL_FILE_LEN`'s identical guard.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        let oversized = "x".repeat(MAX_PROMPT_TEMPLATE_FILE_LEN as usize + 1);
        fs::write(pdir.join("huge.md"), oversized).unwrap();

        let capture = crate::tracing_test::capture(|| {
            let (found, collisions) = discover_with_diagnostics(tmp.path(), true, &[]);
            assert!(
                !found.iter().any(|t| t.name == "huge"),
                "an oversized template must not be discovered: {found:?}"
            );
            assert!(
                collisions.iter().any(|c| c.to_string().contains("exceeds")),
                "got: {collisions:?}"
            );
        });
        assert!(
            capture.messages().iter().any(|m| m.contains("exceeds")),
            "an oversized prompt template must be logged, not silently skipped with no signal: {:?}",
            capture.messages()
        );
    }

    #[test]
    fn an_unreadable_prompt_template_via_standard_roots_is_warned_not_silently_dropped() {
        // Previously `load_file`'s bare `fs::read_to_string(path).ok()?` dropped a read failure with
        // zero diagnostic when reached through `load_dir`'s standard-roots loop — unlike the
        // extra-single-file-root branch a few lines above, which already `tracing::warn!`s and records
        // a `Collision::message_only` on failure. Non-UTF8 bytes make `fs::read_to_string` fail while
        // the file itself stays a regular file, so it isn't filtered out by `load_dir`'s own
        // `is_file()` check before ever reaching `load_file` (a portable way to hit the `Err` branch
        // without a chmod/permissions dance).
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join("bad.md"), [0xff, 0xfe, 0x00, 0xff]).unwrap();

        let capture = crate::tracing_test::capture(|| {
            let (found, collisions) = discover_with_diagnostics(tmp.path(), true, &[]);
            assert!(
                !found.iter().any(|t| t.name == "bad"),
                "an unreadable template must not be discovered: {found:?}"
            );
            assert!(
                collisions
                    .iter()
                    .any(|c| c.to_string().contains("failed to read")),
                "got: {collisions:?}"
            );
        });
        assert!(
            capture
                .messages()
                .iter()
                .any(|m| m.contains("failed to read")),
            "an unreadable prompt template must be logged, not silently dropped: {:?}",
            capture.messages()
        );
    }

    #[test]
    fn a_dotfile_template_is_not_discovered() {
        // Regression guard: `load_dir`'s `WalkBuilder` never sets `.hidden(...)` at all, relying on
        // the crate's own default (`hidden: true`, which — despite the confusing name — means dotfiles
        // *are* skipped), matching pi's own `collectAutoPromptEntries`'s explicit
        // `if (entry.name.startsWith(".")) continue;`. A deliberately-dotted `.draft.md` (a real
        // "hide this WIP template" convention) must not leak into discovery.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join(".draft.md"), "a work-in-progress template").unwrap();

        let templates = discover(tmp.path(), true, &[]);
        assert!(
            !templates
                .iter()
                .any(|t| t.name == ".draft" || t.name == "draft"),
            "a dotfile template must not be discovered: {templates:?}"
        );
    }

    #[test]
    fn a_gitignore_entry_hides_a_matching_prompt_template() {
        // Pi-parity audit H66: `load_dir` did a plain `fs::read_dir` with no ignore-crate usage at
        // all, unlike `skills.rs`'s sibling `walk` — a `.claude/prompts/.gitignore` entry (e.g.
        // `draft-*.md` hiding a work-in-progress template) that pi would honor was silently ignored
        // here, exposing a `/name` command pi itself would never expose.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join(".gitignore"), "draft-*.md\n").unwrap();
        fs::write(pdir.join("draft-outline.md"), "should not surface").unwrap();
        fs::write(pdir.join("real.md"), "a real template").unwrap();

        let templates = discover(tmp.path(), true, &[]);
        assert!(
            !templates.iter().any(|t| t.name == "draft-outline"),
            "a gitignored template must not be discovered: {templates:?}"
        );
        assert!(
            templates.iter().any(|t| t.name == "real"),
            "a non-ignored template in the same directory must still be discovered: {templates:?}"
        );
    }

    #[test]
    fn a_git_info_exclude_entry_does_not_hide_a_real_prompt_template() {
        // Matches `skills.rs`'s identical H65 test: pi's own `IGNORE_FILE_NAMES` never includes
        // `.git/info/exclude` or a global excludes file, so those must not hide a template either.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::create_dir_all(tmp.path().join(".git/info")).unwrap();
        fs::write(tmp.path().join(".git/info/exclude"), "real.md\n").unwrap();
        fs::write(pdir.join("real.md"), "a real template").unwrap();

        let templates = discover(tmp.path(), true, &[]);
        assert!(
            templates.iter().any(|t| t.name == "real"),
            "a .git/info/exclude entry must not hide a real prompt template: {templates:?}"
        );
    }

    #[test]
    fn a_template_in_a_nested_subdirectory_is_not_discovered() {
        // Non-recursive, matching pi's own `readdirSync` (no subdirectory descent) — prompt templates,
        // unlike skills, are never nested in a subdirectory of their own.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(pdir.join("nested")).unwrap();
        fs::write(pdir.join("nested/deep.md"), "should not surface").unwrap();
        fs::write(pdir.join("top.md"), "a top-level template").unwrap();

        let templates = discover(tmp.path(), true, &[]);
        assert!(
            !templates.iter().any(|t| t.name == "deep"),
            "a template nested in a subdirectory must not be discovered: {templates:?}"
        );
        assert!(
            templates.iter().any(|t| t.name == "top"),
            "a top-level template must still be discovered: {templates:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_is_empty_when_no_names_collide() {
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join("solo.md"), "Body.").unwrap();
        // `discover_with_diagnostics` also scans the developer's real `~/.claude/prompts`, which could
        // in principle carry a same-named file — vanishingly unlikely for this fixture's name, and
        // consistent with how this module's other `discover` (not `discover_in`) tests already accept
        // scanning the real user root.
        let (_, collisions) = discover_with_diagnostics(tmp.path(), true, &[]);
        assert!(
            !collisions.iter().any(|c| c.to_string().contains("solo")),
            "got: {collisions:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_populates_structured_collision_fields() {
        // pi-parity fix: collision diagnostics used to be a flattened `Vec<String>` (pi's own
        // `ResourceDiagnostic`/`ResourceCollision`, `diagnostics.ts:1-16`, is structured) — a client had
        // no way to build tooling (e.g. "which one do you want?") on top of a plain sentence. This mirrors
        // `skills.rs`'s own `discover_with_diagnostics_populates_structured_collision_fields` — same
        // `Collision` type, reused by this module.
        let tmp = tempfile::tempdir().unwrap();
        let user_pdir = tmp.path().join("user/.claude/prompts");
        let project_pdir = tmp.path().join("project/.claude/prompts");
        fs::create_dir_all(&user_pdir).unwrap();
        fs::create_dir_all(&project_pdir).unwrap();
        fs::write(user_pdir.join("dup.md"), "user version").unwrap();
        fs::write(project_pdir.join("dup.md"), "project version").unwrap();

        // Standard roots are `$HOME/.claude/prompts` then `cwd/.claude/prompts` — simulate the "user"
        // root directly via an extra root standing in for it, so this test doesn't need to touch the
        // developer's real `$HOME`.
        let (found, collisions) = discover_with_diagnostics(
            &tmp.path().join("project"),
            true,
            &[user_pdir.to_string_lossy().into_owned()],
        );
        let dup = found.iter().find(|t| t.name == "dup").unwrap();
        // The project's own standard root wins over the `--prompt-template` extra root standing in for
        // "user" here.
        assert_eq!(dup.body, "project version");
        let collision = collisions
            .iter()
            .find(|c| c.name == "dup")
            .expect("must report the collision");
        assert_eq!(collision.resource_type, "prompt");
        assert_eq!(collision.winner_path, Some(project_pdir.join("dup.md")));
        assert_eq!(collision.loser_path, Some(user_pdir.join("dup.md")));
        assert!(!collision.message.is_empty());
        assert_eq!(collision.to_string(), collision.message);
    }

    #[test]
    fn discover_with_diagnostics_alphabetizes_the_final_list_regardless_of_registration_order() {
        // Pi-parity awareness note: pi's own `prompt-templates.ts` never sorts at all (no `.sort(` call
        // anywhere in it) — its listing order is whatever order roots were registered/walked in.
        // Beyond always alphabetizes by name (`found.sort_by`), a deliberate divergence — this pins that
        // down against the one scenario where the two orders would visibly differ: a project-root
        // template ("zebra", registered first) vs. an extra-root template ("apple", registered after)
        // must still come out apple-before-zebra, not in registration order.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join("zebra.md"), "Zebra body.").unwrap();

        let extra_root = tmp.path().join("shared-prompts");
        fs::create_dir_all(&extra_root).unwrap();
        fs::write(extra_root.join("apple.md"), "Apple body.").unwrap();

        let (templates, _) = discover_with_diagnostics(
            tmp.path(),
            true,
            &[extra_root.to_string_lossy().into_owned()],
        );
        let names: Vec<&str> = templates
            .iter()
            .filter(|t| t.name == "apple" || t.name == "zebra")
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["apple", "zebra"],
            "must be alphabetized, not registration order (project root before extra root): {names:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_loads_from_an_explicit_extra_root() {
        let tmp = tempfile::tempdir().unwrap();
        let extra_root = tmp.path().join("shared-prompts");
        fs::create_dir_all(&extra_root).unwrap();
        fs::write(extra_root.join("deploy.md"), "Deploy the thing.").unwrap();
        let cwd = tmp.path().join("project");
        fs::create_dir_all(&cwd).unwrap();
        let (found, _) =
            discover_with_diagnostics(&cwd, true, &[extra_root.to_string_lossy().into_owned()]);
        assert!(found.iter().any(|t| t.name == "deploy"), "got: {found:?}");
    }

    #[test]
    fn discover_with_diagnostics_dedupes_an_extra_root_that_is_actually_a_standard_root() {
        // Pi-parity fix (#45/#73): the same real directory reached via two different paths (here, the
        // project's own `.claude/prompts` standard root and an identically-pathed
        // `--prompt-template` extra root) must be walked only once — not double-counted into a phantom
        // "defined at both X and Y" collision for every template inside it.
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("project");
        let standard_root = cwd.join(".claude/prompts");
        fs::create_dir_all(&standard_root).unwrap();
        fs::write(
            standard_root.join("deploy.md"),
            "Deploy the thing, only actually defined once.",
        )
        .unwrap();
        let (found, collisions) =
            discover_with_diagnostics(&cwd, true, &[standard_root.to_string_lossy().into_owned()]);
        assert_eq!(
            found.iter().filter(|t| t.name == "deploy").count(),
            1,
            "the same directory scanned via two paths must not double-count its templates: {found:?}"
        );
        assert!(
            !collisions.iter().any(|c| c.to_string().contains("deploy")),
            "must not report a phantom self-collision: {collisions:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_warns_when_an_extra_root_does_not_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let (_, collisions) =
            discover_with_diagnostics(tmp.path(), true, &[missing.to_string_lossy().into_owned()]);
        assert!(
            collisions.iter().any(|c| {
                let m = c.to_string();
                m.contains("does not exist") && m.contains("does-not-exist")
            }),
            "got: {collisions:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_loads_a_single_standalone_md_file() {
        // pi-parity fix (M3): pi's `--prompt-template` accepts a standalone `.md` file — one template,
        // no directory of its own — in addition to a directory; ours previously rejected anything that
        // wasn't a directory outright with a false "does not exist or is not a directory" diagnostic.
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("solo.md");
        fs::write(
            &file,
            "---\nargument-hint: <thing>\n---\nDo the solo thing: $1",
        )
        .unwrap();
        let cwd = tmp.path().join("project");
        fs::create_dir_all(&cwd).unwrap();
        let (found, collisions) =
            discover_with_diagnostics(&cwd, true, &[file.to_string_lossy().into_owned()]);
        let solo = found.iter().find(|t| t.name == "solo");
        assert!(solo.is_some(), "got: {found:?}, collisions: {collisions:?}");
        assert_eq!(solo.unwrap().argument_hint.as_deref(), Some("<thing>"));
    }

    #[test]
    fn discover_extra_only_skips_standard_roots_but_keeps_an_explicit_extra_root() {
        // pi-parity fix (M2): pi's `--no-prompt-templates` still honors an explicit `--prompt-template`
        // path passed alongside it (`resource-loader.test.ts`, "should still load additional skill
        // paths when noSkills is true" — the prompt-template equivalent of that same documented,
        // tested combination). Ours used to zero out *both* the standard roots and any
        // `extra_roots`/`--prompt-template` path together, discarding an operator-supplied path they
        // explicitly asked to keep.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join("standard.md"), "Standard template body.").unwrap();
        let extra_root = tmp.path().join("custom-prompts");
        fs::create_dir_all(&extra_root).unwrap();
        fs::write(extra_root.join("custom.md"), "Custom template body.").unwrap();

        // Sanity check / positive control: with normal discovery (not `--no-prompt-templates`), the
        // standard root's own template *is* found — proving the assertion below is actually exercising
        // a skip, not just a location `discover_extra_only` was never going to look at regardless.
        let (normally_found, _) = discover_with_diagnostics(tmp.path(), true, &[]);
        assert!(
            normally_found.iter().any(|t| t.name == "standard"),
            "sanity check: the standard root must be discoverable normally: {normally_found:?}"
        );

        let (found, _) = discover_extra_only(&[extra_root.to_string_lossy().into_owned()]);
        assert!(
            found.iter().any(|t| t.name == "custom"),
            "an explicit extra root must still load: {found:?}"
        );
        assert!(
            !found.iter().any(|t| t.name == "standard"),
            "the standard root must be skipped entirely: {found:?}"
        );
    }

    #[test]
    fn discover_with_diagnostics_a_standard_root_wins_over_an_extra_root_on_collision() {
        // pi-parity fix: pi appends `--prompt-template` paths *after* project/user templates and keeps
        // whichever name it sees *first* — an operator-supplied extra root fills a gap, it doesn't
        // silently override a project's own template of the same name.
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(pdir.join("dup.md"), "standard root version").unwrap();
        let extra_root = tmp.path().join("extra");
        fs::create_dir_all(&extra_root).unwrap();
        fs::write(extra_root.join("dup.md"), "extra root version").unwrap();
        let (found, collisions) = discover_with_diagnostics(
            tmp.path(),
            true,
            &[extra_root.to_string_lossy().into_owned()],
        );
        let dup = found.iter().find(|t| t.name == "dup").unwrap();
        assert_eq!(dup.body, "standard root version");
        let collision = collisions
            .iter()
            .find(|c| c.name == "dup")
            .expect("must report the collision");
        assert!(
            collision.to_string().contains("standard root wins"),
            "got: {collisions:?}"
        );
        assert_eq!(collision.resource_type, "prompt");
        assert_eq!(collision.winner_source, Some("standard root"));
        assert_eq!(collision.loser_source, Some("--prompt-template"));
        assert_eq!(collision.winner_path, Some(pdir.join("dup.md")));
        assert_eq!(collision.loser_path, Some(extra_root.join("dup.md")));
    }

    #[test]
    fn discover_with_diagnostics_two_extra_roots_still_shadow_each_other_later_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let extra1 = tmp.path().join("extra1");
        let extra2 = tmp.path().join("extra2");
        fs::create_dir_all(&extra1).unwrap();
        fs::create_dir_all(&extra2).unwrap();
        fs::write(extra1.join("dup.md"), "first extra").unwrap();
        fs::write(extra2.join("dup.md"), "second extra").unwrap();
        let (found, _) = discover_with_diagnostics(
            tmp.path(),
            true,
            &[
                extra1.to_string_lossy().into_owned(),
                extra2.to_string_lossy().into_owned(),
            ],
        );
        let dup = found.iter().find(|t| t.name == "dup").unwrap();
        assert_eq!(dup.body, "second extra");
    }

    #[test]
    fn unknown_slash_is_passed_through() {
        assert_eq!(expand_if_slash("/nope x", &[]), "/nope x");
        assert_eq!(expand_if_slash("plain message", &[]), "plain message");
    }

    #[test]
    fn missing_positional_expands_empty() {
        let t = template("a $1 b $2 c");
        assert_eq!(expand_if_slash("/x only", &[t]), "a only b  c");
    }

    #[test]
    fn positional_zero_is_empty() {
        // `$0`/`${0:-default}` have no meaning in this 1-based scheme and stay empty — unlike
        // `${@:0}` (see `array_slice_start_zero_clamps_to_all_args`), where 0 is a valid slice start
        // that bash convention clamps to 1, not an invalid positional index.
        let t = template("a=[$0] b=[${0:-fallback}]");
        assert_eq!(expand_if_slash("/x one two", &[t]), "a=[] b=[fallback]");
    }

    #[test]
    fn quoted_arguments_stay_together() {
        let args = parse_command_args(r#"foo "bar baz" 'qux quux' end"#);
        assert_eq!(args, vec!["foo", "bar baz", "qux quux", "end"]);
    }

    #[test]
    fn an_empty_quoted_argument_is_dropped_not_kept_as_an_empty_token() {
        // pi: prompt-templates.test.ts, "should handle quoted empty string" — pi's own truthiness
        // check (`if (current) args.push(current)`) drops a truly empty quoted field (`""`) but keeps
        // one that's merely whitespace (`" "`, a non-empty string) — `parseCommandArgs('"" " "')` →
        // `[" "]`, one token, not two (pi-parity fix, L4). Ours used to track "a field is in progress"
        // via a separate flag set the instant a quote opened, so `""` produced an empty-string token
        // that shifted every later `$N` positional index by one relative to pi.
        assert_eq!(parse_command_args("\"\" \" \""), vec![" "]);

        // End-to-end: with `""` dropped instead of kept, `$1` lands on the *next* real argument, not on
        // an empty positional the way it would if the empty quoted field were preserved.
        let t = template("first=[$1]");
        let expanded = expand_if_slash(r#"/x "" real"#, &[t]);
        assert_eq!(expanded, "first=[real]");
    }

    #[test]
    fn quote_aware_positional_substitution() {
        let t = template("first=[$1] second=[$2]");
        let expanded = expand_if_slash(r#"/x "a b" c"#, &[t]);
        assert_eq!(expanded, "first=[a b] second=[c]");
    }

    #[test]
    fn positional_beyond_nine() {
        let t = template("$10 $11");
        let expanded = expand_if_slash("/x 1 2 3 4 5 6 7 8 9 ten eleven", &[t]);
        assert_eq!(expanded, "ten eleven");
    }

    #[test]
    fn array_slice_substitution() {
        let t = template("rest=[${@:2}] two=[${@:2:2}]");
        let expanded = expand_if_slash("/x a b c d e", &[t]);
        assert_eq!(expanded, "rest=[b c d e] two=[b c]");
    }

    #[test]
    fn array_slice_start_zero_clamps_to_all_args() {
        // pi treats `${@:0}` the same as `${@:1}` — bash convention that positional args start at 1,
        // so a slice start of 0 means "from the start" rather than an invalid/negative index. Unlike
        // a bare `$0` (no meaning here, stays empty — see `positional_zero_is_empty`), `${@:0}` is a
        // slice and every slice start is valid.
        let t = template("all=[${@:0}] two=[${@:0:2}]");
        let expanded = expand_if_slash("/x a b c d e", &[t]);
        assert_eq!(expanded, "all=[a b c d e] two=[a b]");
    }

    #[test]
    fn array_slice_with_huge_length_does_not_panic() {
        // A `.claude/prompts/*.md` file is untrusted repo content — a pathological `L` near
        // `usize::MAX` must not overflow `start + len` before `.min(args.len())` clamps it.
        let t = template("slice=[${@:1:18446744073709551615}]");
        let expanded = expand_if_slash("/x a b c", &[t]);
        assert_eq!(expanded, "slice=[a b c]");
    }

    #[test]
    fn an_unrecognized_brace_placeholder_survives_expansion_literally() {
        // pi-parity fix (Task #41 — real content-corruption bug): a `${...}` body that isn't
        // `@:N(:L)?`, `N:-default`, `@`/`ARGUMENTS`, or a bare digit doesn't match pi's own substitution
        // regex at all (prompt-templates.ts:73), so `.replace()` leaves it untouched. `expand_brace`
        // used to fall through to `String::new()` for any such body, silently deleting template text
        // that merely looked like a placeholder — e.g. a template documenting shell/env-var syntax like
        // `${HOME}` must survive expansion verbatim, not vanish.
        let t = template("Set ${HOME}/bin in your profile, then: export PATH=${HOME}/bin:$PATH");
        let expanded = expand_if_slash("/x", &[t]);
        assert_eq!(
            expanded,
            "Set ${HOME}/bin in your profile, then: export PATH=${HOME}/bin:$PATH"
        );
    }

    #[test]
    fn a_bare_braced_positional_with_no_default_stays_literal() {
        // pi-parity fix (Task #41, minor superset, aligned): pi's regex has no alternative for a braced
        // bare digit at all — only the unbraced `\$(\d+)` form is a positional substitution. A bare
        // `${1}` must stay literal, unlike unbraced `$1` which still substitutes normally.
        let t = template("braced=[${1}] unbraced=[$1]");
        let expanded = expand_if_slash("/x hello", &[t]);
        assert_eq!(expanded, "braced=[${1}] unbraced=[hello]");
    }

    #[test]
    fn a_non_digit_brace_body_shaped_like_a_default_or_slice_stays_literal() {
        // Both the `${N:-default}` and `${@:N(:L)?}` forms require `N` (and `L`) to be pure digits to
        // match pi's regex — a non-digit index isn't a match at all and must stay literal, not silently
        // resolve to the default text or an empty/partial slice.
        let t = template("a=[${1x:-default}] b=[${@:abc}] c=[${@:1:x}]");
        let expanded = expand_if_slash("/x hello", &[t]);
        assert_eq!(expanded, "a=[${1x:-default}] b=[${@:abc}] c=[${@:1:x}]");
    }

    #[test]
    fn default_value_substitution() {
        let t = template("name=${2:-anon}");
        assert_eq!(
            expand_if_slash("/x given", std::slice::from_ref(&t)),
            "name=anon"
        );
        assert_eq!(expand_if_slash("/x given second", &[t]), "name=second");
    }

    #[test]
    fn an_arg_value_containing_dollar_arguments_is_not_recursively_resubstituted() {
        // pi: args.test.ts, CRITICAL — `substituteArgs("$ARGUMENTS", ["$1", "$ARGUMENTS"])` must yield
        // the literal joined args, not re-scan the result for further `$`-tokens. `substitute` only
        // ever iterates over `body`'s bytes, never over what it already wrote to `out`, so this should
        // already hold by construction — pinning it so a future rewrite can't regress it silently.
        let t = template("$ARGUMENTS");
        let expanded = expand_if_slash(r#"/x "$1" "$ARGUMENTS""#, &[t]);
        assert_eq!(expanded, "$1 $ARGUMENTS");
    }

    #[test]
    fn a_present_arg_plugged_into_a_default_placeholder_is_not_resubstituted() {
        // pi: args.test.ts — `${1:-7}` with args `["$ARGUMENTS"]` must yield the literal arg value.
        let t = template("${1:-7}");
        let expanded = expand_if_slash(r#"/x "$ARGUMENTS""#, &[t]);
        assert_eq!(expanded, "$ARGUMENTS");
    }

    #[test]
    fn the_default_text_itself_is_not_resubstituted_even_when_it_fires() {
        // pi: args.test.ts — `${3:-$ARGUMENTS}` with only 2 args (index 3 missing, default fires) must
        // yield the literal default text, not expand `$ARGUMENTS` inside it.
        let t = template("${3:-$ARGUMENTS}");
        let expanded = expand_if_slash("/x a b", &[t]);
        assert_eq!(expanded, "$ARGUMENTS");
    }

    #[test]
    fn an_empty_string_positional_still_triggers_the_default() {
        // pi: args.test.ts — `${1:-brief}` with args `[""]` (an explicit empty positional, not a
        // missing one) must still fall back to the default.
        let t = template("${1:-brief}");
        let expanded = expand_if_slash(r#"/x """#, &[t]);
        assert_eq!(expanded, "brief");
    }

    #[test]
    fn unquoted_newlines_split_arguments_like_spaces() {
        // pi: args.test.ts, "should treat unquoted newlines as separators" —
        // `parseCommandArgs("label-2\n\nHere is some description #2.")` → 6 tokens.
        let args = parse_command_args("label-2\n\nHere is some description #2.");
        assert_eq!(
            args,
            vec!["label-2", "Here", "is", "some", "description", "#2."]
        );
    }

    #[test]
    fn parse_command_args_edge_cases() {
        // pi: args.test.ts — empty input, whitespace collapsing (spaces and tabs), and passthrough of
        // special characters/unicode.
        assert_eq!(parse_command_args(""), Vec::<String>::new());
        assert_eq!(parse_command_args("a  b   c"), vec!["a", "b", "c"]);
        assert_eq!(parse_command_args("a\tb\tc"), vec!["a", "b", "c"]);
        assert_eq!(
            parse_command_args("$100 @user #tag"),
            vec!["$100", "@user", "#tag"]
        );
        assert_eq!(parse_command_args("héllo wörld"), vec!["héllo", "wörld"]);
        assert_eq!(
            parse_command_args("  leading and trailing  "),
            vec!["leading", "and", "trailing"]
        );
    }

    #[test]
    fn parse_command_args_splits_on_any_unicode_whitespace_not_just_space_tab_newline() {
        // pi-parity fix (Task #44): pi's own `parseCommandArgs` tests `/\s/.test(char)`, which matches
        // any Unicode whitespace character — a bare `\r`, vertical tab (`\u{0B}`), form feed (`\u{0C}`),
        // and a Unicode space separator (e.g. NBSP `\u{A0}`) must all split fields too, not just
        // space/tab/`\n`. `char::is_whitespace()` is the natural Rust equivalent.
        assert_eq!(parse_command_args("a\rb"), vec!["a", "b"]);
        assert_eq!(parse_command_args("a\u{0B}b"), vec!["a", "b"]);
        assert_eq!(parse_command_args("a\u{0C}b"), vec!["a", "b"]);
        assert_eq!(parse_command_args("a\u{A0}b"), vec!["a", "b"]);
        // A quoted span still preserves an embedded whitespace character verbatim.
        assert_eq!(parse_command_args("\"a\rb\""), vec!["a\rb"]);
    }

    #[test]
    fn empty_argument_hint_is_treated_as_absent() {
        // pi: prompt-templates.test.ts, "should ignore empty argument-hint".
        let (hint, _, _) = parse("---\nargument-hint: \"\"\n---\nBody");
        assert_eq!(hint, None);
    }

    #[test]
    fn a_blank_line_after_the_closing_fence_does_not_leak_into_the_body() {
        // pi-parity fix (Task #42): pi's own `extractFrontmatter` does a full `.trim()` on the body
        // (`normalized.slice(endIndex + 4).trim()`), not just a trailing trim — a blank line left after
        // the closing `---` fence (a common formatting habit) used to leak a leading `"\n"` into the
        // body that pi would strip.
        let (_, _, body) = parse("---\nargument-hint: <x>\n---\n\nFix the bug in $1");
        assert_eq!(body, "Fix the bug in $1");
    }

    #[test]
    fn description_from_frontmatter_then_first_line() {
        assert_eq!(
            parse("---\ndescription: Do the thing\n---\nBody line").1,
            "Do the thing"
        );
        // No `description:` → first non-empty body line.
        assert_eq!(
            parse("---\n---\n\nFirst real line\nmore").1,
            "First real line"
        );
        // No frontmatter at all → first non-empty line of the whole text.
        assert_eq!(parse("Just a body").1, "Just a body");
    }

    #[test]
    fn description_and_argument_hint_support_yaml_block_scalars() {
        // LOW pi-parity gap (fixed): this used to hand-scan frontmatter lines directly, understanding
        // quoted values but not a YAML block scalar (`key: |` / `key: >`) — a `description: |` would
        // parse as the literal one-character string `"|"`, silently dropping every continuation line.
        // Delegating to `skills::parse_frontmatter` (already block-scalar-aware) fixes both fields.
        let (hint, description, body) = parse(
            "---\ndescription: >\n  A folded description\n  that wraps across lines.\n\
             argument-hint: |\n  <first arg>\n  <second arg>\n---\nBody line",
        );
        assert_eq!(
            description, "A folded description that wraps across lines.\n",
            "a folded (>) block scalar joins its lines with spaces, keeping one trailing newline \
             (real YAML's default \"clip\" chomping — pi-parity fix, L3)"
        );
        assert_eq!(
            hint.as_deref(),
            Some("<first arg>\n<second arg>\n"),
            "a literal (|) block scalar joins its lines with newlines, keeping one trailing newline"
        );
        assert_eq!(body, "Body line");
    }

    /// A template with an empty hint/description and the given body, for substitution tests.
    fn template(body: &str) -> PromptTemplate {
        PromptTemplate {
            name: "x".into(),
            argument_hint: None,
            description: String::new(),
            body: body.into(),
            path: PathBuf::from("/x/.claude/prompts/x.md"),
            scope: "user",
        }
    }
}
