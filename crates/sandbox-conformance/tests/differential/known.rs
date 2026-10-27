//! Divergences the differential run has found and that have their own scenario in
//! `sdk/conformance/scenarios/differential-found.json` (marked `xfail` for the drivers that fail
//! it). They are tolerated here by id, with the drivers that must be the odd ones out, so that one
//! known defect does not hide every other: the same arrangement as the `N1..N4` open bugs of the
//! `ShellFs` property test (`oracle::after`). A fix deletes its entry here and its `xfail` mark there.
use serde_json::Value;

fn strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
        Value::Object(o) => o.values().for_each(|x| strings(x, out)),
        _ => {}
    }
}

/// `Some(id)` when `step` is one an open defect explains and exactly the drivers it is known to
/// afflict (`differing`) are the ones that answered unlike the rest or against the filesystem.
pub fn divergence(step: &Value, differing: &[&str]) -> Option<&'static str> {
    // `DIFFERENTIAL_NO_KNOWN=1` reports these too: to see a defect shrunk, or to check one is fixed.
    if std::env::var_os("DIFFERENTIAL_NO_KNOWN").is_some() {
        return None;
    }
    let mut text = vec![];
    strings(step, &mut text);
    let only = |names: &[&str]| differing.iter().all(|d| names.contains(d));
    // D1: Python reads the first line of a failed command's stderr with `str.splitlines()`, which
    // also splits at CR, VT, FF, FS, GS, RS, NEL, LS and PS; the spec splits at LF only.
    if text.iter().any(|s| {
        s.contains([
            '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
        ])
    }) && only(&["python", "python-async"])
    {
        return Some("D1");
    }
    // D3: the TypeScript client compiles the pattern of a non-literal query with a JavaScript
    // RegExp to find the match it clips a long line around; the spec says Rust syntax, so
    // `(?i)`, `(?P<name>..)` and the like do not compile and the line is clipped as a context line.
    if step["op"] == "search"
        && step["literal"] == false
        && step["pattern"].as_str().is_some_and(|p| p.contains("(?"))
        && only(&["ts"])
    {
        return Some("D3");
    }
    None
}
