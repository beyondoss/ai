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
pub fn divergence(step: &Value, _differing: &[&str], answers: &[Value]) -> Option<&'static str> {
    // `DIFFERENTIAL_NO_KNOWN=1` reports these too: to see a defect shrunk, or to check one is fixed.
    if std::env::var_os("DIFFERENTIAL_NO_KNOWN").is_some() {
        return None;
    }
    let mut text = vec![];
    strings(step, &mut text);
    // D5: ripgrep prints `PATH: WARNING: stopped searching binary file after match ...` on stdout
    // when a NUL follows a match. The line has no NUL, so the parser (which scans for the NUL after
    // the path) takes it as the start of the next record's path; which record that is depends on
    // the order rg's threads finish in, so one client's run shows it and another's does not.
    if step["op"] == "search"
        && answers.iter().any(|a| {
            a.to_string()
                .contains("WARNING: stopped searching binary file")
        })
    {
        return Some("D5");
    }
    None
}
