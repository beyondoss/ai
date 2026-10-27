//! The stat line, `w|-` `\t` `f|d|<other>` `\t` `<size>` `\t` `<mtime, maybe fractional>`.
//!
//! Round trip: a printed line parses to the values it was printed from. Raw: any input returns or
//! returns `None`, never panics (including for a modification time no `SystemTime` can hold), and
//! what is returned is what the first line says.
#![no_main]

use std::time::{Duration, UNIX_EPOCH};

use beyond_agent::tools::fs::FileKind;
use beyond_agent::tools::fs::shell::__fuzz::stat;
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fn structured(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<()> {
    let writable = u.arbitrary::<bool>()?;
    let kind = *u.choose(&["f", "d", "l", "p", "s", "c", "b"])?;
    let len = u.arbitrary::<u64>()?;
    // Seconds `stat -c %Y`/`find -printf %T@` can really report: a signed 64-bit count, and a real
    // clock never reads past a few billion.
    let secs = u.int_in_range(0..=4_102_444_800u64)?;
    let frac = if u.ratio(1, 2)? {
        format!(".{:010}", u.int_in_range(0..=9_999_999_999u64)?)
    } else {
        String::new()
    };
    let line = format!(
        "{}\t{kind}\t{len}\t{secs}{frac}\n",
        if writable { "w" } else { "-" }
    );
    let m = stat(&line).expect("a well-formed line parses");
    assert_eq!(m.writable, writable, "{line:?}");
    assert_eq!(m.len, len, "{line:?}");
    assert_eq!(
        m.mtime,
        Some(UNIX_EPOCH + Duration::from_secs(secs)),
        "{line:?}"
    );
    let want = match kind {
        "f" => FileKind::File,
        "d" => FileKind::Dir,
        _ => FileKind::Other,
    };
    assert_eq!(m.kind, want, "{line:?}");
    Ok(())
}

fn raw(data: &[u8]) {
    let input = String::from_utf8_lossy(data);
    let got = stat(&input);
    let first = input.lines().next().unwrap_or("");
    if let Some(m) = got {
        let f: Vec<&str> = first.split('\t').collect();
        assert!(
            f.len() >= 3,
            "accepted a line with {} fields: {first:?}",
            f.len()
        );
        assert_eq!(m.writable, f[0] == "w");
        assert_eq!(Some(m.len), f[2].trim().parse().ok());
    }
}

fuzz_target!(|data: &[u8]| {
    raw(data);
    let _ = structured(&mut Unstructured::new(data));
});
