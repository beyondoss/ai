//! `parse_colon_records`: busybox `grep -rn` output, `<path><sep><line><sep><text>`.
//!
//! The parser documents its cost: with no end-of-path marker, a path holding `<sep><digits><sep>` is
//! ambiguous and a path holding a newline cannot be represented. The round trip therefore only
//! generates unambiguous paths; the raw properties hold for everything.
#![no_main]

use beyond_sandbox::fs::shell::__fuzz::colon_records;
use beyond_sandbox_fuzz::{name, text};
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

/// Byte index of the first `<sep><digits><sep>` in `s`: where the parser will cut the path.
fn first_boundary(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    (0..b.len()).find(|&i| {
        (b[i] == b':' || b[i] == b'-') && {
            let d = b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            d > 0 && b.get(i + 1 + d) == Some(&b[i])
        }
    })
}

fn structured(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<()> {
    let mut printed = String::new();
    let mut want = Vec::new();
    for _ in 0..u.int_in_range(0..=6usize)? {
        let path = name(u, false)?;
        let line = u.int_in_range(0..=100_000usize)?;
        let body = text(u, 40, false)?;
        let is_match = u.arbitrary::<bool>()?;
        let sep = if is_match { ':' } else { '-' };
        // Only paths the parser can represent: the first boundary must be the real one.
        if first_boundary(&format!("{path}{sep}{line}{sep}")) != Some(path.len()) {
            continue;
        }
        if u.ratio(1, 4)? {
            printed.push_str("--\n");
        }
        printed.push_str(&format!("{path}{sep}{line}{sep}{body}\n"));
        want.push((path, line, body.replace('\r', ""), is_match));
    }
    let got: Vec<_> = colon_records(&printed)
        .into_iter()
        .map(|h| {
            (
                h.path.to_string_lossy().into_owned(),
                h.line,
                h.text,
                h.is_match,
            )
        })
        .collect();
    assert_eq!(got, want, "printed: {printed:?}");
    Ok(())
}

fn raw(data: &[u8]) {
    let input = String::from_utf8_lossy(data);
    let hits = colon_records(&input);
    let total: usize = hits
        .iter()
        .map(|h| h.path.as_os_str().len() + h.text.len())
        .sum();
    assert!(
        total <= input.len(),
        "returned {total} bytes from {} bytes",
        input.len()
    );
    assert!(hits.len() <= input.lines().count());
    for h in &hits {
        let p = h.path.to_str().expect("path from a str is a str");
        assert!(input.contains(p), "path {p:?} is not in the input");
    }
}

fuzz_target!(|data: &[u8]| {
    raw(data);
    let _ = structured(&mut Unstructured::new(data));
});
