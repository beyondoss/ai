//! `parse_records`: `rg --null` / `grep -Z` search output.
//!
//! Structured input is printed exactly as the tools print it (`<path>\0<line><:|-><text><eol>`,
//! no group separator, as the engines are run without one; `eol` is `\n`, or `\r\n` as `rg --crlf` echoes a CRLF file) and must
//! parse back to the records it was made from. Raw input must not panic, must not return more bytes
//! than it was given, must return only paths that occur in the input, and must be a fixed point of
//! print-then-parse.
#![no_main]

use beyond_sandbox::fs::Hit;
use beyond_sandbox::fs::shell::__fuzz::records;
use beyond_sandbox_fuzz::{name, text};
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

struct Rec {
    path: String,
    line: usize,
    text: String,
    is_match: bool,
}

fn print(recs: &[(bool, Rec)], crlf: bool) -> String {
    let mut out = String::new();
    for (_, r) in recs {
        out.push_str(&r.path);
        out.push('\0');
        out.push_str(&r.line.to_string());
        out.push(if r.is_match { ':' } else { '-' });
        out.push_str(&r.text);
        out.push_str(if crlf { "\r\n" } else { "\n" });
    }
    out
}

fn structured(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<()> {
    let crlf = u.arbitrary::<bool>()?;
    let mut recs = Vec::new();
    for i in 0..u.int_in_range(0..=6usize)? {
        recs.push((
            i > 0 && u.ratio(1, 3)?,
            Rec {
                path: name(u, true)?,
                line: if u.ratio(1, 8)? {
                    u.int_in_range(0..=usize::MAX)?
                } else {
                    u.int_in_range(1..=100_000usize)?
                },
                text: text(u, 40, false)?,
                is_match: u.arbitrary()?,
            },
        ));
    }
    let printed = print(&recs, crlf);
    let got = records(&printed);
    let want: Vec<(&str, usize, String, bool)> = recs
        .iter()
        .map(|(_, r)| {
            (
                r.path.as_str(),
                r.line,
                r.text.replace('\r', ""),
                r.is_match,
            )
        })
        .collect();
    let got: Vec<(&str, usize, String, bool)> = got
        .iter()
        .map(|h| {
            (
                h.path.to_str().unwrap_or("<non-utf8>"),
                h.line,
                h.text.clone(),
                h.is_match,
            )
        })
        .collect();
    assert_eq!(got, want, "printed: {printed:?}");
    Ok(())
}

fn raw(data: &[u8]) {
    let input = String::from_utf8_lossy(data);
    let hits = records(&input);
    // Records do not overlap, so what comes back cannot outweigh what went in.
    let total: usize = hits
        .iter()
        .map(|h| h.path.as_os_str().len() + h.text.len())
        .sum();
    assert!(
        total <= input.len(),
        "returned {total} bytes from {} bytes",
        input.len()
    );
    for h in &hits {
        let p = h.path.to_str().expect("path from a str is a str");
        assert!(input.contains(p), "path {p:?} is not in the input");
    }
    // Printing what was parsed and parsing it again changes nothing.
    let recs: Vec<(bool, Rec)> = hits.iter().map(to_rec).map(|r| (false, r)).collect();
    let again = records(&print(&recs, false));
    assert_eq!(again, hits, "not a fixed point; input: {input:?}");
}

fn to_rec(h: &Hit) -> Rec {
    Rec {
        path: h.path.to_str().unwrap_or_default().to_string(),
        line: h.line,
        text: h.text.clone(),
        is_match: h.is_match,
    }
}

fuzz_target!(|data: &[u8]| {
    raw(data);
    let _ = structured(&mut Unstructured::new(data));
});
