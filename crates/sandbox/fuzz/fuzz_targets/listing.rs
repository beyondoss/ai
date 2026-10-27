//! The two NUL-terminated listings: `find -printf '%y\t%p\0'` (`parse_typed_paths`) and the directory
//! listing `<type>\t<size>\t<name>\0` (`parse_dir_entries`, with a cap and a hidden-entry filter).
//!
//! Round trip: names with tabs, newlines, colons and dots come back exactly, in order, honouring the
//! cap and the hidden filter. Raw: no panic, never more entries than the cap, never more bytes than
//! the input, every name a substring of it.
#![no_main]

use beyond_sandbox::fs::FileKind;
use beyond_sandbox::fs::shell::__fuzz::{dir_entries, typed_paths};
use beyond_sandbox_fuzz::name;
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fn structured(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<()> {
    let mut find = String::new();
    let mut find_want = Vec::new();
    let mut ls = String::new();
    let mut ls_all = Vec::new();
    let hidden = u.arbitrary::<bool>()?;
    let cap = u.int_in_range(0..=8usize)?;
    for _ in 0..u.int_in_range(0..=8usize)? {
        let kind = *u.choose(&["f", "d", "l", "p", "N", "?"])?;
        let n = name(u, true)?;
        find.push_str(&format!("{kind}\t{n}\0"));
        find_want.push((n.clone(), kind == "d"));
        let size = u.arbitrary::<u64>()?;
        ls.push_str(&format!("{kind}\t{size}\t{n}\0"));
        // Not an entry at all: broken link / unstattable. Hidden: dropped *before* the cap.
        if kind != "N" && kind != "?" && (hidden || !n.starts_with('.')) {
            let k = match kind {
                "f" => FileKind::File,
                "d" => FileKind::Dir,
                _ => FileKind::Other,
            };
            ls_all.push((n, k, size));
        }
    }
    let got: Vec<_> = typed_paths(&find)
        .into_iter()
        .map(|(p, d)| (p.to_string_lossy().into_owned(), d))
        .collect();
    assert_eq!(got, find_want, "find: {find:?}");
    let got: Vec<_> = dir_entries(&ls, cap, hidden)
        .into_iter()
        .map(|e| (e.name, e.kind, e.len))
        .collect();
    ls_all.truncate(cap);
    assert_eq!(got, ls_all, "ls (cap {cap}, hidden {hidden}): {ls:?}");
    Ok(())
}

fn raw(data: &[u8]) {
    let input = String::from_utf8_lossy(data);
    let found = typed_paths(&input);
    let total: usize = found.iter().map(|(p, _)| p.as_os_str().len()).sum();
    assert!(total <= input.len());
    for (p, _) in &found {
        assert!(input.contains(p.to_str().expect("a str")));
    }
    let cap = data.first().map_or(0, |b| *b as usize % 16);
    let entries = dir_entries(&input, cap, data.len().is_multiple_of(2));
    assert!(entries.len() <= cap);
    let total: usize = entries.iter().map(|e| e.name.len()).sum();
    assert!(total <= input.len());
    for e in &entries {
        assert!(input.contains(&e.name));
    }
}

fuzz_target!(|data: &[u8]| {
    raw(data);
    let _ = structured(&mut Unstructured::new(data));
});
