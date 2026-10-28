//! The local half of a download (`bulk::extract`, the policy layer over the `tar` crate) on archives
//! from a hostile sandbox.
//!
//! * Raw: arbitrary bytes as an archive never panic.
//! * Structured: members named from an alphabet of `..`, `.`, `/` and short names, as files,
//!   directories, symlinks and hard links, in any order. Whatever the verdict, nothing appears in the
//!   directory that holds the target besides the target, and a link planted in the target never gets
//!   a file written through it into `outside`.
#![no_main]

use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fn name(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<String> {
    const PARTS: [&str; 7] = ["..", ".", "a", "b", "c", "", "outside"];
    let n = u.int_in_range(1..=4usize)?;
    let mut parts = Vec::new();
    for _ in 0..n {
        parts.push(*u.choose(&PARTS)?);
    }
    let mut s = parts.join("/");
    if u.ratio(1, 8)? {
        s.insert(0, '/');
    }
    Ok(s)
}

fn member(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<Vec<u8>> {
    let mut h = tar::Header::new_gnu();
    let n = name(u)?;
    let g = h.as_gnu_mut().expect("gnu");
    let nb = n.as_bytes();
    g.name[..nb.len().min(100)].copy_from_slice(&nb[..nb.len().min(100)]);
    let kind = *u.choose(&[b'0', b'5', b'2', b'1', b'3'])?;
    h.set_entry_type(tar::EntryType::new(kind));
    h.set_mode(u.int_in_range(0..=0o7777u32)?);
    h.set_mtime(u.arbitrary::<u32>()?.into());
    let body = if kind == b'0' {
        let n = u.int_in_range(0..=40usize)?.min(u.len());
        u.bytes(n)?.to_vec()
    } else {
        Vec::new()
    };
    h.set_size(body.len() as u64);
    if kind == b'2' || kind == b'1' {
        let l = name(u)?;
        let lb = l.as_bytes();
        let g = h.as_gnu_mut().expect("gnu");
        g.linkname[..lb.len().min(100)].copy_from_slice(&lb[..lb.len().min(100)]);
    }
    h.set_cksum();
    let mut out = h.as_bytes().to_vec();
    out.extend_from_slice(&body);
    out.resize(out.len().div_ceil(512) * 512, 0);
    Ok(out)
}

fn check(archive: &[u8]) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("root");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&root).expect("root");
    std::fs::create_dir_all(&outside).expect("outside");
    // A link in the target that points out: nothing may be written through it.
    let _ = std::os::unix::fs::symlink(&outside, root.join("planted"));
    let _ = beyond_sandbox::bulk::extract(archive, &root);
    for e in std::fs::read_dir(tmp.path()).expect("read") {
        let n = e.expect("entry").file_name();
        assert!(
            n == "root" || n == "outside",
            "wrote beside the target: {n:?}"
        );
    }
    assert!(
        std::fs::read_dir(&outside)
            .expect("outside")
            .next()
            .is_none(),
        "wrote through a link"
    );
}

fuzz_target!(|data: &[u8]| {
    let tmp = tempfile::tempdir().expect("tempdir");
    let _ = beyond_sandbox::bulk::extract(data, &tmp.path().join("raw"));
    let mut u = Unstructured::new(data);
    let mut a = Vec::new();
    while !u.is_empty() && a.len() < 64 * 1024 {
        match member(&mut u) {
            Ok(m) => a.extend(m),
            Err(_) => break,
        }
    }
    a.extend_from_slice(&[0u8; 1024]);
    check(&a);
});
