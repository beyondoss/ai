//! Generators: names that break naive shell code, and content that breaks naive byte handling.
use hegel::TestCase;
use hegel::generators as gs;

/// Name fragments. Spaces, newlines, tabs, a leading `-`, `--`, shell metacharacters, non-ASCII, and
/// `.git`. (Names that are not valid UTF-8 cannot be written in a JSON request, so they exist only
/// through the environment steps and show up in listings.) None contains `.tmp.`: a name of that shape
/// is what the temp-file check looks for.
pub const FRAGS: &[&str] = &[
    "a",
    "b",
    "note.txt",
    "log.log",
    "a b",
    " lead",
    "trail ",
    "x\ny",
    "tab\there",
    "cr\rname",
    "-",
    "--",
    "-rf",
    "-n",
    "*",
    "?",
    "[ab]",
    "$HOME",
    "$(true)",
    "`true`",
    "a;b",
    "a|b",
    "a&b",
    "'q'",
    "\"dq\"",
    "\\",
    "~",
    "#",
    "%s",
    "ünï",
    "漢字.txt",
    ".hidden",
    "..x",
    "...",
    ".git",
];

/// The text a file is made of; the same vocabulary the search patterns are drawn from.
pub const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma delta",
    "needle",
    "Needle in a haystack",
    "NEEDLE",
    "foo bar",
    "foo$",
    "^foo",
    "a.b",
    "a*b",
    "(x)",
    "[y]",
    "back\\slash",
    "émoji ✓",
    "漢字 needle",
    "tab\there",
    "x=1; y=2",
    "--",
    "-n",
    "",
    " ",
    "dollar $HOME",
    "'quote'",
    "\"dq\"",
    "e1",
    "x1",
];

pub fn name(tc: &TestCase) -> String {
    match tc.draw(gs::integers::<u8>().min_value(0).max_value(11)) {
        0..=7 => tc.draw(gs::sampled_from(FRAGS)).to_string(),
        8 | 9 => format!(
            "{}{}",
            tc.draw(gs::sampled_from(FRAGS)),
            tc.draw(gs::sampled_from(FRAGS))
        ),
        10 => "n".repeat(tc.draw(gs::sampled_from(vec![100usize, 200, 220, 233, 234, 240, 255]))),
        _ => format!("{}{}", "y".repeat(5), tc.draw(gs::integers::<u8>().max_value(3))),
    }
}

fn lines(tc: &TestCase) -> Vec<u8> {
    let n = tc.draw(gs::integers::<usize>().max_value(8));
    let sep = tc.draw(gs::sampled_from(vec!["\n", "\r\n", "\n"]));
    let mut out = String::new();
    for i in 0..n {
        out.push_str(tc.draw(gs::sampled_from(VOCAB)));
        if i + 1 < n || tc.draw(gs::booleans()) {
            out.push_str(sep);
        }
    }
    out.into_bytes()
}

/// Sizes at and around every boundary the write and read paths have: the 48 KiB write chunk, the 96 KiB
/// read window, and a response cap.
pub fn boundary_sizes(cap: Option<usize>, fill: Option<usize>) -> Vec<usize> {
    let mut v = vec![
        49151, 49152, 49153, 98303, 98304, 98305, 196608, 150_000,
    ];
    if let Some(c) = cap {
        v.extend([c / 3, c / 2, c * 3 / 4 - 400, c * 3 / 4, c * 3 / 4 + 64, c]);
    }
    if let Some(f) = fill {
        // Around the size of the tiny filesystem: half, and the last few blocks.
        v.extend([f / 2, f / 2 + 4096, f - 16384, f - 8192, f - 4096, f]);
    }
    v
}

pub fn content(tc: &TestCase, cap: Option<usize>, fill: Option<usize>) -> Vec<u8> {
    match tc.draw(gs::integers::<u8>().min_value(0).max_value(13)) {
        0 => vec![],
        1 => tc.draw(gs::binary().max_size(40)),
        2..=6 => lines(tc),
        7 => {
            // A NUL somewhere in otherwise ordinary text.
            let mut b = lines(tc);
            let at = tc.draw(gs::integers::<usize>().max_value(b.len()));
            b.insert(at, 0);
            b
        }
        8 => {
            // Invalid UTF-8: a lone continuation byte and a truncated sequence.
            let mut b = lines(tc);
            b.extend_from_slice(b"\xff\xfe\x80 needle \xc3\n");
            b
        }
        9 | 10 => {
            let n = tc.draw(gs::sampled_from(boundary_sizes(cap, fill)));
            let unit = {
                let mut u = tc.draw(gs::binary().min_size(1).max_size(24));
                if tc.draw(gs::booleans()) {
                    u.push(b'\n');
                }
                u
            };
            unit.iter().cycle().take(n).copied().collect()
        }
        11 => {
            // One long line with a needle deep inside, for the clip rules.
            let pad = tc.draw(gs::sampled_from(vec![10usize, 480, 497, 600, 1200]));
            let tail = tc.draw(gs::sampled_from(vec![0usize, 10, 700]));
            format!("{}needle{}\n", "z".repeat(pad), "w".repeat(tail)).into_bytes()
        }
        _ => b"line one\r\nline two\r\nneedle\r\n".to_vec(),
    }
}

/// A pattern both the oracle and the engines agree on (see `search_truth`).
pub fn query(tc: &TestCase) -> (String, bool, bool) {
    let literal = tc.draw(gs::booleans());
    let ignore_case = tc.draw(gs::booleans());
    if literal {
        let w = tc.draw(gs::sampled_from(VOCAB.iter().copied().filter(|w| !w.is_empty()).collect::<Vec<_>>()));
        let chars: Vec<char> = w.chars().collect();
        let (a, b) = if tc.draw(gs::booleans()) && chars.len() > 1 {
            (0, chars.len() - 1)
        } else {
            (0, chars.len())
        };
        (chars[a..b].iter().collect(), true, ignore_case)
    } else {
        let words = ["alpha", "needle", "Needle", "foo", "bar", "delta", "e1", "x1", "a", "z"];
        let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
        let alts: Vec<String> = (0..n)
            .map(|_| {
                let w = tc.draw(gs::sampled_from(words.to_vec()));
                let s = if tc.draw(gs::booleans()) { "^" } else { "" };
                let e = if tc.draw(gs::booleans()) { "$" } else { "" };
                format!("{s}{w}{e}")
            })
            .collect();
        (alts.join("|"), false, ignore_case)
    }
}
