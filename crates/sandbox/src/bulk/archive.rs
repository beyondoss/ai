//! The tar subset of FILES-OVER-EXEC.md section 5.8: header writing for upload, and a push parser
//! for download.
//!
//! Written by hand rather than with a `tar` crate: the subset is small, the same rules are
//! implemented in three other languages, and a parser that is *pushed* chunks as they arrive (a
//! stream frame at a time) needs no thread between the network and the extractor.

use super::BulkError;

pub(super) const BLOCK: usize = 512;

/// A member of an archive, as the reader reports it.
#[derive(Debug, Clone)]
pub(super) struct Entry {
    /// The name exactly as the archive gives it (prefix, `L` and pax already applied).
    pub name: Vec<u8>,
    pub kind: u8,
    pub mode: u32,
    pub size: u64,
    pub mtime: u64,
    pub link: Vec<u8>,
}

fn put_octal(field: &mut [u8], value: u64) {
    let w = field.len() - 1;
    let s = format!("{value:0w$o}");
    // Callers pass values that fit; a value that does not would be a bug in the caller.
    field[..w].copy_from_slice(&s.as_bytes()[s.len() - w..]);
    field[w] = 0;
}

fn raw_header(name: &[u8], kind: u8, mode: u32, size: u64, mtime: u64, link: &[u8]) -> [u8; BLOCK] {
    let mut h = [0u8; BLOCK];
    let n = name.len().min(100);
    h[..n].copy_from_slice(&name[..n]);
    put_octal(&mut h[100..108], u64::from(mode & 0o7777));
    put_octal(&mut h[108..116], 0);
    put_octal(&mut h[116..124], 0);
    if size >= 8u64.pow(11) {
        // Base-256: the high bit of the first byte, then the number big-endian.
        h[124] = 0x80;
        h[128..136].copy_from_slice(&size.to_be_bytes());
    } else {
        put_octal(&mut h[124..136], size);
    }
    put_octal(&mut h[136..148], mtime);
    h[156] = kind;
    let l = link.len().min(100);
    h[157..157 + l].copy_from_slice(&link[..l]);
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].fill(b' ');
    let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
    let c = format!("{sum:06o}");
    h[148..154].copy_from_slice(c.as_bytes());
    h[154] = 0;
    h[155] = b' ';
    h
}

/// A GNU long-name (`L`) or long-link (`K`) member holding `text`.
fn extension(kind: u8, text: &[u8], out: &mut Vec<u8>) {
    let size = text.len() as u64 + 1;
    out.extend_from_slice(&raw_header(b"././@LongLink", kind, 0, size, 0, b""));
    out.extend_from_slice(text);
    out.push(0);
    pad_to_block(out, size);
}

pub(super) fn pad_to_block(out: &mut Vec<u8>, size: u64) {
    let rem = (size % BLOCK as u64) as usize;
    if rem != 0 {
        out.resize(out.len() + BLOCK - rem, 0);
    }
}

/// The header (preceded by its `L`/`K` members when the name or target is over 100 bytes) of one
/// member, appended to `out`. `name` is relative with no trailing slash; a directory gets one.
pub(super) fn write_header(out: &mut Vec<u8>, e: &Entry) {
    let mut name = e.name.clone();
    if e.kind == b'5' {
        name.push(b'/');
    }
    if name.len() > 100 {
        extension(b'L', &name, out);
    }
    if e.link.len() > 100 {
        extension(b'K', &e.link, out);
    }
    out.extend_from_slice(&raw_header(
        &name,
        e.kind,
        e.mode & 0o777,
        e.size,
        e.mtime,
        &e.link,
    ));
}

/// The end of the archive: two zero blocks.
pub(super) fn write_end(out: &mut Vec<u8>) {
    out.resize(out.len() + 2 * BLOCK, 0);
}

fn str_field(f: &[u8]) -> &[u8] {
    &f[..f.iter().position(|&b| b == 0).unwrap_or(f.len())]
}

fn number(f: &[u8], what: &str) -> Result<u64, BulkError> {
    if f[0] & 0x80 != 0 {
        if f[0] == 0xff {
            return Err(BulkError::Archive(format!(
                "negative {what} in a tar header"
            )));
        }
        let mut v: u64 = u64::from(f[0] & 0x7f);
        for &b in &f[1..] {
            v = v
                .checked_mul(256)
                .and_then(|v| v.checked_add(u64::from(b)))
                .ok_or_else(|| BulkError::Archive(format!("{what} overflows in a tar header")))?;
        }
        return Ok(v);
    }
    let mut v: u64 = 0;
    let mut seen = false;
    for &b in f {
        match b {
            b'0'..=b'7' => {
                seen = true;
                v = v
                    .checked_mul(8)
                    .and_then(|v| v.checked_add(u64::from(b - b'0')))
                    .ok_or_else(|| {
                        BulkError::Archive(format!("{what} overflows in a tar header"))
                    })?;
            }
            b' ' | 0 if !seen => {}
            b' ' | 0 => break,
            _ => return Err(BulkError::Archive(format!("bad {what} in a tar header"))),
        }
    }
    Ok(v)
}

fn checksum_ok(h: &[u8]) -> bool {
    let want = match number(&h[148..156], "checksum") {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut unsigned: u64 = 0;
    let mut signed: i64 = 0;
    for (i, &b) in h.iter().enumerate() {
        let b = if (148..156).contains(&i) { b' ' } else { b };
        unsigned += u64::from(b);
        signed += i64::from(b as i8);
    }
    want == unsigned || i64::try_from(want).is_ok_and(|w| w == signed)
}

/// What a parser tells its consumer.
pub(super) trait Sink {
    fn begin(&mut self, e: &Entry) -> Result<(), BulkError>;
    fn data(&mut self, bytes: &[u8]) -> Result<(), BulkError>;
    fn end(&mut self) -> Result<(), BulkError>;
}

enum State {
    Header,
    /// Content of a member the consumer sees, then padding.
    Data {
        left: u64,
        pad: usize,
    },
    /// Content of a meta member (`L`, `K`, `x`, `g`) being collected, then padding.
    Meta {
        kind: u8,
        left: u64,
        pad: usize,
        buf: Vec<u8>,
    },
    /// Padding after a member with no content to deliver.
    Pad(usize),
    Done,
}

/// The longest meta member (long name, long link, pax record block) the reader holds. A path is at
/// most 4096 bytes on Linux; pax blocks carry a few records.
const META_MAX: u64 = 1 << 20;

/// An incremental reader of the archive subset: [`feed`](Self::feed) it chunks as they arrive.
pub(super) struct Parser {
    buf: Vec<u8>,
    state: State,
    zero_blocks: u8,
    long_name: Option<Vec<u8>>,
    long_link: Option<Vec<u8>>,
    pax: Pax,
}

#[derive(Default)]
struct Pax {
    path: Option<Vec<u8>>,
    link: Option<Vec<u8>>,
    size: Option<u64>,
}

impl Parser {
    pub(super) fn new() -> Self {
        Self {
            buf: Vec::new(),
            state: State::Header,
            zero_blocks: 0,
            long_name: None,
            long_link: None,
            pax: Pax::default(),
        }
    }

    pub(super) fn feed(&mut self, mut input: &[u8], sink: &mut dyn Sink) -> Result<(), BulkError> {
        loop {
            match &mut self.state {
                State::Done => return Ok(()),
                State::Header => {
                    let need = BLOCK - self.buf.len();
                    let take = need.min(input.len());
                    self.buf.extend_from_slice(&input[..take]);
                    input = &input[take..];
                    if self.buf.len() < BLOCK {
                        return Ok(());
                    }
                    let block = std::mem::take(&mut self.buf);
                    self.header(&block, sink)?;
                }
                State::Data { left, pad } => {
                    if *left > 0 {
                        let take = (*left).min(input.len() as u64) as usize;
                        if take == 0 {
                            return Ok(());
                        }
                        sink.data(&input[..take])?;
                        input = &input[take..];
                        *left -= take as u64;
                        if *left > 0 {
                            return Ok(());
                        }
                    }
                    let pad = *pad;
                    sink.end()?;
                    self.state = State::Pad(pad);
                }
                State::Meta { left, buf, .. } => {
                    let take = (*left).min(input.len() as u64) as usize;
                    buf.extend_from_slice(&input[..take]);
                    input = &input[take..];
                    *left -= take as u64;
                    if *left > 0 {
                        return Ok(());
                    }
                    let State::Meta { kind, pad, buf, .. } =
                        std::mem::replace(&mut self.state, State::Header)
                    else {
                        unreachable!()
                    };
                    self.meta(kind, buf)?;
                    self.state = State::Pad(pad);
                }
                State::Pad(n) => {
                    let take = (*n).min(input.len());
                    input = &input[take..];
                    *n -= take;
                    if *n > 0 {
                        return Ok(());
                    }
                    self.state = State::Header;
                }
            }
        }
    }

    /// The input is over: it must have ended at the end-of-archive block.
    pub(super) fn finish(&self) -> Result<(), BulkError> {
        match self.state {
            State::Done => Ok(()),
            State::Header if self.buf.is_empty() && self.zero_blocks > 0 => Ok(()),
            _ => Err(BulkError::Archive(
                "the archive ended before its end-of-archive block".into(),
            )),
        }
    }

    fn header(&mut self, h: &[u8], sink: &mut dyn Sink) -> Result<(), BulkError> {
        if h.iter().all(|&b| b == 0) {
            self.zero_blocks += 1;
            if self.zero_blocks >= 2 {
                self.state = State::Done;
            }
            return Ok(());
        }
        self.zero_blocks = 0;
        if !checksum_ok(h) {
            return Err(BulkError::Archive("bad tar header checksum".into()));
        }
        let kind = if h[156] == 0 { b'0' } else { h[156] };
        let mut size = number(&h[124..136], "size")?;
        let pad = |size: u64| ((BLOCK as u64 - size % BLOCK as u64) % BLOCK as u64) as usize;
        if matches!(kind, b'L' | b'K' | b'x' | b'g') {
            if size > META_MAX {
                return Err(BulkError::Archive(format!(
                    "a {size}-byte tar extension header"
                )));
            }
            self.state = State::Meta {
                kind,
                left: size,
                pad: pad(size),
                buf: Vec::with_capacity(size as usize),
            };
            return Ok(());
        }
        let mut name = str_field(&h[0..100]).to_vec();
        // POSIX ustar only: a GNU header ("ustar  \0") keeps other fields where `prefix` is.
        let ustar = &h[257..263] == b"ustar\0";
        if ustar {
            let prefix = str_field(&h[345..500]);
            if !prefix.is_empty() {
                let mut full = prefix.to_vec();
                full.push(b'/');
                full.extend_from_slice(&name);
                name = full;
            }
        }
        let mut link = str_field(&h[157..257]).to_vec();
        if let Some(n) = self.long_name.take() {
            name = n;
        }
        if let Some(l) = self.long_link.take() {
            link = l;
        }
        let pax = std::mem::take(&mut self.pax);
        if let Some(p) = pax.path {
            name = p;
        }
        if let Some(l) = pax.link {
            link = l;
        }
        if let Some(s) = pax.size {
            size = s;
        }
        let entry = Entry {
            name,
            kind,
            mode: number(&h[100..108], "mode")? as u32,
            size,
            mtime: number(&h[136..148], "mtime").unwrap_or(0),
            link,
        };
        // Only regular files carry content; a link or directory with a size has none.
        let content = if matches!(kind, b'0' | b'7') { size } else { 0 };
        sink.begin(&entry)?;
        self.state = State::Data {
            left: content,
            pad: pad(content),
        };
        if content == 0 {
            // `Data` with nothing left ends the member on the next pass.
        }
        Ok(())
    }

    fn meta(&mut self, kind: u8, buf: Vec<u8>) -> Result<(), BulkError> {
        let text = str_field(&buf).to_vec();
        match kind {
            b'L' => self.long_name = Some(text),
            b'K' => self.long_link = Some(text),
            b'x' => {
                // `LEN KEY=VALUE\n`, LEN counting the whole record.
                let mut rest = &buf[..];
                while !rest.is_empty() {
                    let sp = rest
                        .iter()
                        .position(|&b| b == b' ')
                        .ok_or_else(|| BulkError::Archive("bad pax record".into()))?;
                    let len: usize = std::str::from_utf8(&rest[..sp])
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| BulkError::Archive("bad pax record length".into()))?;
                    if len <= sp + 1 || len > rest.len() {
                        return Err(BulkError::Archive("bad pax record length".into()));
                    }
                    let rec = &rest[sp + 1..len];
                    let rec = rec.strip_suffix(b"\n").unwrap_or(rec);
                    if let Some(eq) = rec.iter().position(|&b| b == b'=') {
                        let (k, v) = (&rec[..eq], &rec[eq + 1..]);
                        match k {
                            b"path" => self.pax.path = Some(v.to_vec()),
                            b"linkpath" => self.pax.link = Some(v.to_vec()),
                            b"size" => {
                                self.pax.size =
                                    std::str::from_utf8(v).ok().and_then(|s| s.parse().ok());
                            }
                            _ => {}
                        }
                    }
                    rest = &rest[len..];
                }
            }
            _ => {} // `g`: global records are not used.
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Collect {
        seen: Vec<(Entry, Vec<u8>)>,
    }
    impl Sink for Collect {
        fn begin(&mut self, e: &Entry) -> Result<(), BulkError> {
            self.seen.push((e.clone(), Vec::new()));
            Ok(())
        }
        fn data(&mut self, b: &[u8]) -> Result<(), BulkError> {
            self.seen.last_mut().unwrap().1.extend_from_slice(b);
            Ok(())
        }
        fn end(&mut self) -> Result<(), BulkError> {
            Ok(())
        }
    }

    fn entry(name: &str, kind: u8, size: u64, link: &str) -> Entry {
        Entry {
            name: name.into(),
            kind,
            mode: 0o640,
            size,
            mtime: 1_700_000_000,
            link: link.into(),
        }
    }

    fn archive(parts: &[(Entry, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (e, data) in parts {
            write_header(&mut out, e);
            out.extend_from_slice(data);
            pad_to_block(&mut out, data.len() as u64);
        }
        write_end(&mut out);
        out
    }

    #[test]
    fn what_is_written_is_read_back_at_every_chunk_size() {
        let long = "d".repeat(150);
        let tgt = "t".repeat(130);
        let a = archive(&[
            (entry("a.txt", b'0', 5, ""), b"hello"),
            (entry("sub", b'5', 0, ""), b""),
            (entry(&format!("sub/{long}"), b'0', 3, ""), b"xyz"),
            (entry("l", b'2', 0, &tgt), b""),
            (entry("e", b'0', 0, ""), b""),
        ]);
        for step in [1usize, 7, 511, 512, 513, 100_000] {
            let mut p = Parser::new();
            let mut c = Collect::default();
            for chunk in a.chunks(step) {
                p.feed(chunk, &mut c).unwrap();
            }
            p.finish().unwrap();
            let names: Vec<_> = c
                .seen
                .iter()
                .map(|(e, _)| String::from_utf8(e.name.clone()).unwrap())
                .collect();
            assert_eq!(
                names,
                ["a.txt", "sub/", &format!("sub/{long}"), "l", "e"],
                "step {step}"
            );
            assert_eq!(c.seen[0].1, b"hello");
            assert_eq!(c.seen[2].1, b"xyz");
            assert_eq!(c.seen[3].0.link, tgt.as_bytes());
            assert_eq!(c.seen[0].0.mode, 0o640);
            assert_eq!(c.seen[0].0.mtime, 1_700_000_000);
        }
    }

    #[test]
    fn base_256_sizes_round_trip() {
        let h = raw_header(b"big", b'0', 0o644, 9 << 30, 0, b"");
        assert_eq!(number(&h[124..136], "size").unwrap(), 9 << 30);
    }

    #[test]
    fn a_bad_checksum_and_a_missing_end_are_refused() {
        let mut a = archive(&[(entry("a", b'0', 1, ""), b"x")]);
        let mut p = Parser::new();
        let mut c = Collect::default();
        assert!(p.feed(&a[..a.len() - 1024], &mut c).is_ok());
        assert!(p.finish().is_err(), "no end block");
        a[0] ^= 1;
        let mut p = Parser::new();
        assert!(matches!(
            p.feed(&a, &mut Collect::default()),
            Err(BulkError::Archive(_))
        ));
    }

    #[test]
    fn pax_path_and_size_override_the_header() {
        let rec = |k: &str, v: &str| {
            let body = format!(" {k}={v}\n");
            let mut len = body.len() + 1;
            while len.to_string().len() + body.len() != len {
                len = len.to_string().len() + body.len();
            }
            format!("{len}{body}")
        };
        let recs = format!("{}{}", rec("path", "pax/name"), rec("size", "3"));
        let mut out = Vec::new();
        out.extend_from_slice(&raw_header(
            b"PaxHeader",
            b'x',
            0,
            recs.len() as u64,
            0,
            b"",
        ));
        out.extend_from_slice(recs.as_bytes());
        pad_to_block(&mut out, recs.len() as u64);
        out.extend_from_slice(&raw_header(b"short", b'0', 0o600, 0, 0, b""));
        out.extend_from_slice(b"abc");
        pad_to_block(&mut out, 3);
        write_end(&mut out);
        let mut p = Parser::new();
        let mut c = Collect::default();
        p.feed(&out, &mut c).unwrap();
        p.finish().unwrap();
        assert_eq!(c.seen[0].0.name, b"pax/name");
        assert_eq!(c.seen[0].1, b"abc");
    }
}
