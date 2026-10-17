//! How `read`, `edit` and `write` tell text from binary, and UTF-8 from UTF-16.
//!
//! One classifier, shared by all three tools and run on bytes the [`FsBackend`](super::fs::FsBackend)
//! already returned — so a file is the same kind of file locally and in a sandbox, and the encoding
//! `read` decoded a file with is the one `edit` and `write` write it back in.
//!
//! - **Binary**: a NUL in the first [`SNIFF_LEN`] bytes that is not explained by UTF-16. `read`
//!   reports one line instead of decoding a `.dylib`/`.sqlite`/PDF into kilobytes of replacement
//!   characters (an ELF read with `limit: 5` used to return several KB, most of it NULs).
//! - **UTF-16**: a byte-order mark, or — for files written without one — text whose every other
//!   byte is zero in the pattern only UTF-16 produces. Windows tools (PowerShell redirection,
//!   `regedit` exports, some `.rc`/`.inf` files) write it; decoding it as UTF-8 showed `h e l l o`,
//!   and `edit` refused it. It is decoded for reading and re-encoded (same byte order, the BOM kept
//!   when it had one) for writing.

/// How many leading bytes are examined. Matches git's own binary heuristic (`buffer_is_binary`),
/// which looks for a NUL in the first 8000 bytes.
pub const SNIFF_LEN: usize = 8192;

/// What a file's bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// UTF-8, or anything else that is not one of the cases below (Latin-1 included: it is decoded
    /// lossily by `read` and refused by `edit`, as before).
    Utf8,
    /// UTF-16, little- or big-endian. A BOM, if present, decodes to a leading U+FEFF and is
    /// re-encoded from it, so it round-trips without being tracked separately.
    Utf16 { big_endian: bool },
    /// Not text.
    Binary,
}

/// Classify a file from its first bytes (at most [`SNIFF_LEN`] are looked at).
pub fn sniff(bytes: &[u8]) -> Encoding {
    let head = &bytes[..bytes.len().min(SNIFF_LEN)];
    match head {
        [0xFF, 0xFE, ..] => return Encoding::Utf16 { big_endian: false },
        [0xFE, 0xFF, ..] => return Encoding::Utf16 { big_endian: true },
        _ => {}
    }
    if !head.contains(&0) {
        return Encoding::Utf8;
    }
    for big_endian in [false, true] {
        if looks_like_bomless_utf16(head, big_endian) {
            return Encoding::Utf16 { big_endian };
        }
    }
    Encoding::Binary
}

/// UTF-16 text with no BOM: no NUL code unit, at least 70% of units in the ASCII range (whose high
/// byte is the zero that put us here), and no ASCII control character other than whitespace. A
/// binary format full of zero bytes fails the last two; CJK-heavy UTF-16 has few zero bytes, and
/// without a BOM is reported as binary rather than guessed at.
fn looks_like_bomless_utf16(head: &[u8], big_endian: bool) -> bool {
    let units = head.len() / 2;
    if units == 0 {
        return false;
    }
    let mut ascii = 0usize;
    for pair in head.chunks_exact(2) {
        let u = unit(pair, big_endian);
        match u {
            0 => return false,
            0x01..=0x08 | 0x0B | 0x0E..=0x1F | 0x7F => return false,
            0x09..=0x7E => ascii += 1,
            _ => {}
        }
    }
    ascii * 10 >= units * 7
}

/// One UTF-16 code unit from a byte pair in the given order.
pub fn unit(pair: &[u8], big_endian: bool) -> u16 {
    let b = [pair[0], pair[1]];
    if big_endian {
        u16::from_be_bytes(b)
    } else {
        u16::from_le_bytes(b)
    }
}

/// Decode a whole UTF-16 file. `None` for an odd byte count or an unpaired surrogate — not text this
/// tool can edit without changing bytes it was not asked to change.
pub fn decode_utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    char::decode_utf16(bytes.chunks_exact(2).map(|p| unit(p, big_endian)))
        .collect::<Result<String, _>>()
        .ok()
}

/// Encode `text` as UTF-16 in the given byte order. A leading U+FEFF becomes the BOM.
pub fn encode_utf16(text: &str, big_endian: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() * 2);
    for u in text.encode_utf16() {
        out.extend_from_slice(&if big_endian {
            u.to_be_bytes()
        } else {
            u.to_le_bytes()
        });
    }
    out
}

/// The name `read` shows for a decoded file, so the model knows what `edit` will preserve.
pub fn utf16_label(big_endian: bool) -> &'static str {
    if big_endian { "UTF-16BE" } else { "UTF-16LE" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le(s: &str) -> Vec<u8> {
        encode_utf16(s, false)
    }

    #[test]
    fn plain_text_and_latin1_are_utf8() {
        assert_eq!(sniff(b"hello\nworld\n"), Encoding::Utf8);
        assert_eq!(sniff(b"caf\xe9\n"), Encoding::Utf8);
        assert_eq!(sniff(b""), Encoding::Utf8);
    }

    #[test]
    fn a_bom_selects_utf16_and_its_byte_order() {
        assert_eq!(
            sniff(&le("\u{feff}hi")),
            Encoding::Utf16 { big_endian: false }
        );
        assert_eq!(
            sniff(&encode_utf16("\u{feff}hi", true)),
            Encoding::Utf16 { big_endian: true }
        );
    }

    #[test]
    fn bomless_ascii_utf16_is_recognized_in_either_order() {
        assert_eq!(
            sniff(&le("hello world\r\n")),
            Encoding::Utf16 { big_endian: false }
        );
        assert_eq!(
            sniff(&encode_utf16("hello world\r\n", true)),
            Encoding::Utf16 { big_endian: true }
        );
    }

    #[test]
    fn nul_bearing_binaries_are_binary() {
        // An ELF header: magic, then zero padding — NUL code units fail the UTF-16 test.
        let mut elf = b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0".to_vec();
        elf.extend_from_slice(&[0u8; 64]);
        assert_eq!(sniff(&elf), Encoding::Binary);
        assert_eq!(sniff(b"SQLite format 3\0\x10\0\x01\x01"), Encoding::Binary);
        assert_eq!(sniff(b"text\0text"), Encoding::Binary);
    }

    #[test]
    fn a_nul_past_the_sniff_window_does_not_make_a_file_binary() {
        let mut bytes = vec![b'a'; SNIFF_LEN];
        bytes.push(0);
        assert_eq!(sniff(&bytes), Encoding::Utf8);
    }

    #[test]
    fn utf16_round_trips_including_the_bom_and_astral_characters() {
        for big_endian in [false, true] {
            let text = "\u{feff}héllo 𝄞\r\n";
            let bytes = encode_utf16(text, big_endian);
            assert_eq!(decode_utf16(&bytes, big_endian).as_deref(), Some(text));
        }
    }

    #[test]
    fn malformed_utf16_does_not_decode() {
        assert_eq!(decode_utf16(b"\xff\xfeh", false), None); // odd length
        assert_eq!(decode_utf16(&[0x00, 0xD8, 0x41, 0x00], false), None); // lone surrogate
    }
}
