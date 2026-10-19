use super::*;

// ---- Storage layout internals ---------------------------------------------------------------

#[test]
fn segment_names_round_trip_and_reject_everything_else() {
    assert_eq!(segment_name(1), "000001.jsonl");
    assert_eq!(segment_name(1_234_567), "1234567.jsonl");
    assert_eq!(segment_epoch("000001.jsonl"), Some(1));
    assert_eq!(segment_epoch("1234567.jsonl"), Some(1_234_567));
    // Anything that isn't a zero-padded decimal is not a segment — including the names the
    // listing cache, the lock and a stray temp file take.
    for name in [
        "lock",
        ".listings.json",
        "1.jsonl",
        "00001.jsonl",
        "00000a.jsonl",
        "000001.jsonl.tmp",
        "000001",
        "",
    ] {
        assert_eq!(
            segment_epoch(name),
            None,
            "{name} must not parse as a segment"
        );
    }
}

#[test]
fn a_segment_header_carries_every_seal_since_the_last_base() {
    let seals: BTreeMap<u64, u64> = [(3, 120), (4, 40)].into_iter().collect();
    let line = SegmentHeader::line(5, &seals, false).unwrap();
    assert_eq!(
        String::from_utf8(line.clone()).unwrap(),
        "{\"type\":\"segment\",\"epoch\":5,\"sealed\":{\"3\":120,\"4\":40},\"base\":false}\n",
        "the header is plaintext, stable and first"
    );
    let parsed: SegmentHeader = serde_json::from_slice(line.trim_ascii_end()).unwrap();
    assert_eq!(parsed.epoch, 5);
    assert!(!parsed.base);
    assert_eq!(parsed.seals(), seals);
    // A line that isn't a segment header is not mistaken for one.
    assert!(serde_json::from_slice::<SegmentHeader>(BASE_TRAILER).is_err());
}

#[test]
fn the_codec_binds_a_line_to_its_tenant_and_session() {
    let codec = TenantCodec::new("tenant-a", &[9u8; 32]);
    let sealed = codec.seal_line("s1", br#"{"type":"message"}"#).unwrap();
    assert!(
        sealed.starts_with(b"e1.") && sealed.contains(&b':'),
        "framed as e1.<kfp8>:<blob>"
    );
    assert_ne!(sealed.first(), Some(&b'{'), "never mistaken for plaintext");
    assert_eq!(
        codec.open_line("s1", &sealed).unwrap(),
        br#"{"type":"message"}"#
    );

    // Wrong session, wrong tenant, wrong key: all refused, none silently empty.
    assert!(codec.open_line("s2", &sealed).is_err());
    assert!(
        TenantCodec::new("tenant-b", &[9u8; 32])
            .open_line("s1", &sealed)
            .is_err()
    );
    assert!(
        TenantCodec::new("tenant-a", &[8u8; 32])
            .open_line("s1", &sealed)
            .is_err()
    );
    // And a flipped bit in the ciphertext, which is what the tag is for.
    let mut tampered = sealed.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    assert!(codec.open_line("s1", &tampered).is_err());

    // The failure is the kind a replay must not walk past.
    let err = codec.open_line("s2", &sealed).unwrap_err();
    assert!(is_seal_failure(&err), "a decode failure must be hard");
}

#[test]
fn the_codec_separates_its_three_surfaces() {
    let codec = TenantCodec::new("t", &[3u8; 32]);
    let line = codec.seal_line("s1", b"payload").unwrap();
    let listing = codec.seal_listing(b"payload").unwrap();
    let doc = codec.seal_doc(b"payload").unwrap();
    // One surface's key never opens another's.
    assert!(codec.open_listing(&line).is_err());
    assert!(codec.open_doc(&listing).is_err());
    assert!(codec.open_line("s1", &doc).is_err());
    assert_eq!(codec.open_listing(&listing).unwrap(), b"payload");
    assert_eq!(codec.open_doc(&doc).unwrap(), b"payload");
    // The nonce is fresh per call, so identical plaintext never seals identically.
    assert_ne!(codec.seal_doc(b"payload").unwrap(), doc);
    // Content that was never sealed reads back verbatim, so a store predating sealing still works.
    assert_eq!(codec.open_doc(b"plain").unwrap(), b"plain");
}

#[test]
fn sealing_a_buffer_frames_one_line_at_a_time() {
    let codec = TenantCodec::new("t", &[1u8; 32]);
    let mut out = Vec::new();
    append_sealed(&mut out, Some(&codec), "s1", b"{\"a\":1}\n{\"b\":2}\n").unwrap();
    let lines: Vec<&[u8]> = out
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(codec.open_line("s1", lines[0]).unwrap(), b"{\"a\":1}");
    assert_eq!(codec.open_line("s1", lines[1]).unwrap(), b"{\"b\":2}");

    // Without a codec it is a straight copy — the unsealed path costs nothing.
    let mut plain = Vec::new();
    append_sealed(&mut plain, None, "s1", b"{\"a\":1}\n").unwrap();
    assert_eq!(plain, b"{\"a\":1}\n");
}

#[test]
fn a_bounded_line_read_reports_where_it_stopped() {
    let data = b"one\ntwo\nthr";
    let mut reader = std::io::BufReader::new(&data[..]);
    let mut buf = Vec::new();
    let mut limit = None;

    let r = read_capped_line_limited(&mut reader, &mut buf, &mut limit)
        .unwrap()
        .unwrap();
    assert_eq!(
        (buf.as_slice(), r.terminated, r.consumed),
        (&b"one"[..], true, 4)
    );

    // A budget cuts the range short, and the truncated line reports itself unterminated.
    let mut limit = Some(2u64);
    let r = read_capped_line_limited(&mut reader, &mut buf, &mut limit)
        .unwrap()
        .unwrap();
    assert_eq!(
        (buf.as_slice(), r.terminated, limit),
        (&b"tw"[..], false, Some(0))
    );

    // Reading on past a spent budget is the end of the range, not the end of the file.
    assert!(
        read_capped_line_limited(&mut reader, &mut buf, &mut limit)
            .unwrap()
            .is_none()
    );

    // Unbounded again, the reader picks up mid-line exactly where the budget left it.
    let mut rest = None;
    let r = read_capped_line_limited(&mut reader, &mut buf, &mut rest)
        .unwrap()
        .unwrap();
    assert_eq!((buf.as_slice(), r.terminated), (&b"o"[..], true));

    // A file that ends without a newline is the torn-write case.
    let r = read_capped_line_limited(&mut reader, &mut buf, &mut rest)
        .unwrap()
        .unwrap();
    assert_eq!((buf.as_slice(), r.terminated), (&b"thr"[..], false));
    assert!(
        read_capped_line_limited(&mut reader, &mut buf, &mut rest)
            .unwrap()
            .is_none()
    );
}

/// An owner that already holds an open segment must still notice a successor.
///
/// The roll-time checks in `ensure_target` do not run here: this store has a target and appends
/// straight into it. Before the per-commit check existed, every one of these appends succeeded
/// and was acknowledged, while a reader — bounded by the seal the successor recorded — ignored
/// all of them. The fleet simulator caught it against a real NFSv4 lease; this pins it in
/// milliseconds, by doing to the directory exactly what a successor does: claiming the next
/// epoch with `O_EXCL`.
#[test]
fn an_owner_holding_a_segment_still_notices_a_successor() {
    let dir = tempfile::tempdir().unwrap();
    let repo = SessionRepo::open_with(
        dir.path(),
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: None,
        },
    )
    .unwrap();
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    let session_dir = store.path().to_path_buf();

    store.append_new(&[Message::user("mine")]).unwrap();
    assert!(!store.superseded(), "nobody else has touched it yet");
    let target = scan_segments(&session_dir).unwrap().last().unwrap().epoch;

    // What a successor does, and the only part of it that matters here.
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(session_dir.join(segment_name(target + 1)))
        .unwrap();

    let err = store
        .append_new(&[Message::user("mine"), Message::user("after the seal")])
        .expect_err("a fenced owner must not report a successful write");
    assert!(
        is_superseded(&err),
        "and it must be the takeover signal, not an I/O error: {err:?}"
    );
    assert!(
        store.superseded(),
        "poisoned, so the session ends rather than accepting the next turn too"
    );
}

#[test]
fn a_stamp_moves_for_every_kind_of_change() {
    let dir = tempfile::tempdir().unwrap();
    let repo = SessionRepo::open_with(
        dir.path(),
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: None,
        },
    )
    .unwrap();
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    let session_dir = store.path().to_path_buf();
    let first = log_stamp(&session_dir, true).unwrap();
    assert_eq!((first.epoch, first.segments), (1, 1));

    store.append_new(&[Message::user("one")]).unwrap();
    let appended = log_stamp(&session_dir, true).unwrap();
    assert!(
        appended.size > first.size,
        "an append grows the newest segment"
    );
    drop(store);

    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    let rolled = log_stamp(&session_dir, true).unwrap();
    assert_eq!(rolled.epoch, 2, "a roll moves the newest epoch");
    assert_eq!(rolled.segments, 2, "…and the segment count");

    // A base is what `base_epoch` used to be carried for, and the reason it no longer has to be:
    // `write_base` always creates `pred + 1` with `create_new`, so it can only ever *advance* the
    // newest epoch. There is no in-place rewrite for the stamp to miss.
    store.rewrite(&[Message::user("small")]).unwrap();
    let based = log_stamp(&session_dir, true).unwrap();
    assert_eq!(based.epoch, 3, "a base moves the newest epoch");
    assert_ne!(based, rolled);

    // A prune is the one change that shrinks rather than grows: the count catches it.
    fs::remove_file(session_dir.join(segment_name(1))).unwrap();
    let pruned = log_stamp(&session_dir, true).unwrap();
    assert_eq!(pruned.epoch, based.epoch);
    assert_eq!(pruned.segments, based.segments - 1);
    assert_ne!(pruned, based, "a prune must invalidate a cached listing");

    // A directory with no segments has no stamp at all — it is not a session.
    assert!(log_stamp(&dir.path().join("nothing"), true).is_none());
}
