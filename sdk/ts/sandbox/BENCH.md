# Codec hot-path benches

Run (from this directory):

```sh
npm run bench:codec
```

Compares **legacy** (pre-fix, inlined in `scripts/bench-codec.ts`) against **current** production code. Node **v22.23.4**, median of 5–7 timed rounds after warmup.

## Results (2026-10-29)

| Suite | Input | Legacy | After / candidate | Speedup |
| --- | --- | --- | --- | --- |
| `excerpt` | 4 MiB body | 38.5 ms | **16 µs** | **~2400×** |
| `splitLines` (NDJSON) | 4096 lines, one chunk | 56.7 ms | **2.0 ms** | **~29×** |
| `fromBase64` | 256 KiB … 12 MiB | see below | JS-validate candidate | **rejected** |

### `fromBase64` — candidate rejected

A JS alphabet + trailing-bits pre-check, then a single `Buffer.from` (no re-encode), was measured against the existing native decode + re-encode compare:

| Decoded size | Legacy (decode+re-encode) | JS-validate + decode | Verdict |
| --- | --- | --- | --- |
| 256 KiB | 1.03 ms | 1.29 ms | legacy faster |
| 1 MiB | 4.47 ms | 5.65 ms | legacy faster |
| 4 MiB | 15.2 ms | 27.5 ms | legacy faster |
| 12 MiB | ~48–61 ms | ~57 ms | within noise / no clear win |

**Production keeps decode+re-encode.** Native `Buffer#toString("base64")` beats a full JS scan of the base64 string on Node.

## What shipped

1. **`excerpt`** — decode at most 1600 UTF-8 bytes, then take ≤400 code points (error paths can no longer `Array.from` a multi-MiB body).
2. **`HttpTransport.lines`** — copy each line once; compact the consumed prefix only when appending the next network chunk (avoids O(lines × remaining) remainder copies on dense NDJSON).
3. **`fromBase64`** — unchanged after measurement; comment documents why.
