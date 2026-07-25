// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! `web` extract/table bench with **allocations** beside timing (`divan`'s `AllocProfiler`, the global
//! allocator below). The subject is [M17]: extract/table rows used to be a `HashMap<String,String>` per
//! row, cloning the field/header *name* into every row for every field — N×M String clones + hashing —
//! purely so the `where` filter could look a field up by name. The rows are now positional `Vec<String>`
//! aligned to the fixed field list, and the filter gets a name→index lookup computed once.
//!
//! **What this measures, and its limitation.** The only public seam into the extract path is the whole
//! `web` tool (`extract`/`whereexpr` are private modules), so this drives `Web::run` against a loopback
//! fixture serving a synthetic 500×8 table — a real end-to-end call. The alloc/ns totals therefore
//! include the HTTP fetch and the `scraper` HTML parse, which are **identical before and after** this
//! change; the *delta* between the pre- and post-change runs is attributable to row storage alone. Read
//! the columns as "cost of a whole extract call", and the before/after gap as the prize for M17. The
//! three arms isolate the three affected paths: `extract` (per-field name clone), `extract_where` (adds
//! the filter lookup), and `table` (per-header name clone).
//!
//! Run: `cargo bench -p beyond-ai-agent --bench web_extract`.

use std::hint::black_box;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::OnceLock;

use agent_core::Tool;
use beyond_ai_agent::tools::web::Web;
use divan::Bencher;
use serde_json::json;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const ROWS: usize = 500;
const COLS: usize = 8;

/// A synthetic HTML page: one big `<table>` (500 rows × 8 columns) wrapped so both `table` mode and
/// `extract` mode (rows = `tr`, fields = per-`td` selectors) have something realistic to chew on. Built
/// once and served verbatim to every request.
fn page_html() -> &'static str {
    static HTML: OnceLock<String> = OnceLock::new();
    HTML.get_or_init(|| {
        let mut s = String::with_capacity(ROWS * COLS * 24);
        s.push_str("<html><body><table>");
        s.push_str("<tr>");
        for c in 0..COLS {
            s.push_str(&format!("<th>col{c}</th>"));
        }
        s.push_str("</tr>");
        for r in 0..ROWS {
            s.push_str("<tr>");
            for c in 0..COLS {
                // Column 0 is a monotonically increasing number so the `where` filter has a numeric
                // field to compare; the rest are short text cells.
                if c == 0 {
                    s.push_str(&format!("<td class=\"c{c}\">{r}</td>"));
                } else {
                    s.push_str(&format!("<td class=\"c{c}\">cell {r}-{c} value</td>"));
                }
            }
            s.push_str("</tr>");
        }
        s.push_str("</table></body></html>");
        s
    })
}

/// A persistent loopback HTTP/1.1 server that answers every connection with the synthetic page. Spawned
/// once; the thread lives for the whole bench run. Returns the base URL (e.g. `http://127.0.0.1:PORT`).
fn fixture() -> &'static str {
    static BASE: OnceLock<String> = OnceLock::new();
    BASE.get_or_init(|| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = page_html();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { break };
                // Drain the request (we serve the same page regardless), then reply.
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/")
    })
}

/// The tool with the fixture's loopback host allow-listed (extract needs no other egress).
fn tool() -> Web {
    Web::new(false, &["127.0.0.1".to_string()], Some(5000))
}

fn run(args: serde_json::Value) -> usize {
    let url = fixture().to_string();
    let mut args = args;
    args.as_object_mut()
        .unwrap()
        .insert("url".into(), json!(url));
    let tool = tool();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let out = rt.block_on(tool.run(args)).unwrap();
    out.text.len()
}

/// `extract` mode: 500 rows × 8 text fields — the path that cloned each field name into a per-row map.
#[divan::bench(sample_count = 20)]
fn extract(bencher: Bencher) {
    let fields = (0..COLS)
        .map(|c| format!("f{c}=.c{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    bencher.bench_local(|| {
        black_box(run(json!({
            "mode": "extract",
            "selector": "tr",
            "fields": fields,
            "max_items": ROWS,
        })));
    });
}

/// `extract` with a `where` filter — exercises the name→index lookup that replaced the per-row map key.
#[divan::bench(sample_count = 20)]
fn extract_where(bencher: Bencher) {
    let fields = (0..COLS)
        .map(|c| format!("f{c}=.c{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    bencher.bench_local(|| {
        black_box(run(json!({
            "mode": "extract",
            "selector": "tr",
            "fields": fields,
            "where": "f0 >= 100",
            "max_items": ROWS,
        })));
    });
}

/// `table` mode: the path that cloned every header string into each row's map.
#[divan::bench(sample_count = 20)]
fn table(bencher: Bencher) {
    bencher.bench_local(|| {
        black_box(run(json!({
            "mode": "table",
            "max_items": ROWS,
        })));
    });
}
