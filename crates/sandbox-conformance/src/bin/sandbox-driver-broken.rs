//! A deliberately broken driver, to prove the harness catches a bad client: it runs the driver given
//! in `BROKEN_INNER` and corrupts the answer of one op, named by `BROKEN_OP` (`read`: a byte of the
//! data is flipped; `stat`: the size is off by one; `exec`: a byte of stdout is flipped).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Command, Stdio};

use base64::Engine as _;
use serde_json::Value;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn flip(v: &mut Value, key: &str) {
    if let Some(s) = v.get(key).and_then(Value::as_str) {
        let mut bytes = B64.decode(s).unwrap();
        if let Some(b) = bytes.first_mut() {
            *b ^= 1;
        }
        v[key] = Value::String(B64.encode(bytes));
    }
}

fn main() {
    let inner = std::env::var("BROKEN_INNER").expect("BROKEN_INNER");
    let broken = std::env::var("BROKEN_OP").expect("BROKEN_OP");
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(format!("exec {inner}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut to = child.stdin.take().unwrap();
    let mut from = BufReader::new(child.stdout.take().unwrap());
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let req: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        writeln!(to, "{line}").unwrap();
        to.flush().unwrap();
        let mut answer = String::new();
        from.read_line(&mut answer).unwrap();
        let mut resp: Value = serde_json::from_str(&answer).unwrap();
        if req["op"] == broken.as_str() && resp.get("ok").is_some() {
            match broken.as_str() {
                "read" => flip(&mut resp["ok"], "data_b64"),
                "exec" => flip(&mut resp["ok"], "stdout_b64"),
                "stat" => {
                    if let Some(n) = resp["ok"]["size"].as_u64() {
                        resp["ok"]["size"] = (n + 1).into();
                    }
                }
                other => panic!("cannot break {other}"),
            }
        }
        writeln!(out, "{resp}").unwrap();
        out.flush().unwrap();
    }
    drop(to);
    let _ = child.wait();
}
