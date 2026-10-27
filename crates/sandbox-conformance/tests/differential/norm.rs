//! What two drivers' answers are compared as.
//!
//! The comparison is byte-for-byte on the JSON a driver answered, after exactly these changes, each
//! of which the spec allows to differ between two runs of the same operation:
//!
//! | what                          | becomes                          | why                                            |
//! | ----------------------------- | -------------------------------- | ---------------------------------------------- |
//! | `id`                          | dropped                          | a per-driver request counter                   |
//! | `error.detail`                | dropped                          | "free text for humans" (DRIVER.md section 2)   |
//! | the case's root path (also inside `*_b64` output) | `${ROOT}`           | every driver works in its own root             |
//! | `stat.mtime`                  | `{"$type":"number"}`             | wall clock at the time of the write            |
//! | `size` of a directory         | `{"$type":"number"}`             | the filesystem's business                      |
//! | `list.entries` order          | sorted, as `$unordered`          | "order unspecified" (FILES-OVER-EXEC 5.6)      |
//! | `search.note` (when present)  | `{"$type":"string"}`             | text of the engine's first error, has paths    |
//!
//! The result is a valid `expect` for a scenario step, which is how a divergence is printed.
use sandbox_conformance::eval::B64;
use base64::Engine as _;
use serde_json::{Value, json};

fn replace_root(v: &mut Value, root: &str) {
    match v {
        Value::String(s) => {
            if s.contains(root) {
                *s = s.replace(root, "${ROOT}");
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| replace_root(x, root)),
        Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                // Output bytes (an error message from `mkdir`) name the path too.
                if k.ends_with("_b64")
                    && let Value::String(s) = x
                    && let Ok(bytes) = B64.decode(s.as_bytes())
                {
                    let text = String::from_utf8_lossy(&bytes).replace(root, "${ROOT}");
                    *s = B64.encode(text.as_bytes());
                } else {
                    replace_root(x, root);
                }
            }
        }
        _ => {}
    }
}

fn any_number() -> Value {
    json!({"$type": "number"})
}

pub fn canon(op: &str, resp: &Value, root: &str) -> Value {
    if let Some(e) = resp.get("error") {
        return json!({"error": {"kind": e["kind"]}});
    }
    let mut ok = resp.get("ok").cloned().unwrap_or(Value::Null);
    match op {
        "stat" if ok["exists"] == true => {
            ok["mtime"] = any_number();
            if ok["kind"] == "dir" {
                ok["size"] = any_number();
            }
        }
        "list" => {
            if let Some(Value::Array(entries)) = ok.get_mut("entries") {
                for e in entries.iter_mut() {
                    if e["kind"] == "dir" {
                        e["size"] = any_number();
                    }
                }
                entries.sort_by_key(|e| e["name"].as_str().unwrap_or("").as_bytes().to_vec());
                let sorted = std::mem::take(entries);
                ok["entries"] = json!({"$unordered": sorted});
            }
        }
        "search" if !ok["note"].is_null() => ok["note"] = json!({"$type": "string"}),
        _ => {}
    }
    replace_root(&mut ok, root);
    json!({"ok": ok})
}
