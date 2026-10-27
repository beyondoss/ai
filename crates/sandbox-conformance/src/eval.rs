//! The scenario language of `sdk/conformance/DRIVER.md`: variable substitution, derived views of
//! base64 fields, and the matchers. Pure functions over JSON; no I/O.
use std::collections::BTreeMap;

use base64::Engine as _;
use serde_json::{Value, json};
use sha2::Digest as _;

pub const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

pub type Vars = BTreeMap<String, Value>;

/// Replace `${name}` in every string. A string that is exactly one variable becomes the saved
/// value, type and all; otherwise the value is interpolated as text. `{"$repeat": ...}` expands to
/// base64.
pub fn subst(v: &Value, vars: &Vars) -> Value {
    match v {
        Value::String(s) => {
            if let Some(name) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}'))
                && !name.contains("${")
                && let Some(val) = vars.get(name)
            {
                return val.clone();
            }
            let mut out = s.clone();
            for (k, val) in vars {
                let text = match val {
                    Value::String(t) => t.clone(),
                    other => other.to_string(),
                };
                out = out.replace(&format!("${{{k}}}"), &text);
            }
            Value::String(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| subst(x, vars)).collect()),
        Value::Object(o) => {
            if o.len() == 1
                && let Some(r) = o.get("$repeat")
            {
                let chunk = B64
                    .decode(r["b64"].as_str().expect("$repeat.b64"))
                    .expect("$repeat.b64 is base64");
                let times = r["times"].as_u64().expect("$repeat.times") as usize;
                let mut bytes = chunk.repeat(times);
                if let Some(len) = r["len"].as_u64() {
                    bytes.truncate(len as usize);
                }
                return Value::String(B64.encode(bytes));
            }
            Value::Object(o.iter().map(|(k, x)| (k.clone(), subst(x, vars))).collect())
        }
        other => other.clone(),
    }
}

/// Add `<name>_utf8`, `<name>_len` and `<name>_sha256` beside every `<name>_b64` string, so a
/// scenario can state an expectation about bytes in text or by digest.
pub fn derive(v: &mut Value) {
    match v {
        Value::Object(o) => {
            let mut extra = Vec::new();
            for (k, x) in o.iter() {
                if let (Some(base), Value::String(s)) = (k.strip_suffix("_b64"), x) {
                    let bytes = B64.decode(s).unwrap_or_default();
                    extra.push((
                        format!("{base}_utf8"),
                        json!(String::from_utf8_lossy(&bytes)),
                    ));
                    extra.push((format!("{base}_len"), json!(bytes.len())));
                    extra.push((
                        format!("{base}_sha256"),
                        json!(hex::encode(sha2::Sha256::digest(&bytes))),
                    ));
                }
            }
            o.extend(extra);
            for x in o.values_mut() {
                derive(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(derive),
        _ => {}
    }
}

pub fn dotted<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(v, |cur, key| match cur {
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => cur.get(key),
    })
}

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

pub fn matches(exp: &Value, act: &Value, at: &str) -> Result<(), String> {
    if let Value::Object(o) = exp
        && o.len() == 1
        && let Some((k, arg)) = o.iter().next()
        && k.starts_with('$')
    {
        return matcher(k, arg, act, at);
    }
    match (exp, act) {
        (Value::Object(e), Value::Object(a)) => {
            for (k, ev) in e {
                matches(ev, a.get(k).unwrap_or(&Value::Null), &format!("{at}.{k}"))?;
            }
            Ok(())
        }
        (Value::Object(e), other) => Err(format!("{at}: expected an object {e:?}, got {other}")),
        (Value::Array(e), Value::Array(a)) => {
            if e.len() != a.len() {
                return Err(format!(
                    "{at}: expected {} items, got {}: {act}",
                    e.len(),
                    a.len()
                ));
            }
            for (i, (ev, av)) in e.iter().zip(a).enumerate() {
                matches(ev, av, &format!("{at}[{i}]"))?;
            }
            Ok(())
        }
        (Value::Number(e), Value::Number(a)) if e.as_f64() == a.as_f64() => Ok(()),
        (e, a) if e == a => Ok(()),
        (e, a) => Err(format!("{at}: expected {e}, got {a}")),
    }
}

fn matcher(name: &str, arg: &Value, act: &Value, at: &str) -> Result<(), String> {
    let fail = |why: String| Err(format!("{at}: {name} {arg}: {why} (got {act})"));
    match name {
        "$any" => {
            if act.is_null() {
                return fail("absent".into());
            }
            Ok(())
        }
        "$absent" => {
            if act.is_null() {
                Ok(())
            } else {
                fail("present".into())
            }
        }
        "$type" => {
            let t = match act {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            if Some(t) == arg.as_str() {
                Ok(())
            } else {
                fail(format!("is {t}"))
            }
        }
        "$re" => {
            let re = regex::Regex::new(arg.as_str().unwrap()).expect("$re is a regex");
            match act.as_str() {
                Some(s) if re.is_match(s) => Ok(()),
                _ => fail("no match".into()),
            }
        }
        "$contains" => match act.as_str() {
            Some(s) if s.contains(arg.as_str().unwrap()) => Ok(()),
            _ => fail("not contained".into()),
        },
        "$gte" | "$lte" | "$gt" | "$lt" => {
            let (a, b) = (act.as_f64(), arg.as_f64());
            let ok = match (name, a, b) {
                ("$gte", Some(a), Some(b)) => a >= b,
                ("$lte", Some(a), Some(b)) => a <= b,
                ("$gt", Some(a), Some(b)) => a > b,
                ("$lt", Some(a), Some(b)) => a < b,
                _ => false,
            };
            if ok {
                Ok(())
            } else {
                fail("comparison failed".into())
            }
        }
        "$oneof" => {
            let any = arg
                .as_array()
                .unwrap()
                .iter()
                .any(|e| matches(e, act, at).is_ok());
            if any {
                Ok(())
            } else {
                fail("none equal".into())
            }
        }
        "$not" => {
            if matches(arg, act, at).is_err() {
                Ok(())
            } else {
                fail("matched".into())
            }
        }
        "$len" => {
            let n = act
                .as_array()
                .map(Vec::len)
                .or_else(|| act.as_str().map(str::len));
            if n == arg.as_u64().map(|n| n as usize) {
                Ok(())
            } else {
                fail(format!("length {n:?}"))
            }
        }
        // The same items in any order.
        "$unordered" | "$includes" => {
            let Some(items) = act.as_array() else {
                return fail("not an array".into());
            };
            let want = arg.as_array().unwrap();
            if name == "$unordered" && want.len() != items.len() {
                return fail(format!("{} items", items.len()));
            }
            let mut used = vec![false; items.len()];
            for w in want {
                let slot = items
                    .iter()
                    .enumerate()
                    .position(|(i, a)| !used[i] && matches(w, a, at).is_ok());
                match slot {
                    Some(i) => used[i] = true,
                    None => return fail(format!("no item matches {w}")),
                }
            }
            Ok(())
        }
        other => Err(format!("{at}: unknown matcher {other}")),
    }
}
