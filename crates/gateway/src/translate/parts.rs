//! Content parts (images, files, text) and the URLs they carry, between the three wires.

use super::*;

/// Chat Completions `file` part → Anthropic `document`. Inline data (`file_data`, a data URI) maps
/// to a base64 source. A `file_id` names an OpenAI Files upload Anthropic cannot read, so the part
/// is forwarded as-is for the provider to reject by name.
pub(super) fn openai_file_to_anthropic(part: &Value) -> Value {
    let file = part.get("file");
    let Some((media_type, data)) = file
        .and_then(|f| f.get("file_data"))
        .and_then(Value::as_str)
        .and_then(parse_data_uri)
    else {
        return part.clone();
    };
    let mut doc = json!({
        "type": "document",
        "source": { "type": "base64", "media_type": media_type, "data": data },
    });
    if let Some(name) = file
        .and_then(|f| f.get("filename"))
        .filter(|n| n.is_string())
        && let Some(obj) = doc.as_object_mut()
    {
        obj.insert("title".into(), name.clone());
    }
    doc
}

/// A Chat Completions image part onto Messages. A data URI that is not base64
/// (`data:image/svg+xml;utf8,…`) has no Messages image source: it is forwarded for the provider to
/// reject, since dropped, the model would describe a picture it never saw. Other schemes
/// (`file://`) are never forwarded in any shape.
pub(super) fn openai_image_part_to_anthropic(p: &Value) -> Option<Value> {
    if let Some(b) = openai_image_to_anthropic(p) {
        return Some(b);
    }
    chat_image_url(p)
        .is_some_and(|u| u.starts_with("data:"))
        .then(|| p.clone())
}

/// The URL of a Chat Completions image part: `{image_url: {url}}`, or a bare `url`.
fn chat_image_url(p: &Value) -> Option<&str> {
    p.pointer("/image_url/url")
        .or_else(|| p.get("url"))
        .and_then(Value::as_str)
}

fn openai_image_to_anthropic(part: &Value) -> Option<Value> {
    let url = chat_image_url(part)?;
    if is_http_url(url) {
        return Some(json!({
            "type": "image",
            "source": { "type": "url", "url": url },
        }));
    }
    let (media_type, data) = parse_data_uri(url)?;
    Some(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    }))
}

/// An image the upstream downloads itself. Carried as a URL on every wire rather than dropped:
/// a silently image-less request produces a confident answer about a picture the model never saw.
fn is_http_url(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://")
}

/// Whether a translated image or file URL may go upstream at all: an http(s) URL the upstream
/// fetches, or inline data. Any other scheme (`file://`) names something on the client's machine
/// and is never forwarded in any shape, on any wire pair.
pub(crate) fn is_forwardable_url(url: &str) -> bool {
    is_http_url(url) || url.starts_with("data:")
}

pub(super) fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    if meta.is_empty() || data.is_empty() {
        return None;
    }
    Some((meta, data))
}

/// Anthropic `image` → a Chat Completions `image_url` part. A source Chat Completions cannot
/// reference (a Files API `file_id`) is forwarded as-is for the provider to reject by name. A
/// `url` source that is not http(s) is never forwarded in any shape.
pub(super) fn anthropic_image_part(b: &Value) -> Option<Value> {
    let Some(mut part) = anthropic_image_to_openai(b) else {
        let url_source = b.pointer("/source/type").and_then(Value::as_str) == Some("url");
        return (!url_source).then(|| b.clone());
    };
    if let Some(obj) = part.as_object_mut() {
        copy_cache_control(obj, b);
    }
    Some(part)
}

/// Anthropic `document` → a Chat Completions part. A base64 PDF becomes a `file` part with a data
/// URI and a plain-text document becomes text. A URL or Files-API document has no Chat Completions
/// equivalent and is forwarded as-is, so the provider rejects it by name instead of the model
/// answering without it.
pub(super) fn anthropic_document_to_openai(b: &Value) -> Value {
    let src = b.get("source");
    let kind = src.and_then(|s| s.get("type")).and_then(Value::as_str);
    let media = src
        .and_then(|s| s.get("media_type"))
        .and_then(Value::as_str);
    let data = src.and_then(|s| s.get("data")).and_then(Value::as_str);
    match (kind, media, data) {
        (Some("base64"), Some(media), Some(data)) => {
            let filename = b
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("document.pdf");
            json!({
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": format!("data:{media};base64,{data}"),
                }
            })
        }
        (Some("text"), _, Some(text)) => json!({ "type": "text", "text": text }),
        _ => b.clone(),
    }
}

fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    match src.get("type").and_then(Value::as_str) {
        Some("base64") => {}
        Some("url") => {
            let url = src
                .get("url")
                .and_then(Value::as_str)
                .filter(|u| is_http_url(u))?;
            return Some(json!({
                "type": "image_url",
                "image_url": { "url": url },
            }));
        }
        _ => return None,
    }
    let media = src.get("media_type").and_then(Value::as_str)?;
    let data = src.get("data").and_then(Value::as_str)?;
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{media};base64,{data}") },
    }))
}

/// A Chat Completions message's content as Responses input content. `cache_control` is Anthropic's
/// and goes; `thinking` / `redacted_thinking` blocks carry another vendor's reasoning, which a
/// Responses model cannot read (it is the earlier turn's own scratch work, not input); parts with
/// no Responses equivalent (`input_audio`) are forwarded for the provider to reject by name.
pub(super) fn chat_content_to_responses(content: Option<&Value>, assistant: bool) -> Value {
    let text_type = if assistant {
        "output_text"
    } else {
        "input_text"
    };
    let text = |t: &Value| json!({ "type": text_type, "text": t });
    match content {
        Some(Value::String(s)) => json!([text(&json!(s))]),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                    Some("text") | None => p
                        .get("text")
                        .cloned()
                        .or_else(|| p.as_str().map(|s| json!(s)))
                        .map(|t| text(&t)),
                    Some("refusal") if assistant => Some(json!({
                        "type": "refusal",
                        "refusal": p.get("refusal").cloned().unwrap_or(json!("")),
                    })),
                    Some("refusal") => p.get("refusal").map(text),
                    Some("image_url") => chat_image_to_responses(p),
                    Some("file") => {
                        let mut m = Map::new();
                        m.insert("type".into(), json!("input_file"));
                        if let Some(file) = p.get("file") {
                            for key in ["file_data", "file_id", "filename"] {
                                copy_if(&mut m, file, key);
                            }
                        }
                        Some(Value::Object(m))
                    }
                    Some("thinking" | "redacted_thinking") => None,
                    Some(_) => Some(p.clone()),
                })
                .collect(),
        ),
        _ => json!([]),
    }
}

fn chat_image_to_responses(p: &Value) -> Option<Value> {
    let Some(url) = p.pointer("/image_url/url").and_then(Value::as_str) else {
        return Some(p.clone());
    };
    // Unreachable from the proxy: `route::refused_input` refused the request first.
    if !is_forwardable_url(url) {
        return None;
    }
    let mut m = json!({ "type": "input_image", "image_url": url });
    if let Some(detail) = p.pointer("/image_url/detail")
        && let Some(obj) = m.as_object_mut()
    {
        obj.insert("detail".into(), detail.clone());
    }
    Some(m)
}

pub(super) fn responses_content_to_openai(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let mapped: Vec<Value> = parts.iter().filter_map(responses_part_to_openai).collect();
            if mapped.len() == 1
                && mapped[0].get("type").and_then(Value::as_str) == Some("text")
                && mapped[0].get("cache_control").is_none()
            {
                mapped[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::Array(mapped))
            } else {
                Value::Array(mapped)
            }
        }
        Some(other) => other.clone(),
        None => Value::String(String::new()),
    }
}

pub(super) fn responses_part_to_openai(part: &Value) -> Option<Value> {
    let typ = part.get("type").and_then(Value::as_str).unwrap_or("text");
    match typ {
        "input_text" | "output_text" | "text" => {
            let mut m = Map::new();
            m.insert("type".into(), json!("text"));
            m.insert(
                "text".into(),
                part.get("text").cloned().unwrap_or(json!("")),
            );
            copy_cache_control(&mut m, part);
            Some(Value::Object(m))
        }
        "input_image" | "image_url" => {
            let url = part
                .pointer("/image_url/url")
                .or_else(|| part.get("image_url"))
                .or_else(|| part.get("url"));
            let url = match url {
                Some(Value::String(s)) => s.as_str(),
                // A Files API image (`file_id`): Chat Completions images are URLs only. Forwarded
                // for the provider to reject by name, not answered without the picture.
                _ => return Some(part.clone()),
            };
            // Unreachable from the proxy: `route::refused_input` refused the request first.
            if !is_forwardable_url(url) {
                return None;
            }
            let mut image_url = json!({ "url": url });
            if let (Some(d), Some(obj)) = (part.get("detail"), image_url.as_object_mut()) {
                obj.insert("detail".into(), d.clone());
            }
            let mut m = json!({ "type": "image_url", "image_url": image_url });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        // A `file_url` document has no Chat Completions `file` field: forwarded as-is, unless it
        // names a local file.
        "input_file" if part.get("file_url").is_some() => part
            .get("file_url")
            .and_then(Value::as_str)
            .is_none_or(is_forwardable_url)
            .then(|| part.clone()),
        "input_file" => {
            let mut file = Map::new();
            for key in ["file_data", "file_id", "filename"] {
                copy_if(&mut file, part, key);
            }
            Some(json!({ "type": "file", "file": file }))
        }
        "thinking" | "redacted_thinking" => Some(part.clone()),
        // Chat Completions' own audio part, forwarded for the provider to reject by name rather
        // than answered without the audio.
        "input_audio" => Some(part.clone()),
        _ => None,
    }
}

pub(super) fn output_text_part(text: &str) -> Value {
    json!({ "type": "output_text", "text": text, "annotations": [] })
}

pub(super) fn refusal_part(refusal: &str) -> Value {
    json!({ "type": "refusal", "refusal": refusal })
}
