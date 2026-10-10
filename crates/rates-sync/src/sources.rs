//! Every primary source the table is generated from: where it is fetched, which file under
//! `verify/rates_sources/` holds its snapshot, and how the raw response is normalized into that
//! snapshot.
//!
//! Normalization keeps what prices a request and drops what churns between two fetches of the
//! same prices (OpenRouter's uptime and latency figures, xAI's build fingerprints, an HTML page's
//! navigation and asset hashes), and orders what has no meaningful order. Two fetches of
//! unchanged prices are byte-identical snapshots.

use crate::Result;
use crate::json::{canonical, str_at};
use serde_json::{Map, Value, json};

/// How a source authenticates. Only keys the repo's `.env` already holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Auth {
    /// None: a public page or API.
    Public,
    /// `Authorization: Bearer $VAR`.
    Bearer(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A markdown page, kept as served (line endings normalized).
    Markdown,
    /// An HTML page: only the `<article>` element.
    HtmlArticle,
    /// An HTML page: only the element with `id="UCAP-CONTENT"` (gov.cn's legal text).
    HtmlGovCn,
    /// An HTML page: only its JSON-LD blocks.
    HtmlJsonLd,
    /// An HTML page as plain text: scripts, styles and tags removed, one line per block.
    HtmlText,
    /// `GET https://api.x.ai/v1/language-models`.
    XaiLanguageModels,
    /// `GET https://api.together.xyz/v1/models`.
    TogetherModels,
    /// The AWS Price List bulk offer for `AmazonBedrockFoundationModels`: its Claude products.
    AwsBedrock,
    /// `GET https://openrouter.ai/api/v1/models`: the catalog's slugs.
    OpenRouterModels,
    /// `GET https://openrouter.ai/api/v1/models/{slug}/endpoints`.
    OpenRouterEndpoints,
}

#[derive(Clone, Debug)]
pub struct SourceDef {
    /// Stable id (`anthropic.pricing`, `openrouter.endpoints/z-ai/glm-5.3`).
    pub id: String,
    pub url: String,
    /// Path under `verify/rates_sources/`.
    pub file: String,
    pub auth: Auth,
    pub shape: Shape,
}

fn def(id: &str, url: &str, file: &str, auth: Auth, shape: Shape) -> SourceDef {
    SourceDef {
        id: id.to_owned(),
        url: url.to_owned(),
        file: file.to_owned(),
        auth,
        shape,
    }
}

pub const ANTHROPIC_PRICING: &str = "https://platform.claude.com/docs/en/about-claude/pricing.md";
pub const OPENAI_PRICING: &str = "https://developers.openai.com/api/docs/pricing.md";
pub const XAI_PRICING: &str = "https://docs.x.ai/developers/pricing.md";
pub const XAI_MODELS: &str = "https://api.x.ai/v1/language-models";
pub const DEEPSEEK_PRICING: &str = "https://api-docs.deepseek.com/quick_start/pricing";
pub const GOV_CN_HOLIDAYS: &str = "https://www.gov.cn/zhengce/content/202411/content_6986380.htm";
pub const GROQ_MODELS: &str = "https://console.groq.com/docs/models.md";
pub const GROQ_CACHING: &str = "https://console.groq.com/docs/prompt-caching.md";
pub const GROQ_FLEX: &str = "https://console.groq.com/docs/flex-processing.md";
pub const FIREWORKS_PRICING: &str = "https://docs.fireworks.ai/serverless/pricing.md";
pub const TOGETHER_MODELS: &str = "https://api.together.xyz/v1/models";
pub const TOGETHER_DOCS: &str = "https://docs.together.ai/docs/serverless/models.md";
pub const AWS_BEDROCK: &str = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json";
pub const OPENROUTER_MODELS: &str = "https://openrouter.ai/api/v1/models?output_modalities=all";
pub const OPENROUTER_PRICING: &str = "https://openrouter.ai/pricing";
pub const TOGETHER_PRICING: &str = "https://www.together.ai/pricing";

/// Every source, the OpenRouter endpoint lists last (one per slug the catalog prices through
/// OpenRouter, sorted).
pub fn all(or_slugs: &[String]) -> Vec<SourceDef> {
    use Auth::*;
    use Shape::*;
    let mut v = vec![
        def(
            "anthropic.pricing",
            ANTHROPIC_PRICING,
            "anthropic/pricing.md",
            Public,
            Markdown,
        ),
        def(
            "openai.pricing",
            OPENAI_PRICING,
            "openai/pricing.md",
            Public,
            Markdown,
        ),
        def(
            "xai.pricing",
            XAI_PRICING,
            "xai/pricing.md",
            Public,
            Markdown,
        ),
        def(
            "xai.language_models",
            XAI_MODELS,
            "xai/language-models.json",
            Bearer("XAI_API_KEY"),
            XaiLanguageModels,
        ),
        def(
            "deepseek.pricing",
            DEEPSEEK_PRICING,
            "deepseek/pricing.html",
            Public,
            HtmlArticle,
        ),
        def(
            "gov_cn.holidays",
            GOV_CN_HOLIDAYS,
            "deepseek/gov-cn-holidays.html",
            Public,
            HtmlGovCn,
        ),
        def(
            "groq.models",
            GROQ_MODELS,
            "groq/models.md",
            Public,
            Markdown,
        ),
        def(
            "groq.prompt_caching",
            GROQ_CACHING,
            "groq/prompt-caching.md",
            Public,
            Markdown,
        ),
        def(
            "groq.flex",
            GROQ_FLEX,
            "groq/flex-processing.md",
            Public,
            Markdown,
        ),
        def(
            "fireworks.pricing",
            FIREWORKS_PRICING,
            "fireworks/pricing.md",
            Public,
            Markdown,
        ),
        def(
            "together.models",
            TOGETHER_MODELS,
            "together/models.json",
            Bearer("TOGETHER_API_KEY"),
            TogetherModels,
        ),
        def(
            "together.docs",
            TOGETHER_DOCS,
            "together/serverless-models.md",
            Public,
            Markdown,
        ),
        def(
            "together.pricing",
            TOGETHER_PRICING,
            "together/pricing.txt",
            Public,
            HtmlText,
        ),
        def(
            "aws.bedrock",
            AWS_BEDROCK,
            "bedrock/foundation-models-us-east-1.json",
            Public,
            AwsBedrock,
        ),
        def(
            "openrouter.models",
            OPENROUTER_MODELS,
            "openrouter/models.json",
            Public,
            OpenRouterModels,
        ),
        def(
            "openrouter.pricing",
            OPENROUTER_PRICING,
            "openrouter/pricing.json",
            Public,
            HtmlJsonLd,
        ),
    ];
    let mut slugs = or_slugs.to_vec();
    slugs.sort();
    slugs.dedup();
    for slug in slugs {
        v.push(SourceDef {
            id: format!("openrouter.endpoints/{slug}"),
            url: format!("https://openrouter.ai/api/v1/models/{slug}/endpoints"),
            file: format!("openrouter/endpoints/{slug}.json"),
            auth: Public,
            shape: OpenRouterEndpoints,
        });
    }
    v
}

/// The snapshot text for a raw response.
pub fn normalize(def: &SourceDef, raw: &[u8], or_slugs: &[String]) -> Result<String> {
    let text = std::str::from_utf8(raw).map_err(|e| format!("{}: not UTF-8: {e}", def.id))?;
    let text = text.replace("\r\n", "\n");
    let err = |e: String| format!("{}: {e}", def.id);
    match def.shape {
        Shape::Markdown => {
            if text.trim_start().starts_with('<') {
                return Err(err("expected markdown, got HTML".into()));
            }
            let mut t = text.trim_end().to_owned();
            t.push('\n');
            Ok(t)
        }
        Shape::HtmlArticle => element(&text, "<article", "article")
            .map_err(err)
            .map(|s| s + "\n"),
        Shape::HtmlGovCn => element(&text, "id=\"UCAP-CONTENT\"", "div")
            .map_err(err)
            .map(|s| s + "\n"),
        Shape::HtmlJsonLd => json_ld(&text).map_err(err),
        Shape::HtmlText => Ok(page_text(&text)),
        Shape::XaiLanguageModels => xai(&text).map_err(err),
        Shape::TogetherModels => together(&text).map_err(err),
        Shape::AwsBedrock => aws(&text).map_err(err),
        Shape::OpenRouterModels => or_models(&text, or_slugs).map_err(err),
        Shape::OpenRouterEndpoints => or_endpoints(&text).map_err(err),
    }
}

fn parse(text: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))
}

/// The one element whose start tag contains `marker`, through its matching close tag.
pub fn element(html: &str, marker: &str, tag: &str) -> Result<String> {
    let hits: Vec<usize> = html.match_indices(marker).map(|(i, _)| i).collect();
    let [at] = hits[..] else {
        return Err(format!(
            "expected exactly one `{marker}`, found {}",
            hits.len()
        ));
    };
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = html[..at + marker.len()]
        .rfind(&open)
        .ok_or_else(|| format!("`{marker}` is not inside a <{tag}>"))?;
    let mut depth = 0usize;
    let mut i = start;
    while i < html.len() {
        let rest = &html[i..];
        if rest.starts_with(&close) {
            depth -= 1;
            if depth == 0 {
                return Ok(html[start..i + close.len()].to_owned());
            }
            i += close.len();
        } else if rest.starts_with(&open)
            && rest[open.len()..].starts_with(|c: char| c == '>' || c.is_ascii_whitespace())
        {
            depth += 1;
            i += open.len();
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    Err(format!("<{tag}> holding `{marker}` is never closed"))
}

/// The visible text of a page: `<script>`, `<style>`, `<svg>` and `<noscript>` bodies dropped,
/// tags removed, one line per block element, whitespace collapsed within a line.
fn page_text(html: &str) -> String {
    let mut s = html.to_owned();
    for tag in ["script", "style", "svg", "noscript", "head"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut out = String::with_capacity(s.len());
        let mut rest = s.as_str();
        while let Some(i) = rest.find(&open) {
            out.push_str(&rest[..i]);
            match rest[i..].find(&close) {
                Some(j) => rest = &rest[i + j + close.len()..],
                None => rest = "",
            }
        }
        out.push_str(rest);
        s = out;
    }
    for tag in [
        "</p>", "</div>", "</li>", "</h1>", "</h2>", "</h3>", "</h4>", "</tr>", "<br>", "<br/>",
    ] {
        s = s.replace(tag, &format!("{tag}\n"));
    }
    let mut lines: Vec<String> = Vec::new();
    for line in s.split('\n') {
        let t = crate::md::html_text(line);
        if !t.is_empty() {
            lines.push(t);
        }
    }
    lines.join("\n") + "\n"
}

fn json_ld(html: &str) -> Result<String> {
    let open = "<script type=\"application/ld+json\">";
    let mut blocks = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find(open) {
        let body = &rest[i + open.len()..];
        let end = body.find("</script>").ok_or("an unclosed JSON-LD block")?;
        blocks.push(parse(&body[..end])?);
        rest = &body[end..];
    }
    if blocks.is_empty() {
        return Err("no JSON-LD blocks".into());
    }
    Ok(canonical(&Value::Array(blocks)))
}

fn xai(text: &str) -> Result<String> {
    let v = parse(text)?;
    let mut models: Vec<Value> = v
        .get("models")
        .and_then(Value::as_array)
        .ok_or("no `models` array")?
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Some(o) = m.as_object_mut() {
                // A build hash: changes on a redeploy at the same prices.
                o.remove("fingerprint");
            }
            m
        })
        .collect();
    models.sort_by(|a, b| {
        a.get("id")
            .map(Value::to_string)
            .cmp(&b.get("id").map(Value::to_string))
    });
    Ok(canonical(&json!({ "models": models })))
}

fn together(text: &str) -> Result<String> {
    let v = parse(text)?;
    let all = v.as_array().ok_or("expected an array of models")?;
    let mut out = Vec::new();
    for m in all {
        let id = str_at(m, "id", "a Together model")?;
        let mut keep = Map::new();
        for k in [
            "id",
            "type",
            "display_name",
            "organization",
            "context_length",
            "pricing",
        ] {
            if let Some(x) = m.get(k) {
                keep.insert(k.to_owned(), x.clone());
            }
        }
        out.push((id.to_owned(), Value::Object(keep)));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(canonical(&Value::Array(
        out.into_iter().map(|(_, v)| v).collect(),
    )))
}

/// The Claude products and their on-demand terms. The offer holds every Bedrock model; only the
/// Claude models are priced from it (every Bedrock candidate is a Claude `us.` profile).
fn aws(text: &str) -> Result<String> {
    let v = parse(text)?;
    let products = v
        .get("products")
        .and_then(Value::as_object)
        .ok_or("no `products`")?;
    let terms = v
        .pointer("/terms/OnDemand")
        .and_then(Value::as_object)
        .ok_or("no `terms.OnDemand`")?;
    let mut keep_p = Map::new();
    let mut keep_t = Map::new();
    for (sku, p) in products {
        let model = p
            .pointer("/attributes/model")
            .and_then(Value::as_str)
            .unwrap_or("");
        if model.starts_with("Claude ") {
            keep_p.insert(sku.clone(), p.clone());
            if let Some(t) = terms.get(sku) {
                keep_t.insert(sku.clone(), t.clone());
            }
        }
    }
    if keep_p.is_empty() {
        return Err("no Claude products in the offer".into());
    }
    let mut out = Map::new();
    for k in ["offerCode", "version", "publicationDate"] {
        if let Some(x) = v.get(k) {
            out.insert(k.to_owned(), x.clone());
        }
    }
    out.insert("products".into(), Value::Object(keep_p));
    out.insert("terms".into(), json!({ "OnDemand": keep_t }));
    Ok(canonical(&Value::Object(out)))
}

fn or_models(text: &str, slugs: &[String]) -> Result<String> {
    let v = parse(text)?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or("no `data` array")?;
    let mut out = Vec::new();
    for m in data {
        let id = str_at(m, "id", "an OpenRouter model")?;
        if slugs.iter().any(|s| s == id) {
            let mut keep = Map::new();
            for k in ["id", "name", "pricing", "context_length"] {
                if let Some(x) = m.get(k) {
                    keep.insert(k.to_owned(), x.clone());
                }
            }
            out.push((id.to_owned(), Value::Object(keep)));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(canonical(
        &json!({ "data": out.into_iter().map(|(_, v)| v).collect::<Vec<_>>() }),
    ))
}

/// The endpoint fields that say who serves and what it costs; uptime, latency, throughput and
/// status change minute to minute at the same prices.
fn or_endpoints(text: &str) -> Result<String> {
    let v = parse(text)?;
    let data = v.get("data").ok_or("no `data`")?;
    let id = str_at(data, "id", "the endpoint list")?;
    let eps = data
        .get("endpoints")
        .and_then(Value::as_array)
        .ok_or("no `data.endpoints`")?;
    let mut out = Vec::new();
    for e in eps {
        let mut keep = Map::new();
        for k in [
            "provider_name",
            "tag",
            "pricing",
            "quantization",
            "context_length",
            "max_prompt_tokens",
        ] {
            if let Some(x) = e.get(k) {
                keep.insert(k.to_owned(), x.clone());
            }
        }
        let key = format!(
            "{}\u{0}{}\u{0}{}",
            e.get("tag").map(Value::to_string).unwrap_or_default(),
            e.get("provider_name")
                .map(Value::to_string)
                .unwrap_or_default(),
            e.get("pricing").map(Value::to_string).unwrap_or_default()
        );
        out.push((key, Value::Object(keep)));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(canonical(&json!({
        "data": { "id": id, "endpoints": out.into_iter().map(|(_, v)| v).collect::<Vec<_>>() }
    })))
}
