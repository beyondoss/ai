//! Second opinions: LiteLLM's `model_prices_and_context_window.json` and models.dev's
//! `api.json`. Neither is authoritative; a disagreement is a lead to check against the primary
//! source, never a correction on its own. `rates-sync audit` prints every disagreement on a
//! direct-vendor card's standard rates.

use crate::Result;
use crate::build::{Table, Tr};
use crate::dec::Rate;
use serde_json::Value;

pub const LITELLM: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
pub const MODELS_DEV: &str = "https://models.dev/api.json";

/// Where each third party files a card: (LiteLLM key candidates, models.dev provider, model).
fn keys(card: &str) -> Option<(Vec<String>, &'static str, String)> {
    let (vendor, model) = card.split_once(':')?;
    let m = model.to_owned();
    Some(match vendor {
        "anthropic" => (vec![m.clone(), format!("anthropic/{m}")], "anthropic", m),
        "openai" => (vec![m.clone(), format!("openai/{m}")], "openai", m),
        "xai" => (vec![format!("xai/{m}"), m.clone()], "xai", m),
        "deepseek" => (vec![format!("deepseek/{m}"), m.clone()], "deepseek", m),
        "groq" => (vec![format!("groq/{m}")], "groq", m),
        "together" => (vec![format!("together_ai/{m}")], "togetherai", m),
        "fireworks" => (vec![format!("fireworks_ai/{m}")], "fireworks-ai", m),
        "bedrock" => (
            vec![format!("us.anthropic.{m}")],
            "amazon-bedrock",
            format!("us.anthropic.{m}"),
        ),
        _ => return None,
    })
}

fn per_token(v: Option<&Value>) -> Option<Rate> {
    let x = v?.as_f64()?;
    // Per-token floats: to per million, at six places.
    crate::dec::float_per_million(x * 1e6).ok()
}

fn per_million(v: Option<&Value>) -> Option<Rate> {
    crate::dec::float_per_million(v?.as_f64()?).ok()
}

fn show(r: Option<Rate>) -> String {
    r.map_or("-".into(), |r| {
        r.table_text().unwrap_or_else(|_| r.to_string())
    })
}

/// Compare `ours` with a third party's (input, output, cache read, cache write).
fn cmp(
    out: &mut Vec<String>,
    who: &str,
    card: &str,
    key: &str,
    ours: &Tr,
    theirs: [Option<Rate>; 4],
) {
    let mine = [
        Some(ours.input),
        Some(ours.output),
        Some(ours.cache_read),
        Some(ours.cache_write_5m),
    ];
    let names = ["input", "output", "cache_read", "cache_write"];
    for i in 0..4 {
        // An absent cache price is not a disagreement: we hold the input rate there by rule.
        if let Some(t) = theirs[i]
            && Some(t) != mine[i]
        {
            out.push(format!(
                "{who}\t{card}\t{key}\t{}\tours {}\ttheirs {}",
                names[i],
                show(mine[i]),
                show(Some(t))
            ));
        }
    }
}

/// Every disagreement between the table's direct-vendor cards and the two third parties.
pub fn run(table: &Table, litellm: &Value, models_dev: &Value) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for (card, c) in &table.cards {
        let Some((lkeys, mdp, mdm)) = keys(card) else {
            continue;
        };
        match lkeys.iter().find_map(|k| litellm.get(k).map(|v| (k, v))) {
            Some((k, v)) => cmp(
                &mut out,
                "litellm",
                card,
                k,
                &c.standard,
                [
                    per_token(v.get("input_cost_per_token")),
                    per_token(v.get("output_cost_per_token")),
                    per_token(v.get("cache_read_input_token_cost")),
                    per_token(v.get("cache_creation_input_token_cost")),
                ],
            ),
            None => out.push(format!("litellm\t{card}\t-\tnot listed")),
        }
        match models_dev.pointer(&format!("/{mdp}/models/{}", mdm.replace('/', "~1"))) {
            Some(v) => cmp(
                &mut out,
                "models.dev",
                card,
                &format!("{mdp}/{mdm}"),
                &c.standard,
                [
                    per_million(v.pointer("/cost/input")),
                    per_million(v.pointer("/cost/output")),
                    per_million(v.pointer("/cost/cache_read")),
                    per_million(v.pointer("/cost/cache_write")),
                ],
            ),
            None => out.push(format!("models.dev\t{card}\t-\tnot listed")),
        }
    }
    Ok(out)
}
