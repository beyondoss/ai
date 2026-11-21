//! What the upstream model id says about what it accepts.

/// Which reasoning and sampling controls a Claude model accepts. Decided from the id the candidate
/// will receive, because the same OpenAI `reasoning_effort` must become different requests: the
/// old `thinking.budget_tokens` shape is a 400 on every current model, and adaptive thinking is a
/// 400 before 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaudeGen {
    /// Claude 3.x through 4.5, and non-Claude models behind a Messages-compatible API:
    /// `thinking: {type: enabled, budget_tokens}` (less than `max_tokens`), sampling allowed
    /// without thinking.
    Budget,
    /// Claude 4.6: adaptive thinking with `output_config.effort` `low`..`max` (no `xhigh`),
    /// sampling allowed without thinking.
    Adaptive46,
    /// Claude 4.7 and later, Fable, Mythos: adaptive thinking and effort (`xhigh` included).
    /// `budget_tokens` and non-default `temperature` / `top_p` are 400s, and several reject
    /// `thinking: disabled`, so "no reasoning" is an omitted `thinking` at effort `low`.
    Adaptive,
}

/// What the Claude model behind a candidate accepts, parsed from the id it will receive (any
/// spelling: `claude-opus-4-8`, OpenRouter's `anthropic/claude-opus-4.8`, Bedrock's
/// `global.anthropic.claude-haiku-4-5-…`). A dated suffix (`claude-sonnet-4-20250514`) is not a
/// minor version. An unrecognized Claude family is assumed to behave like the newest models; a
/// model that is not Claude at all keeps the long-standing shapes Messages-compatible APIs
/// implement (`budget_tokens`, forced `tool_choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaudeModel {
    pub(crate) reasoning: ClaudeGen,
    /// Whether forced `tool_choice` (`any` / `tool`) is accepted. Claude Fable 5.1, Mythos 5.1,
    /// Opus 5.5 and Sonnet 5.5 reject it with a 400. Haiku 5.5 accepts it ("Claude Haiku 5.5
    /// accepts a forced `tool_choice` (`any` or a named tool)", its migration guide; measured
    /// 2026-10-11).
    pub(crate) forced_tool_choice: bool,
    /// Whether `{"role": "system"}` may appear inside `messages`: Opus 4.8 and later, Fable,
    /// Mythos, Sonnet and Haiku 5.5 and later. Not Sonnet 5 or older (the docs list it as
    /// unsupported), and never a model that is not Claude.
    pub(crate) mid_system: bool,
    /// Whether the request should carry `thinking.block_binding` (see
    /// `bind_thinking_with_drop_block`): a model that binds thinking to its conversation (Fable
    /// 5.1, Opus, Sonnet and Haiku 5.5 and later; not Mythos 5.1, which skips that check), spelled as
    /// Anthropic's own API spells it. Bedrock (`…anthropic.claude-…`) and OpenRouter
    /// (`anthropic/…`) spellings are excluded: whether those hosts accept the beta header that the
    /// field requires is unverified, and a field without its header is a 400.
    pub(crate) binding_controls: bool,
    /// Whether `thinking: {type: between_tools}` exists: Sonnet 5.5 and later Sonnets only.
    pub(crate) between_tools: bool,
}

impl ClaudeModel {
    pub(crate) fn of(model: &str) -> Self {
        const NOT_CLAUDE: ClaudeModel = ClaudeModel {
            reasoning: ClaudeGen::Budget,
            forced_tool_choice: true,
            mid_system: false,
            binding_controls: false,
            between_tools: false,
        };
        let Some(at) = model.find("claude-") else {
            return NOT_CLAUDE;
        };
        let first_party = at == 0;
        let newest = ClaudeModel {
            reasoning: ClaudeGen::Adaptive,
            forced_tool_choice: false,
            mid_system: true,
            binding_controls: first_party,
            between_tools: false,
        };
        let mut parts = model[at + "claude-".len()..].split(['-', '.']);
        let family = parts.next().unwrap_or("");
        if family.starts_with(|c: char| c.is_ascii_digit()) {
            // `claude-3-haiku`, `claude-3-5-sonnet-…`.
            return NOT_CLAUDE;
        }
        let Some(major) = parts.next().and_then(|p| p.parse::<u32>().ok()) else {
            return newest;
        };
        let minor = parts
            .next()
            .filter(|p| p.len() <= 2)
            .and_then(|p| p.parse::<u32>().ok())
            .unwrap_or(0);
        let version = (major, minor);
        match family {
            "opus" | "sonnet" | "haiku" => ClaudeModel {
                reasoning: match version {
                    (5.., _) | (4, 7..) => ClaudeGen::Adaptive,
                    (4, 6) => ClaudeGen::Adaptive46,
                    _ => ClaudeGen::Budget,
                },
                forced_tool_choice: family == "haiku" || version < (5, 5),
                mid_system: match family {
                    "opus" => version >= (4, 8),
                    _ => version >= (5, 5),
                },
                binding_controls: first_party && version >= (5, 5),
                between_tools: family == "sonnet" && version >= (5, 5),
            },
            "fable" | "mythos" => ClaudeModel {
                reasoning: ClaudeGen::Adaptive,
                forced_tool_choice: version < (5, 1),
                mid_system: true,
                binding_controls: first_party && family == "fable" && version >= (5, 1),
                between_tools: false,
            },
            _ => newest,
        }
    }
}

/// Which reasoning and sampling controls an OpenAI model accepts, parsed from the id this
/// attempt's candidate receives. Measured live on 2026-09-30 against Chat Completions and
/// Responses; where the two differ (o-series `xhigh`, GPT-6 `max`), the set is what both accept.
///
/// Applied only to OpenAI's own ids. Other hosts (xAI, DeepSeek, Groq's `openai/gpt-oss-…`) keep
/// every field as the client sent it. OpenRouter's `openai/…` normalizes `max_tokens` and drops
/// sampling on its own, but passes the effort through (`none` on GPT-5 is "Reasoning is mandatory",
/// `xhigh` on o4-mini a 400), so only the effort is clamped there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAiModel {
    /// OpenAI's own API (bare id): `max_tokens` is rejected on reasoning models, and non-default
    /// sampling is rejected unless reasoning is off.
    pub(crate) native: bool,
    pub(crate) reasoning: OpenAiReasoning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAiReasoning {
    /// Not an OpenAI model this table knows: effort and sampling pass through.
    Unknown,
    /// No reasoning (`gpt-4`, `gpt-4o`, `gpt-4.1`): `reasoning_effort` is an unrecognized argument.
    Off,
    /// A reasoning model and the `reasoning_effort` values it accepts, lowest first.
    Efforts(&'static [&'static str]),
}

/// Effort names, lowest first. Anthropic's `max` is the top rung.
const EFFORT_LADDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

impl OpenAiModel {
    const UNKNOWN: Self = Self {
        native: false,
        reasoning: OpenAiReasoning::Unknown,
    };

    pub(crate) fn of(model: &str) -> Self {
        // OpenAI's open-weight gpt-oss, on whichever host serves it (Groq, Together, OpenRouter's
        // `openai/gpt-oss-…`, Fireworks' `accounts/fireworks/models/gpt-oss-…`): reasoning is
        // mandatory and the efforts are `low`–`high` (`none` is "Reasoning is mandatory for this
        // endpoint", measured on OpenRouter). Never OpenAI's own API, so every other field stays.
        if model
            .rsplit('/')
            .next()
            .is_some_and(|id| id.starts_with("gpt-oss"))
        {
            return Self {
                native: false,
                reasoning: OpenAiReasoning::Efforts(&["low", "medium", "high"]),
            };
        }
        let (native, id) = match model.split_once('/') {
            None => (true, model),
            Some(("openai", id)) => (false, id),
            Some(_) => return Self::UNKNOWN,
        };
        let is_openai = id.starts_with("chatgpt-")
            || (id.starts_with("gpt-") && !id.starts_with("gpt-oss"))
            || id
                .strip_prefix('o')
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
        if !is_openai {
            return Self::UNKNOWN;
        }
        Self {
            native,
            reasoning: openai_reasoning_of(id),
        }
    }

    /// The field that caps output on Chat Completions: OpenAI's own API rejects `max_tokens` on
    /// every reasoning model and accepts `max_completion_tokens` on all of them.
    pub(super) fn max_tokens_key(self) -> &'static str {
        if self.native {
            "max_completion_tokens"
        } else {
            "max_tokens"
        }
    }

    /// The `reasoning_effort` to send for a requested effort, or `None` to send none. A value this
    /// model lacks becomes the nearest one above it (`none` on a model that always reasons becomes
    /// its lowest), or its highest when nothing is above. An unrecognized name passes through for
    /// the provider to reject by name.
    ///
    /// A model this table does not know gets the classic `low` / `medium` / `high` in place of the
    /// rungs only OpenAI's newer families define: `xhigh` / `max` (what a large Anthropic
    /// `budget_tokens` or `effort: max` maps to) become `high`, `minimal` becomes `low`. Hosts that
    /// take `reasoning_effort` at all take those three; Groq documents no other.
    pub(super) fn effort(self, requested: &str) -> Option<String> {
        let requested = match requested {
            "off" | "disabled" => "none",
            other => other,
        };
        match self.reasoning {
            OpenAiReasoning::Unknown => Some(
                match requested {
                    "xhigh" | "max" => "high",
                    "minimal" => "low",
                    other => other,
                }
                .to_owned(),
            ),
            OpenAiReasoning::Off => None,
            OpenAiReasoning::Efforts(accepted) => {
                let rank = |e: &str| EFFORT_LADDER.iter().position(|x| *x == e);
                let Some(want) = rank(requested) else {
                    return Some(requested.to_owned());
                };
                accepted
                    .iter()
                    .find(|a| rank(a).is_some_and(|r| r >= want))
                    .or_else(|| accepted.last())
                    .map(|e| (*e).to_owned())
            }
        }
    }

    /// Whether `temperature` / `top_p` may go upstream given the effort being sent. OpenAI's
    /// reasoning models accept only the default unless reasoning is `none`; OpenRouter drops them
    /// itself, and a model this table does not know keeps them.
    pub(super) fn keeps_sampling(self, effort: Option<&str>) -> bool {
        !self.native
            || !matches!(self.reasoning, OpenAiReasoning::Efforts(_))
            || effort == Some("none")
    }

    /// The effort `v` asks for (`reasoning_effort`, or Responses' `reasoning.effort`) fitted to
    /// this model, and `temperature` / `top_p` copied into `out` when [`Self::keeps_sampling`]
    /// allows. The same in both directions between Chat Completions and Responses; the caller
    /// writes the returned effort in its own shape.
    pub(super) fn fit_effort_and_sampling(
        self,
        out: &mut serde_json::Map<String, serde_json::Value>,
        v: &serde_json::Value,
    ) -> Option<String> {
        let effort = v
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                v.pointer("/reasoning/effort")
                    .and_then(serde_json::Value::as_str)
            })
            .and_then(|e| self.effort(e));
        if self.keeps_sampling(effort.as_deref()) {
            super::copy_if(out, v, "temperature");
            super::copy_if(out, v, "top_p");
        }
        effort
    }
}

/// Accepted efforts by family, measured live (2026-09-30). Pro variants take fewer.
fn openai_reasoning_of(id: &str) -> OpenAiReasoning {
    const O_SERIES: &[&str] = &["low", "medium", "high"];
    const GPT5: &[&str] = &["minimal", "low", "medium", "high"];
    const GPT5_PRO: &[&str] = &["high"];
    const GPT51: &[&str] = &["none", "low", "medium", "high"];
    const GPT52: &[&str] = &["none", "low", "medium", "high", "xhigh"];
    const GPT6: &[&str] = &["low", "medium", "high", "xhigh"];
    const PRO: &[&str] = &["medium", "high", "xhigh"];
    if id.starts_with('o') {
        return OpenAiReasoning::Efforts(O_SERIES);
    }
    let Some(rest) = id.strip_prefix("gpt-") else {
        // `chatgpt-4o-latest`.
        return OpenAiReasoning::Off;
    };
    fn digits(s: &str) -> (Option<u32>, &str) {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        (s[..n].parse::<u32>().ok(), &s[n..])
    }
    let (Some(major), after) = digits(rest) else {
        return OpenAiReasoning::Unknown;
    };
    let (minor, suffix) = match after.strip_prefix('.') {
        Some(m) => {
            let (minor, suffix) = digits(m);
            (minor.unwrap_or(0), suffix)
        }
        None => (0, after),
    };
    let pro = suffix.split('-').any(|s| s == "pro");
    if major <= 4 {
        return OpenAiReasoning::Off;
    }
    if suffix.split('-').any(|s| s == "chat") {
        // ChatGPT's instant snapshots (`gpt-5.2-chat`): not verified; leave them alone.
        return OpenAiReasoning::Unknown;
    }
    OpenAiReasoning::Efforts(match (major, minor, pro) {
        (5, 0, true) => GPT5_PRO,
        (5, 0, false) => GPT5,
        (5, 1, _) => GPT51,
        (5, _, true) | (6.., _, true) => PRO,
        (5, _, false) => GPT52,
        _ => GPT6,
    })
}
