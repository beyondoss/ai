//! The billing facts beyond token counts: server-side tool use by kind, the served speed and
//! inference geography, and what the upstream reported about the call itself (its ids and its own
//! price). Read from the same usage blocks as the token counts, leniently: a malformed value costs
//! that value, never the usage block.

use super::{ServiceTier, Usage};
use arrayvec::ArrayString;
use serde::Deserialize;

/// A provider id echoed in a response (a generation, message, response or container id).
/// Inline so [`Usage`] stays `Copy`; longer than this, or outside [`id_byte`], is dropped.
pub type IdStr = ArrayString<128>;

/// A serving host's display name (OpenRouter's `provider`, e.g. `Amazon Bedrock`). Longer than
/// this, or outside [`host_byte`], is dropped.
pub type HostStr = ArrayString<64>;

/// Server-side tool use by kind: the per-call (or per-item) fees a pricer adds to the tokens.
/// Each count is what the provider reported, or, where it reports none (OpenAI Responses), the
/// hosted-tool items in the output. `0` is both "none" and "not reported".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerTools {
    /// Billable web searches: Anthropic `server_tool_use.web_search_requests`, OpenAI/xAI
    /// `web_search_call` items (action `search`, or no action seen), xAI `web_search_calls`,
    /// OpenRouter `server_tool_use(_details).web_search_requests`.
    pub web_search: u32,
    /// OpenAI `web_search_call` items whose action is `open_page` / `find_in_page` (no fee).
    pub web_search_page: u32,
    /// Anthropic `server_tool_use.web_fetch_requests` (no per-call fee; tokens only).
    pub web_fetch: u32,
    /// Anthropic `server_tool_use.code_execution_requests`, OpenAI `code_interpreter_call`
    /// items, xAI `code_interpreter_calls`. Calls, not container sessions.
    pub code_execution: u32,
    /// OpenAI `file_search_call` items, xAI `file_search_calls`.
    pub file_search: u32,
    /// OpenAI `image_generation_call` items, xAI `image_generation_calls`.
    pub image_generation: u32,
    /// OpenAI `computer_call` items (billed as tokens; counted for completeness).
    pub computer_use: u32,
    /// OpenAI `mcp_call` items, xAI `mcp_calls`.
    pub mcp: u32,
    /// OpenAI hosted `shell_call` / `local_shell_call` items.
    pub shell: u32,
    /// OpenAI `tool_search_call` items.
    pub tool_search: u32,
    /// xAI `x_search_calls`.
    pub x_search: u32,
    /// xAI `x_posts_fetched` (X search bills per post).
    pub x_posts: u32,
    /// xAI `x_users_fetched` (X search bills per profile).
    pub x_users: u32,
    /// xAI `document_search_calls`.
    pub document_search: u32,
    /// xAI `num_sources_used`.
    pub sources: u32,
    /// OpenRouter `server_tool_use(_details).tool_calls_executed`: its server tools of every kind.
    pub tool_calls: u32,
}

impl ServerTools {
    /// `(row key, count)` for every kind, in the row's fixed order.
    pub fn entries(&self) -> [(&'static str, u32); 16] {
        [
            ("web_search", self.web_search),
            ("web_search_page", self.web_search_page),
            ("web_fetch", self.web_fetch),
            ("code_execution", self.code_execution),
            ("file_search", self.file_search),
            ("image_generation", self.image_generation),
            ("computer_use", self.computer_use),
            ("mcp", self.mcp),
            ("shell", self.shell),
            ("tool_search", self.tool_search),
            ("x_search", self.x_search),
            ("x_posts", self.x_posts),
            ("x_users", self.x_users),
            ("document_search", self.document_search),
            ("sources", self.sources),
            ("tool_calls", self.tool_calls),
        ]
    }

    /// Whether any kind has a nonzero count.
    pub fn any(&self) -> bool {
        self.entries().iter().any(|(_, n)| *n > 0)
    }

    /// The row's `server_tools` field: `kind=count` pairs joined with `,`, nonzero kinds only, in
    /// [`Self::entries`] order. `None` when every count is zero.
    pub fn to_row(&self) -> Option<String> {
        use std::fmt::Write;
        let mut out = String::new();
        for (k, n) in self.entries() {
            if n > 0 {
                if !out.is_empty() {
                    out.push(',');
                }
                let _ = write!(out, "{k}={n}");
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// Per kind, the larger of the two: a provider that reports a count in `usage` and lists the
    /// items too (xAI Responses) is counted once.
    pub fn merge_max(&mut self, o: &ServerTools) {
        macro_rules! m {
            ($($f:ident),*) => { $( self.$f = self.$f.max(o.$f); )* };
        }
        m!(
            web_search,
            web_search_page,
            web_fetch,
            code_execution,
            file_search,
            image_generation,
            computer_use,
            mcp,
            shell,
            tool_search,
            x_search,
            x_posts,
            x_users,
            document_search,
            sources,
            tool_calls
        );
    }
}

/// What the upstream itself reported about the call, for reconciling a row against the vendor:
/// its ids and, where it reports one, its own price.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Upstream {
    /// The vendor's id for this generation: OpenRouter's `X-Generation-Id` (or body `gen-…`),
    /// Anthropic `msg_…`, OpenAI `chatcmpl-…` / `resp_…`, xAI's response id.
    pub generation_id: Option<IdStr>,
    /// The host OpenRouter routed to (a root `provider`, which its OpenAPI spec does not list),
    /// when present.
    pub served_by: Option<HostStr>,
    /// The vendor's reported cost of the call, in units of 1e-10 USD: OpenRouter `usage.cost`
    /// (credits; 1 credit = 1 USD) or xAI `usage.cost_in_usd_ticks` (already 1e-10 USD).
    pub cost_e10: Option<u64>,
    /// OpenRouter `usage.cost_details.upstream_inference_cost` (BYOK: what the upstream charged).
    pub inference_cost_e10: Option<u64>,
    /// OpenRouter `usage.cost_details.server_tool_cost` (its metered server tools).
    pub tool_cost_e10: Option<u64>,
    /// OpenRouter `usage.is_byok`.
    pub byok: Option<bool>,
    /// The code-execution container the turn used: Anthropic root `container.id` (or
    /// `message_delta.delta.container.id`), OpenAI `code_interpreter_call.container_id`.
    /// Container time bills per container, across requests, so a pricer dedupes on it.
    pub container_id: Option<IdStr>,
}

/// A vendor's dollar amount (`f64` USD) as 1e-10 USD units; `None` for a negative or non-finite
/// one.
pub(super) fn usd_to_e10(v: f64) -> Option<u64> {
    let x = (v * 1e10).round();
    // `u64::MAX as f64` rounds up to 2^64, so `<` keeps the cast in range.
    #[allow(clippy::cast_precision_loss)]
    let max = u64::MAX as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (x.is_finite() && x >= 0.0 && x < max).then_some(x as u64)
}

/// 1e-10 USD units as a decimal USD string (`12345` → `"0.0000012345"`), trailing zeros trimmed:
/// exact, so a pricer reads a decimal rather than a float.
pub fn e10_to_usd(v: u64) -> String {
    let whole = v / 10_000_000_000;
    let frac = v % 10_000_000_000;
    if frac == 0 {
        return whole.to_string();
    }
    let f = format!("{frac:010}");
    format!("{whole}.{}", f.trim_end_matches('0'))
}

/// Deserialize a string into an inline `ArrayString` when every byte passes `ok`, leniently: any
/// other JSON value, or a string empty, too long or outside the charset, reads as `None`.
fn de_token<'de, D: serde::Deserializer<'de>, const N: usize>(
    d: D,
    ok: fn(u8) -> bool,
) -> Result<Option<ArrayString<N>>, D::Error> {
    struct V<const N: usize>(fn(u8) -> bool);
    impl<'de, const N: usize> serde::de::Visitor<'de> for V<N> {
        type Value = Option<ArrayString<N>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an identifier")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            let good = !v.is_empty() && v.bytes().all(self.0);
            Ok(good.then(|| ArrayString::from(v).ok()).flatten())
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a
                .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                .is_some()
            {}
            Ok(None)
        }
    }
    d.deserialize_any(V::<N>(ok))
}

/// The bytes an echoed id may hold.
pub fn id_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')
}

/// The bytes a serving host's display name may hold.
pub fn host_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b' ' | b'_' | b'-' | b'.' | b'(' | b')' | b'/')
}

/// An id from untrusted text, checked as [`de_id`] checks one.
pub fn id_from(v: &str) -> Option<IdStr> {
    (!v.is_empty() && v.bytes().all(id_byte))
        .then(|| IdStr::from(v).ok())
        .flatten()
}

/// A host name from untrusted text, checked as [`de_host`] checks one.
pub fn host_from(v: &str) -> Option<HostStr> {
    (!v.is_empty() && v.bytes().all(host_byte))
        .then(|| HostStr::from(v).ok())
        .flatten()
}

pub(super) fn de_id<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<IdStr>, D::Error> {
    de_token(d, id_byte)
}

pub(super) fn de_host<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<HostStr>, D::Error> {
    de_token(d, host_byte)
}

/// Any JSON value: `Some` only for a number.
fn de_num<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Option<f64>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a number")
        }
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v))
        }
        #[allow(clippy::cast_precision_loss)]
        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v as f64))
        }
        #[allow(clippy::cast_precision_loss)]
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as f64))
        }
        fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a
                .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                .is_some()
            {}
            Ok(None)
        }
    }
    d.deserialize_any(V)
}

/// A count: zero unless a non-negative integer, saturated to `u32`.
fn de_count<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(de_num(d)?
        .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0)
        .map_or(0, |n| n.min(f64::from(u32::MAX)) as u32))
}

/// A boolean, or `None` for anything else.
fn de_bool<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Option<bool>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a boolean")
        }
        fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
            Ok(Some(v))
        }
        fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a
                .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                .is_some()
            {}
            Ok(None)
        }
    }
    d.deserialize_any(V)
}

/// The vendor-specific usage members beyond token counts, shared by every usage shape (each usage
/// struct carries one as a `vendor` field, filled by `#[serde(flatten)]`-free duplication: see
/// [`vendor_fields`]). OpenRouter's cost accounting and server-tool counts, xAI's cost ticks and
/// tool details, Anthropic's (and OpenRouter Messages') `server_tool_use`, `speed` and
/// `inference_geo`.
#[derive(Default)]
pub(super) struct VendorUsage {
    pub cost: Option<f64>,
    pub is_byok: Option<bool>,
    pub cost_details: Option<CostDetails>,
    pub cost_in_usd_ticks: Option<f64>,
    pub server_tool_use: Option<ToolUseCounts>,
    pub server_tool_use_details: Option<ToolUseCounts>,
    pub server_side_tool_usage_details: Option<XaiToolDetails>,
    pub num_sources_used: u32,
    pub speed: Option<ServiceTier>,
    pub inference_geo: Option<ServiceTier>,
}

/// Declares the [`VendorUsage`] members as fields of a usage struct and a `vendor()` that moves
/// them out. Plain fields rather than `#[serde(flatten)]`, which buffers every unknown member into
/// an allocated `Content` tree: the cost the typed views exist to avoid.
macro_rules! vendor_fields {
    ($(#[$m:meta])* struct $name:ident { $($body:tt)* }) => {
        $(#[$m])*
        struct $name {
            $($body)*
            /// OpenRouter: USD (credits).
            #[serde(default, deserialize_with = "vendor::de_num_pub")]
            cost: Option<f64>,
            #[serde(default, deserialize_with = "vendor::de_bool_pub")]
            is_byok: Option<bool>,
            #[serde(default)]
            cost_details: Option<vendor::CostDetails>,
            /// xAI: 1e-10 USD.
            #[serde(default, deserialize_with = "vendor::de_num_pub")]
            cost_in_usd_ticks: Option<f64>,
            /// Anthropic; OpenRouter Messages; OpenRouter's overview docs for Chat.
            #[serde(default)]
            server_tool_use: Option<vendor::ToolUseCounts>,
            /// OpenRouter Chat / Responses (its OpenAPI spec's name for the same block).
            #[serde(default)]
            server_tool_use_details: Option<vendor::ToolUseCounts>,
            /// xAI, both APIs.
            #[serde(default)]
            server_side_tool_usage_details: Option<vendor::XaiToolDetails>,
            #[serde(default, deserialize_with = "vendor::de_count_pub")]
            num_sources_used: u32,
            #[serde(default, deserialize_with = "de_service_tier")]
            speed: Option<ServiceTier>,
            #[serde(default, deserialize_with = "de_service_tier")]
            inference_geo: Option<ServiceTier>,
        }

        impl $name {
            fn vendor(&mut self) -> vendor::VendorUsage {
                vendor::VendorUsage {
                    cost: self.cost.take(),
                    is_byok: self.is_byok.take(),
                    cost_details: self.cost_details.take(),
                    cost_in_usd_ticks: self.cost_in_usd_ticks.take(),
                    server_tool_use: self.server_tool_use.take(),
                    server_tool_use_details: self.server_tool_use_details.take(),
                    server_side_tool_usage_details: self.server_side_tool_usage_details.take(),
                    num_sources_used: std::mem::take(&mut self.num_sources_used),
                    speed: self.speed.take(),
                    inference_geo: self.inference_geo.take(),
                }
            }
        }
    };
}
pub(super) use vendor_fields;

pub(super) fn de_num_pub<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    de_num(d)
}
pub(super) fn de_bool_pub<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<bool>, D::Error> {
    de_bool(d)
}
pub(super) fn de_count_pub<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    de_count(d)
}

#[derive(Deserialize, Default)]
pub(super) struct CostDetails {
    #[serde(default, deserialize_with = "de_num")]
    upstream_inference_cost: Option<f64>,
    #[serde(default, deserialize_with = "de_num")]
    server_tool_cost: Option<f64>,
}

#[derive(Deserialize, Default)]
pub(super) struct ToolUseCounts {
    #[serde(default, deserialize_with = "de_count")]
    web_search_requests: u32,
    #[serde(default, deserialize_with = "de_count")]
    web_fetch_requests: u32,
    #[serde(default, deserialize_with = "de_count")]
    code_execution_requests: u32,
    #[serde(default, deserialize_with = "de_count")]
    tool_calls_executed: u32,
}

#[derive(Deserialize, Default)]
pub(super) struct XaiToolDetails {
    #[serde(default, deserialize_with = "de_count")]
    web_search_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    x_search_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    x_posts_fetched: u32,
    #[serde(default, deserialize_with = "de_count")]
    x_users_fetched: u32,
    #[serde(default, deserialize_with = "de_count")]
    code_interpreter_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    file_search_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    mcp_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    document_search_calls: u32,
    #[serde(default, deserialize_with = "de_count")]
    image_generation_calls: u32,
}

impl VendorUsage {
    /// Fold these members into `u`: what this block reports replaces what an earlier block said
    /// (usage blocks are cumulative), and what it leaves out keeps the earlier value.
    pub(super) fn apply(self, u: &mut Usage) {
        let mut t = u.server_tools;
        for c in [self.server_tool_use, self.server_tool_use_details]
            .into_iter()
            .flatten()
        {
            t.web_search = t.web_search.max(c.web_search_requests);
            t.web_fetch = t.web_fetch.max(c.web_fetch_requests);
            t.code_execution = t.code_execution.max(c.code_execution_requests);
            t.tool_calls = t.tool_calls.max(c.tool_calls_executed);
        }
        if let Some(x) = self.server_side_tool_usage_details {
            t.web_search = t.web_search.max(x.web_search_calls);
            t.x_search = t.x_search.max(x.x_search_calls);
            t.x_posts = t.x_posts.max(x.x_posts_fetched);
            t.x_users = t.x_users.max(x.x_users_fetched);
            t.code_execution = t.code_execution.max(x.code_interpreter_calls);
            t.file_search = t.file_search.max(x.file_search_calls);
            t.mcp = t.mcp.max(x.mcp_calls);
            t.document_search = t.document_search.max(x.document_search_calls);
            t.image_generation = t.image_generation.max(x.image_generation_calls);
        }
        t.sources = t.sources.max(self.num_sources_used);
        u.server_tools = t;
        // xAI's ticks are exact integers; OpenRouter's `cost` is a float in USD.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ticks = self
            .cost_in_usd_ticks
            .filter(|t| t.is_finite() && *t >= 0.0 && *t < 1.8e19)
            .map(|t| t.round() as u64);
        if let Some(c) = ticks.or_else(|| self.cost.and_then(usd_to_e10)) {
            u.upstream.cost_e10 = Some(c);
        }
        if let Some(d) = self.cost_details {
            if let Some(c) = d.upstream_inference_cost.and_then(usd_to_e10) {
                u.upstream.inference_cost_e10 = Some(c);
            }
            if let Some(c) = d.server_tool_cost.and_then(usd_to_e10) {
                u.upstream.tool_cost_e10 = Some(c);
            }
        }
        if self.is_byok.is_some() {
            u.upstream.byok = self.is_byok;
        }
        if self.speed.is_some() {
            u.speed = self.speed;
        }
        if self.inference_geo.is_some() {
            u.inference_geo = self.inference_geo;
        }
    }
}
