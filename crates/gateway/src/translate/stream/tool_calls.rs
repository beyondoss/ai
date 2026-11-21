//! Stream: Chat Completions tool-call chunks, gathered by call.

use super::*;

/// One call in a Chat Completions stream, gathered across its deltas.
pub(in crate::translate) struct ChatCall {
    /// The stream's own `index`, when it sends one.
    pub(in crate::translate) index: Option<u64>,
    pub(in crate::translate) id: String,
    pub(super) name: String,
    /// JSON arguments, or a `custom` (free-form) tool's raw `input`.
    pub(super) args: String,
    /// A `custom` tool call (`{"type": "custom", "custom": {name, input}}`), not a function call.
    pub(super) custom: bool,
    /// A Responses client's call to a `custom` tool sent to Messages wrapped (see [`ToolNames`]):
    /// `args` is the `{"input": …}` object, shown only once whole, as the raw input.
    pub(super) wrapped: bool,
    /// The `namespace` a Responses client offered this tool in (see [`ToolNames`]).
    pub(super) namespace: Option<String>,
    pub(super) opened: bool,
    pub(super) closed: bool,
    /// Where the client sees it: an Anthropic block index or a Responses `output_index`.
    pub(super) slot: usize,
    /// The Responses item id (`fc_…`); unused for an Anthropic client.
    pub(super) item: String,
}

pub(in crate::translate) enum CallStep {
    /// The call is ready to show: emit its start, then all of `args` gathered so far.
    Open(usize),
    /// More arguments for an open call.
    Args(usize, String),
}

/// Chat Completions tool-call deltas, keyed the way the stream means them.
///
/// A delta names its call by `index`. Some upstreams repeat the `id` on every delta, some omit
/// `index`, and some reuse one `index` for parallel calls with different ids; keyed on either
/// alone, all three open a new call per delta or pour one call's arguments into another's. A call opens once its
/// name is known (or at the end), so a client never sees a nameless tool.
///
/// Each delta finds its call in O(1) (`slots`, `by_id`) rather than by a backward scan of every
/// call so far, which made a response with many calls quadratic.
#[derive(Default)]
pub(in crate::translate) struct ToolCalls {
    pub(in crate::translate) calls: Vec<ChatCall>,
    /// Per stream `index`: the calls that carry it.
    slots: rustc_hash::FxHashMap<u64, Slot>,
    /// The last call per id, for an upstream that names calls by id alone.
    by_id: rustc_hash::FxHashMap<Box<str>, usize>,
    /// Bytes of ids, names and arguments held (see [`SseBridge::accumulated`]).
    pub(super) held: usize,
}

/// The calls that carry one stream `index`, as [`ToolCalls::locate`] reads them.
#[derive(Default)]
struct Slot {
    /// The last call with this index.
    last: usize,
    /// Calls with this index that have no id yet, in order.
    unnamed: Vec<usize>,
    /// The last call with this index per id.
    by_id: rustc_hash::FxHashMap<Box<str>, usize>,
}

/// Record `at` as the newest call under `id` in `map`, unless a later call already holds it.
fn note_id(map: &mut rustc_hash::FxHashMap<Box<str>, usize>, id: &str, at: usize) {
    match map.get_mut(id) {
        Some(p) => *p = (*p).max(at),
        None => {
            map.insert(id.into(), at);
        }
    }
}

impl ToolCalls {
    pub(super) fn feed(&mut self, c: &Value, steps: &mut Vec<CallStep>) {
        let index = c.get("index").and_then(Value::as_u64);
        let id = non_empty_str(c, "id");
        let at = self.locate(index, id);
        if let Some(id) = id
            && self.calls.get(at).is_some_and(|call| call.id.is_empty())
        {
            self.set_id(at, id.to_owned());
        }
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        // A custom tool's name and input ride `custom`, the way a function's ride `function`.
        let custom = c.get("custom");
        if custom.is_some() || c.get("type").and_then(Value::as_str) == Some("custom") {
            call.custom = true;
        }
        let func = c.get("function").or(custom).unwrap_or(c);
        if call.name.is_empty()
            && let Some(name) = non_empty_str(func, "name")
        {
            name.clone_into(&mut call.name);
            self.held = self.held.saturating_add(name.len());
        }
        let args = func
            .get("arguments")
            .or_else(|| func.get("input"))
            .and_then(Value::as_str)
            .unwrap_or("");
        call.args.push_str(args);
        self.held = self.held.saturating_add(args.len());
        if !call.opened && !call.name.is_empty() {
            self.open(at, steps);
        } else if call.opened && !args.is_empty() {
            steps.push(CallStep::Args(at, args.to_owned()));
        }
    }

    /// Open every call still waiting for its name.
    pub(super) fn open_rest(&mut self, steps: &mut Vec<CallStep>) {
        for at in 0..self.calls.len() {
            if self.calls.get(at).is_some_and(|c| !c.opened) {
                self.open(at, steps);
            }
        }
    }

    pub(in crate::translate) fn open(&mut self, at: usize, steps: &mut Vec<CallStep>) {
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        call.opened = true;
        if call.id.is_empty() {
            self.set_id(at, fresh_id("call"));
        }
        steps.push(CallStep::Open(at));
    }

    /// Give the call at `at`, which has none, its id, and file it under that id.
    pub(in crate::translate) fn set_id(&mut self, at: usize, id: String) {
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        self.held = self.held.saturating_add(id.len());
        note_id(&mut self.by_id, &id, at);
        if let Some(slot) = call.index.and_then(|i| self.slots.get_mut(&i)) {
            if let Some(k) = slot.unnamed.iter().rposition(|&u| u == at) {
                slot.unnamed.remove(k);
            }
            note_id(&mut slot.by_id, &id, at);
        }
        call.id = id;
    }

    /// The call a delta with `index` and `id` continues, else a new one: the last call with the
    /// same `index` whose id is the delta's or not yet known (with no `index`, the last with the
    /// same id; with neither, the last call).
    pub(in crate::translate) fn locate(&mut self, index: Option<u64>, id: Option<&str>) -> usize {
        let found = match (index, id) {
            (Some(i), None) => self.slots.get(&i).map(|s| s.last),
            (Some(i), Some(id)) => self
                .slots
                .get(&i)
                .and_then(|s| s.unnamed.last().copied().max(s.by_id.get(id).copied())),
            (None, Some(id)) => self.by_id.get(id).copied(),
            (None, None) => self.calls.len().checked_sub(1),
        };
        found.unwrap_or_else(|| {
            let at = self.calls.len();
            if let Some(i) = index {
                let slot = self.slots.entry(i).or_default();
                slot.last = at;
                slot.unnamed.push(at);
            }
            // A call's own bookkeeping, held whatever its strings (so an upstream cannot open
            // calls for free).
            self.held = self.held.saturating_add(std::mem::size_of::<ChatCall>());
            self.calls.push(ChatCall {
                index,
                id: String::new(),
                name: String::new(),
                args: String::new(),
                custom: false,
                wrapped: false,
                namespace: None,
                opened: false,
                closed: false,
                slot: 0,
                item: String::new(),
            });
            at
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}
