// FLT-1 fault probe for the Node SDKs: one call through the gateway at the SDK's DEFAULT retry
// policy. Same contract as ../py/fault_probe.py: `node fault_probe.mjs <openai|anthropic>
// <stream|nonstream>`, env VERIFY_BASE, VERIFY_KEY, VERIFY_MODEL; prints one final line
// `VERIFY {json}` with ok, terminal, usage, error, attempts, elapsed.
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const BASE = process.env.VERIFY_BASE;
const KEY = process.env.VERIFY_KEY;
const MODEL = process.env.VERIFY_MODEL;
const PROMPT = "Count from 1 to 20, separated by spaces. Output only the numbers.";
const MAX_TOKENS = 80;

const START = performance.now();
const since = () => Math.round(performance.now() - START) / 1000;
const attempts = [];

// Every HTTP attempt the SDK makes goes through this fetch; the SDK's retry loop sits above it.
const recordingFetch = async (input, init) => {
  const a = { t: since(), status: null, request_id: null, retry_after: null };
  attempts.push(a);
  const resp = await fetch(input, init);
  a.status = resp.status;
  a.request_id = resp.headers.get("x-beyond-request-id");
  a.retry_after = resp.headers.get("retry-after");
  return resp;
};

const header = (h, name) => (h ? (typeof h.get === "function" ? h.get(name) : h[name]) ?? null : null);
const describe = (e) => ({
  kind: e?.constructor?.name ?? typeof e,
  status: typeof e?.status === "number" ? e.status : null,
  message: String(e?.message ?? e).slice(0, 500),
  body: e?.error && typeof e.error === "object" ? e.error : null,
  request_id: header(e?.headers, "x-beyond-request-id"),
  retry_after: header(e?.headers, "retry-after"),
});

async function openaiCall(stream) {
  const c = new OpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch });
  const messages = [{ role: "user", content: PROMPT }];
  if (!stream) {
    const r = await c.chat.completions.create({ model: MODEL, max_tokens: MAX_TOKENS, messages });
    const u = r.usage;
    return [true, { input_total: u.prompt_tokens, output: u.completion_tokens, cache_read: u.prompt_tokens_details?.cached_tokens ?? 0 }, r.choices[0].message.content ?? ""];
  }
  const s = await c.chat.completions.create({ model: MODEL, max_tokens: MAX_TOKENS, messages, stream: true, stream_options: { include_usage: true } });
  let terminal = false, usage = null, text = "";
  for await (const chunk of s) {
    if (chunk.usage) usage = { input_total: chunk.usage.prompt_tokens, output: chunk.usage.completion_tokens, cache_read: chunk.usage.prompt_tokens_details?.cached_tokens ?? 0 };
    for (const ch of chunk.choices ?? []) {
      text += ch.delta?.content ?? "";
      if (ch.finish_reason) terminal = true;
    }
  }
  return [terminal, usage, text];
}

async function anthropicCall(stream) {
  const c = new Anthropic({ baseURL: BASE, apiKey: KEY, fetch: recordingFetch });
  const messages = [{ role: "user", content: PROMPT }];
  const usageOf = (u) => {
    const cr = u.cache_read_input_tokens ?? 0;
    const cw = u.cache_creation_input_tokens ?? 0;
    return { input_total: (u.input_tokens ?? 0) + cr + cw, output: u.output_tokens ?? 0, cache_read: cr };
  };
  if (!stream) {
    const r = await c.messages.create({ model: MODEL, max_tokens: MAX_TOKENS, messages });
    return [true, usageOf(r.usage), r.content.filter((b) => b.type === "text").map((b) => b.text).join("")];
  }
  const s = await c.messages.create({ model: MODEL, max_tokens: MAX_TOKENS, messages, stream: true });
  let terminal = false, usage = null, text = "";
  for await (const ev of s) {
    if (ev.type === "message_start") usage = usageOf(ev.message.usage);
    else if (ev.type === "message_delta" && usage) {
      // Counts are cumulative; a translated stream learns its input only at the end.
      usage.output = ev.usage.output_tokens ?? usage.output;
      if (ev.usage.input_tokens != null) {
        const late = usageOf(ev.usage);
        usage.input_total = Math.max(usage.input_total, late.input_total);
        usage.cache_read = Math.max(usage.cache_read, late.cache_read);
      }
    }
    else if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") text += ev.delta.text;
    else if (ev.type === "message_stop") terminal = true;
  }
  return [terminal, usage, text];
}

const [sdk, mode] = process.argv.slice(2);
const out = { ok: false, terminal: false, usage: null, error: null, text: "" };
try {
  const [terminal, usage, text] = await (sdk === "openai" ? openaiCall : anthropicCall)(mode === "stream");
  Object.assign(out, { ok: true, terminal, usage, text: text.slice(0, 200) });
} catch (e) {
  out.error = describe(e);
}
out.attempts = attempts;
out.elapsed = since();
console.log("VERIFY " + JSON.stringify(out));
