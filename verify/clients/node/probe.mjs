// Live probes for the Node clients: drive a stock SDK through the gateway the way an application
// would. Same contract as ../py/probe.py: env VERIFY_BASE, VERIFY_KEY, VERIFY_MODEL; prints one
// final line `VERIFY {json}` with ok, calls[{request_id, wire, usage{input_total, output,
// cache_read}}], detail.
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { generateText, streamText, tool, jsonSchema, stepCountIs } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";

const BASE = process.env.VERIFY_BASE;
const KEY = process.env.VERIFY_KEY;
const MODEL = process.env.VERIFY_MODEL;

const calls = [];
const ids = [];

// A fetch that records each response's x-beyond-request-id, handed to every SDK.
const recordingFetch = async (input, init) => {
  const resp = await fetch(input, init);
  ids.push(resp.headers.get("x-beyond-request-id"));
  return resp;
};
const takeId = () => ids.shift() ?? null;
const record = (wire, usage) => calls.push({ request_id: takeId(), wire, usage });

const chatUsage = (u) =>
  u && { input_total: u.prompt_tokens, output: u.completion_tokens, cache_read: u.prompt_tokens_details?.cached_tokens ?? 0 };
const messagesUsage = (u) => {
  const cr = u.cache_read_input_tokens ?? 0;
  const cw = u.cache_creation_input_tokens ?? 0;
  return { input_total: u.input_tokens + cr + cw, output: u.output_tokens, cache_read: cr };
};

const openai = () => new OpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch, maxRetries: 0 });
const anthropic = () => new Anthropic({ baseURL: BASE, apiKey: KEY, fetch: recordingFetch, maxRetries: 0 });

const probes = {
  // E1 via openai-node: non-stream, then a stream with include_usage.
  async chat_basic() {
    const c = openai();
    const r = await c.chat.completions.create({ model: MODEL, max_tokens: 1024, messages: [{ role: "user", content: "Reply with the single word: pong" }] });
    record("chat", chatUsage(r.usage));
    const s = await c.chat.completions.create({ model: MODEL, max_tokens: 1024, stream: true, stream_options: { include_usage: true }, messages: [{ role: "user", content: "Count from 1 to 5." }] });
    let text = "", usage = null, roles = 0;
    for await (const chunk of s) {
      if (chunk.usage) usage = chunk.usage;
      for (const ch of chunk.choices ?? []) {
        if (ch.delta?.role) roles++;
        text += ch.delta?.content ?? "";
      }
    }
    record("chat", chatUsage(usage));
    return [Boolean(r.choices[0].message.content) && text.length > 0 && roles === 1, { roles, text: text.slice(0, 80) }];
  },

  // E2 via @anthropic-ai/sdk: non-stream, then stream().finalMessage().
  async messages_basic() {
    const c = anthropic();
    const r = await c.messages.create({ model: MODEL, max_tokens: 1024, messages: [{ role: "user", content: "Reply with the single word: pong" }] });
    record("messages", messagesUsage(r.usage));
    const final = await c.messages.stream({ model: MODEL, max_tokens: 1024, messages: [{ role: "user", content: "Count from 1 to 5." }] }).finalMessage();
    record("messages", messagesUsage(final.usage));
    return [r.content[0]?.type === "text" && final.content.some((b) => b.type === "text"), { stop: r.stop_reason }];
  },

  // E3 / TRN-1 via openai-node: stock responses.create(), no `store`.
  async responses_basic() {
    const c = openai();
    const r = await c.responses.create({ model: MODEL, input: "Reply with the single word: pong", max_output_tokens: 1024 });
    record("responses", { input_total: r.usage.input_tokens, output: r.usage.output_tokens, cache_read: r.usage.input_tokens_details?.cached_tokens ?? 0 });
    return [r.status === "completed" && r.output_text.length > 0, { status: r.status }];
  },

  // E4 / E7 via openai-node and anthropic-ts: list models and read the card fields.
  async models_list() {
    const o = [];
    for await (const m of openai().models.list()) o.push(m);
    const a = [];
    for await (const m of anthropic().models.list({ limit: 1000 })) a.push(m);
    ids.length = 0;
    const m = o.find((x) => x.id === MODEL);
    const ok = o.length >= 90 && a.length === o.length && m && m.context_window > 0 && m.pricing && m.capabilities;
    return [Boolean(ok), { openai: o.length, anthropic: a.length }];
  },

  // E1 / T1 via the Vercel AI SDK (OpenAI provider, Chat Completions): generate, then a tool loop.
  async ai_sdk_openai() {
    const provider = createOpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch });
    return aiSdk(provider.chat(MODEL), "chat");
  },

  // E2 / T1 via the Vercel AI SDK (Anthropic provider, Messages).
  async ai_sdk_anthropic() {
    const provider = createAnthropic({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch });
    return aiSdk(provider(MODEL), "messages");
  },
};

async function aiSdk(model, wire) {
  const usage = (u) => ({ input_total: u.inputTokens, output: u.outputTokens, cache_read: u.inputTokenDetails?.cacheReadTokens ?? u.cachedInputTokens ?? 0 });
  const g = await generateText({ model, maxOutputTokens: 1024, prompt: "Reply with the single word: pong" });
  record(wire, usage(g.usage));
  const weather = tool({
    description: "Weather for a city",
    inputSchema: jsonSchema({ type: "object", properties: { city: { type: "string" } }, required: ["city"] }),
    execute: async ({ city }) => `Sunny in ${city}, 31C`,
  });
  const s = streamText({ model, maxOutputTokens: 1024, tools: { weather }, stopWhen: stepCountIs(3), prompt: "What's the weather in Paris? Use the weather tool, then answer." });
  const text = await s.text;
  const steps = await s.steps;
  for (const st of steps) record(wire, usage(st.usage));
  const called = steps.some((st) => st.toolCalls?.length);
  return [g.text.length > 0 && called && /31/.test(text), { steps: steps.length, text: text.slice(0, 120) }];
}

let ok = false, detail;
try {
  [ok, detail] = await probes[process.argv[2]]();
} catch (e) {
  detail = { exception: `${e?.name}: ${e?.message}`, stack: String(e?.stack ?? "").slice(0, 1500) };
}
console.log("VERIFY " + JSON.stringify({ ok: Boolean(ok), calls, detail }));
