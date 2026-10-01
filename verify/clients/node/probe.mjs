// Live probes for the Node clients: drive a stock SDK through the gateway the way an application
// would. Same contract as ../py/probe.py: env VERIFY_BASE, VERIFY_KEY, VERIFY_MODEL; prints one
// final line `VERIFY {json}` with ok, calls[{request_id, wire, usage{input_total, output,
// cache_read}}, expect?], detail. VERIFY_PROVIDER and VERIFY_CLIENT name the route's first pool
// and the cell's client, for probes shared across SDKs.
import { readFileSync } from "node:fs";
import { randomBytes } from "node:crypto";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { generateText, streamText, tool, jsonSchema, stepCountIs, Output, APICallError } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";

const BASE = process.env.VERIFY_BASE;
const KEY = process.env.VERIFY_KEY;
const MODEL = process.env.VERIFY_MODEL;
const PROVIDER = process.env.VERIFY_PROVIDER ?? "";
const CLIENT = process.env.VERIFY_CLIENT ?? "";

const calls = [];
const ids = [];

// A fetch that records each response's x-beyond-request-id, handed to every SDK.
const recordingFetch = async (input, init) => {
  const resp = await fetch(input, init);
  ids.push(resp.headers.get("x-beyond-request-id"));
  return resp;
};
const takeId = () => ids.shift() ?? null;
// `expect` shapes what the ledger must hold for the call (see live.rs `call_problems`).
const record = (wire, usage, expect) => calls.push({ request_id: takeId(), wire, usage, ...(expect ? { expect } : {}) });

// Reasoning a provider reports beside the completion count (xAI: total = prompt + completion +
// reasoning) is billed output; OpenAI counts it inside completion_tokens.
const outside = (prompt, completion, total, reasoning) => (reasoning && total === prompt + completion + reasoning ? reasoning : 0);
const chatUsage = (u) =>
  u && {
    input_total: u.prompt_tokens,
    output: u.completion_tokens + outside(u.prompt_tokens, u.completion_tokens, u.total_tokens, u.completion_tokens_details?.reasoning_tokens ?? 0),
    cache_read: u.prompt_tokens_details?.cached_tokens ?? 0,
  };
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

// The AI SDK's usage, normalized. It derives totalTokens itself, so it can't say which convention
// the provider used; it does report reasoning separately, so the billed output is one of exactly
// two counts.
const aiUsage = (u) => {
  const r = u.outputTokenDetails?.reasoningTokens ?? u.reasoningTokens ?? 0;
  return {
    input_total: u.inputTokens,
    output: u.outputTokens + outside(u.inputTokens, u.outputTokens, u.totalTokens, r),
    ...(r ? { output_with_reasoning: u.outputTokens + r } : {}),
    cache_read: u.inputTokenDetails?.cacheReadTokens ?? u.cachedInputTokens ?? 0,
  };
};

async function aiSdk(model, wire) {
  const g = await generateText({ model, maxOutputTokens: 1024, prompt: "Reply with the single word: pong" });
  record(wire, aiUsage(g.usage));
  const weather = tool({
    description: "Weather for a city",
    inputSchema: jsonSchema({ type: "object", properties: { city: { type: "string" } }, required: ["city"] }),
    execute: async ({ city }) => `Sunny in ${city}, 31C`,
  });
  const s = streamText({ model, maxOutputTokens: 1024, tools: { weather }, stopWhen: stepCountIs(3), prompt: "What's the weather in Paris? Use the weather tool, then answer." });
  const text = await s.text;
  const steps = await s.steps;
  for (const st of steps) record(wire, aiUsage(st.usage));
  const called = steps.some((st) => st.toolCalls?.length);
  return [g.text.length > 0 && called && /31/.test(text), { steps: steps.length, text: text.slice(0, 120) }];
}

// --- Translation and billing detail (same oracles as the Python probes of the same name) ------

const FIXTURES = new URL("../fixtures/", import.meta.url);
const fixture = (name) => readFileSync(new URL(name, FIXTURES));
const IMAGE_URL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png";
// A per-run nonce heads every cached prefix, so a cell's first call is a cache write.
const NONCE = randomBytes(6).toString("hex");
const SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["city", "country", "population_millions"],
  properties: { city: { type: "string" }, country: { type: "string" }, population_millions: { type: "number" } },
};
const validates = (o) =>
  o && typeof o === "object" && Object.keys(o).sort().join() === "city,country,population_millions" &&
  typeof o.city === "string" && typeof o.country === "string" && typeof o.population_millions === "number";

// ~6k tokens of distinct facts: past Claude Haiku 4.5's 4096-token cache minimum.
function longPrefix(lines = 420) {
  const colors = ["red", "teal", "amber", "violet", "olive", "coral", "slate"];
  const animals = ["heron", "otter", "lynx", "marmot", "ibis", "gecko", "bison", "wren"];
  const cities = ["Oslo", "Lima", "Hanoi", "Accra", "Perth", "Quito", "Riga", "Turin", "Kyoto"];
  const body = Array.from({ length: lines }, (_, i) => `Fact ${i}: the ${colors[i % 7]} ${animals[i % 8]} numbered ${(i * 37) % 1000} lives in ${cities[i % 9]}.`);
  return `Session ${NONCE}. You are a terse assistant. Reference facts follow.\n${body.join("\n")}`;
}

// The AI SDK model for the route: its Anthropic provider where Anthropic serves, else OpenAI Chat.
const aiModel = () =>
  PROVIDER === "anthropic"
    ? [createAnthropic({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch })(MODEL), "messages"]
    : [createOpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch }).chat(MODEL), "chat"];

const CORRUPT_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfAAAA";
const WEATHER_TOOL = { name: "get_weather", description: "Weather for a city", input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };

Object.assign(probes, {
  // T2 via anthropic-ts: thinking plus a tool; turn 2 replays the signed thinking block.
  async thinking_replay() {
    const c = anthropic();
    const msgs = [{ role: "user", content: "What's the weather in Paris? Use the get_weather tool." }];
    const thinking = { type: "enabled", budget_tokens: 1024 };
    const r = await c.messages.create({ model: MODEL, max_tokens: 4096, thinking, tools: [WEATHER_TOOL], messages: msgs });
    record("messages", messagesUsage(r.usage));
    const th = r.content.find((b) => b.type === "thinking");
    const tu = r.content.find((b) => b.type === "tool_use");
    if (!th?.signature || !tu) return [false, { why: "turn 1 lacks a signed thinking block or a tool_use", types: r.content.map((b) => b.type) }];
    msgs.push({ role: "assistant", content: r.content });
    msgs.push({ role: "user", content: [{ type: "tool_result", tool_use_id: tu.id, content: "Sunny, 31C" }] });
    const f = await c.messages.stream({ model: MODEL, max_tokens: 4096, thinking, tools: [WEATHER_TOOL], messages: msgs }).finalMessage();
    record("messages", messagesUsage(f.usage));
    const text = f.content.filter((b) => b.type === "text").map((b) => b.text).join("");
    return [/31/.test(text), { final: text.slice(0, 160) }];
  },

  // T3 via the AI SDK: a base64 image, an image URL and a PDF; the model reads each fixture.
  async ai_sdk_vision() {
    const [model, wire] = aiModel();
    const g = await generateText({
      model,
      maxOutputTokens: 2048,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Three attachments follow: two images (one is a company logo) and a PDF. For each one, reply with the words, digits or brand name it shows, one line per attachment, copied exactly." },
          { type: "image", image: fixture("codeword.png"), mediaType: "image/png" },
          { type: "image", image: new URL(IMAGE_URL) },
          { type: "file", data: fixture("codeword.pdf"), mediaType: "application/pdf", filename: "codeword.pdf" },
        ],
      }],
    });
    record(wire, aiUsage(g.usage));
    const t = g.text.toUpperCase();
    return [t.includes("4821") && t.includes("7316") && t.includes("GOOGLE"), { text: g.text.slice(0, 200) }];
  },

  // T4 via openai-node: json_schema response_format validates.
  async structured_chat() {
    const r = await openai().chat.completions.create({
      model: MODEL,
      max_completion_tokens: 2048,
      messages: [{ role: "user", content: "Give the city, country and population (millions) of Paris." }],
      response_format: { type: "json_schema", json_schema: { name: "city_facts", strict: true, schema: SCHEMA } },
    });
    record("chat", chatUsage(r.usage));
    let obj;
    try {
      obj = JSON.parse(r.choices[0].message.content ?? "");
    } catch {
      return [false, { why: "not JSON", text: r.choices[0].message.content }];
    }
    return [validates(obj), { obj }];
  },

  // T4 via the AI SDK: Output.object over OpenAI Chat (response_format json_schema).
  async ai_sdk_structured() {
    const model = createOpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch }).chat(MODEL);
    const g = await generateText({ model, maxOutputTokens: 2048, output: Output.object({ schema: jsonSchema(SCHEMA) }), prompt: "Give the city, country and population (millions) of Paris." });
    record("chat", aiUsage(g.usage));
    return [validates(g.output), { obj: g.output }];
  },

  // T5 via openai-node: reasoning_effort low and high, no upstream 400.
  async reasoning_effort() {
    const c = openai();
    const seen = {};
    for (const effort of ["low", "high"]) {
      const r = await c.chat.completions.create({ model: MODEL, reasoning_effort: effort, max_completion_tokens: 4096, messages: [{ role: "user", content: "What is 17 * 23? Reply with just the number." }] });
      record("chat", chatUsage(r.usage));
      seen[effort] = r.choices[0].message.content ?? "";
    }
    return [Object.values(seen).every((t) => t.includes("391")), seen];
  },

  // T6 via openai-node / anthropic-ts: a corrupt PNG only the provider can reject arrives as the
  // SDK's typed 400 in its own envelope, with the provider's message, and bills nothing.
  async typed_error() {
    const anthropicClient = CLIENT.startsWith("anthropic");
    const wire = anthropicClient ? "messages" : "chat";
    try {
      if (anthropicClient) {
        await anthropic().messages.create({ model: MODEL, max_tokens: 64, messages: [{ role: "user", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: CORRUPT_PNG } }, { type: "text", text: "What is this?" }] }] });
      } else {
        await openai().chat.completions.create({ model: MODEL, max_completion_tokens: 64, messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: `data:image/png;base64,${CORRUPT_PNG}` } }, { type: "text", text: "What is this?" }] }] });
      }
    } catch (e) {
      record(wire, null, { error: true });
      const typed = anthropicClient ? e instanceof Anthropic.BadRequestError : e instanceof OpenAI.BadRequestError;
      // openai-node unwraps the `error` object into e.error; anthropic-ts keeps the whole body.
      const msg = anthropicClient ? e.error?.error?.message : e.error?.message;
      const envelope = anthropicClient ? e.error?.type === "error" && e.error?.error?.type === "invalid_request_error" : typeof msg === "string";
      return [typed && e.status === 400 && envelope && /image/i.test(msg ?? ""), { name: e.constructor?.name, body: e.error }];
    }
    record(wire, null, { error: true });
    return [false, { why: "a corrupt image was accepted" }];
  },

  // T6 via the AI SDK: the same refusal is an APICallError with status 400 and the provider's message.
  async ai_sdk_error() {
    const [model, wire] = aiModel();
    try {
      await generateText({ model, maxOutputTokens: 64, maxRetries: 0, messages: [{ role: "user", content: [{ type: "image", image: Buffer.from(CORRUPT_PNG, "base64"), mediaType: "image/png" }, { type: "text", text: "What is this?" }] }] });
    } catch (e) {
      record(wire, null, { error: true });
      return [APICallError.isInstance(e) && e.statusCode === 400 && /image/i.test(e.message), { name: e.name, status: e.statusCode, message: e.message?.slice(0, 200) }];
    }
    record(wire, null, { error: true });
    return [false, { why: "a corrupt image was accepted" }];
  },

  // K1 via openai-node: a long system prompt, no cache_control; turns 2+ report cached tokens.
  async auto_cache() {
    const c = openai();
    const msgs = [{ role: "system", content: longPrefix() }];
    const reads = [];
    for (const q of ["Which city does fact 3 name?", "And fact 4?", "And fact 5?"]) {
      msgs.push({ role: "user", content: q });
      const r = await c.chat.completions.create({ model: MODEL, max_completion_tokens: 1024, messages: msgs });
      const u = chatUsage(r.usage);
      reads.push(u.cache_read);
      record("chat", u, reads.length > 1 ? { row_min: { cache_read_tokens: 1 } } : undefined);
      msgs.push({ role: "assistant", content: r.choices[0].message.content || "ok" });
    }
    return [reads.slice(1).every((x) => x > 0), { cache_read: reads }];
  },

  // K1 via the AI SDK's OpenAI Chat provider: the same, as opencode-style callers send it.
  async ai_sdk_cache() {
    const model = createOpenAI({ baseURL: `${BASE}/v1`, apiKey: KEY, fetch: recordingFetch }).chat(MODEL);
    const messages = [];
    const reads = [];
    for (const q of ["Which city does fact 3 name?", "And fact 4?", "And fact 5?"]) {
      messages.push({ role: "user", content: q });
      const g = await generateText({ model, maxOutputTokens: 1024, system: longPrefix(), messages });
      const u = aiUsage(g.usage);
      reads.push(u.cache_read);
      record("chat", u, reads.length > 1 ? { row_min: { cache_read_tokens: 1 } } : undefined);
      messages.push({ role: "assistant", content: g.text || "ok" });
    }
    return [reads.slice(1).every((x) => x > 0), { cache_read: reads }];
  },
});

let ok = false, detail;
try {
  [ok, detail] = await probes[process.argv[2]]();
} catch (e) {
  detail = { exception: `${e?.name}: ${e?.message}`, stack: String(e?.stack ?? "").slice(0, 1500) };
}
console.log("VERIFY " + JSON.stringify({ ok: Boolean(ok), calls, detail }));
