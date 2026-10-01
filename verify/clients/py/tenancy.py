"""Tenancy sessions (TEN-1, TEN-2): stock Python SDKs driven through the gateway while the control
plane changes under them.

Usage: tenancy.py <scenario>, driven by crates/verify/tests/tenancy_live.rs, with env VERIFY_BASE,
VERIFY_MODEL, VERIFY_CLIENT (openai-py | anthropic-py | claude-code) and VERIFY_KEYS, a JSON map of
key name -> {token, tenant, key_id} minted by the cell (tenant A = "a*", tenant B = "b*").

The cell owns NATS. A scenario asks it to change the control plane with a line on stdout and waits
for the answer on stdin:

  KV put <key> <value>   ->  OK      (the JetStream write was acknowledged)
  KV del <key>           ->  OK
  ROWS <n>               ->  OK <k>  (the gateway's log holds k >= n ai.usage rows, or timed out)

It ends with one line `VERIFY {json}`: ok, calls (one per HTTP call, with the key it used, the
usage the client saw and `expect`, checked against the ledger by the cell), detail.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

BASE = os.environ["VERIFY_BASE"]
MODEL = os.environ["VERIFY_MODEL"]
CLIENT = os.environ.get("VERIFY_CLIENT", "openai-py")
KEYS = json.loads(os.environ["VERIFY_KEYS"])

# The stated bound (tests/claims_security.rs SEC-17): a deny refuses the next request within 2s.
BOUND = 2.0
# A long answer: the stream is still running well after the deny lands.
LONG_ASK = "Write the numbers from one to three hundred as English words, one per line, nothing else."
SHORT_ASK = "Reply with the single word: pong"

calls = []
_lock = threading.Lock()
_proto = threading.Lock()


def control(line):
    """One request to the cell; returns its answer line."""
    with _proto:
        print(line, flush=True)
        answer = sys.stdin.readline().strip()
    if not answer.startswith("OK"):
        raise RuntimeError(f"cell refused {line!r}: {answer!r}")
    return answer


def kv_put(key, value="spend"):
    control(f"KV put {key} {value}")
    return time.monotonic()


def kv_del(key):
    control(f"KV del {key}")
    return time.monotonic()


def record(key, request_id, usage, **expect):
    call = {"key": key, "request_id": request_id, "usage": usage}
    if expect:
        call["expect"] = expect
    with _lock:
        calls.append(call)


# --- clients -------------------------------------------------------------------------------------

def is_anthropic():
    return CLIENT.startswith("anthropic")


def client(key):
    if is_anthropic():
        import anthropic
        return anthropic.Anthropic(base_url=BASE, api_key=KEYS[key]["token"], max_retries=0, timeout=120)
    import openai
    return openai.OpenAI(base_url=f"{BASE}/v1", api_key=KEYS[key]["token"], max_retries=0, timeout=120)


def chat_usage(u):
    d = getattr(u, "prompt_tokens_details", None)
    return {"input_total": u.prompt_tokens, "output": u.completion_tokens,
            "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}


def messages_usage(u):
    cr = getattr(u, "cache_read_input_tokens", None) or 0
    cw = getattr(u, "cache_creation_input_tokens", None) or 0
    return {"input_total": u.input_tokens + cr + cw, "output": u.output_tokens, "cache_read": cr}


def messages_for(ask):
    return [{"role": "user", "content": ask}] if isinstance(ask, str) else ask


def once(key, ask=SHORT_ASK, max_tokens=16, headers=None):
    """One non-streaming call. Returns (status, request_id, usage or None, error body, headers)."""
    c = client(key)
    try:
        if is_anthropic():
            raw = c.messages.with_raw_response.create(model=MODEL, max_tokens=max_tokens,
                                                      messages=messages_for(ask), extra_headers=headers)
            r = raw.parse()
            return 200, raw.headers.get("x-beyond-request-id"), messages_usage(r.usage), None, raw.headers
        raw = c.chat.completions.with_raw_response.create(model=MODEL, max_tokens=max_tokens,
                                                          messages=messages_for(ask), extra_headers=headers)
        r = raw.parse()
        return 200, raw.headers.get("x-beyond-request-id"), chat_usage(r.usage), None, raw.headers
    except Exception as e:  # noqa: BLE001 - the SDK's typed error is the evidence
        resp = getattr(e, "response", None)
        if resp is None:
            raise
        return (resp.status_code, resp.headers.get("x-beyond-request-id"), None,
                {"sdk_error": type(e).__name__, "body": getattr(e, "body", None)}, resp.headers)


def clear_refusal(status, err, want_status, want_type=None):
    """A refusal the SDK surfaces cleanly: the typed status, and a JSON body naming why."""
    body = (err or {}).get("body")
    # openai-py hands back the envelope's inner object; anthropic-py the whole envelope.
    inner = body.get("error", body) if isinstance(body, dict) else None
    ok = status == want_status and isinstance(inner, dict) and bool(inner.get("message"))
    if ok and want_type:
        ok = inner.get("type") == want_type or inner.get("code") == want_type
    return ok


class Stream:
    """A long streaming answer consumed on its own thread. `started` is set after a few content
    chunks; `done` when the stream ends, with the usage the client saw (None if it was cut)."""

    def __init__(self, key, ask=LONG_ASK, max_tokens=2500):
        self.key, self.ask, self.max_tokens = key, ask, max_tokens
        self.started, self.done = threading.Event(), threading.Event()
        self.request_id = self.usage = self.error = self.finish = None
        self.status = None
        self.chunks = 0
        self.ended_at = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            c = client(self.key)
            if is_anthropic():
                s = c.messages.create(model=MODEL, max_tokens=self.max_tokens, stream=True,
                                      messages=messages_for(self.ask))
                self.request_id = s.response.headers.get("x-beyond-request-id")
                self.status = s.response.status_code
                start = delta = None
                for ev in s:
                    if ev.type == "message_start":
                        start = ev.message.usage
                    elif ev.type == "content_block_delta":
                        self._chunk()
                    elif ev.type == "message_delta":
                        delta = ev.usage
                        self.finish = ev.delta.stop_reason
                if start is not None and delta is not None:
                    u = messages_usage(start)
                    u["output"] = delta.output_tokens
                    self.usage = u
            else:
                s = c.chat.completions.create(model=MODEL, max_tokens=self.max_tokens, stream=True,
                                              stream_options={"include_usage": True},
                                              messages=messages_for(self.ask))
                self.request_id = s.response.headers.get("x-beyond-request-id")
                self.status = s.response.status_code
                for chunk in s:
                    if chunk.usage is not None:
                        self.usage = chat_usage(chunk.usage)
                    for ch in chunk.choices:
                        if ch.delta.content:
                            self._chunk()
                        if ch.finish_reason:
                            self.finish = ch.finish_reason
        except Exception as e:  # noqa: BLE001
            resp = getattr(e, "response", None)
            if resp is not None and self.status is None:
                self.status = resp.status_code
                self.request_id = resp.headers.get("x-beyond-request-id")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.ended_at = time.monotonic()
            self.started.set()
            self.done.set()

    def _chunk(self):
        self.chunks += 1
        if self.chunks >= 5:
            self.started.set()

    def wait_started(self, timeout=60):
        if not self.started.wait(timeout) or self.chunks < 5:
            raise RuntimeError(f"stream never got going: {self.error}")

    def wait(self, timeout=180):
        self.done.wait(timeout)
        return self.done.is_set()

    def summary(self):
        return {"status": self.status, "chunks": self.chunks, "finish": self.finish, "usage": self.usage,
                "error": self.error}


def poll_until(key, want, since, bound=BOUND * 2):
    """Send small calls with `key` until one returns `want` (a status). Every call is recorded: a
    200 is an ordinary billed call, a refusal bills nothing. Returns (seconds after `since`, the
    matching call's (status, err), every status seen)."""
    seen = []
    while True:
        sent = time.monotonic()
        status, rid, usage, err, _ = once(key)
        # A refusal is decided at admission and answered at once: time it by its answer. A 200's
        # answer takes as long as the generation; it was admitted when sent, so time it by that.
        now = time.monotonic() if status != 200 else sent
        seen.append(status)
        if status == 200:
            record(key, rid, usage)
        else:
            record(key, rid, None, refused=True)
        if status == want:
            return now - since, (status, err), seen
        if now - since > bound:
            return None, (status, err), seen
        time.sleep(0.05)


# --- TEN-1 ---------------------------------------------------------------------------------------

def revoke(kind):
    """A long stream with key a1 is in flight when the control plane writes `kind`:
    - tenant:    blackhole.{tenant A} spend   -> a1 and a2 402; tenant B keeps working
    - key:       blackhole.key.{a1's id}      -> a1 402; a2 (same tenant) and b1 keep working
    - allowance: allowance.{tenant A}         -> a1 and a2 402 insufficient_quota; b1 keeps working
    The in-flight stream runs to completion and is billed exactly; deleting the entry restores a1
    within the same bound."""
    a1, a2, b1 = KEYS["a1"], KEYS["a2"], KEYS["b1"]
    key, want_type, a2_refused = {
        "tenant": (f"blackhole.{a1['tenant']}", "access_denied", True),
        "key": (f"blackhole.key.{a1['key_id']}", "access_denied", False),
        "allowance": (f"allowance.{a1['tenant']}", "insufficient_quota", True),
    }[kind]
    s = Stream("a1")
    s.wait_started()
    written = kv_put(key, "spend")
    latency, (status, err), seen = poll_until("a1", 402, written)
    refused_at = time.monotonic()
    in_flight_at_refusal = not s.done.is_set()
    a2_status, rid, usage, a2_err, _ = once("a2")
    record("a2", rid, usage, **({"refused": True} if a2_status != 200 else {}))
    b_status, rid, usage, _, _ = once("b1")
    record("b1", rid, usage, **({"refused": True} if b_status != 200 else {}))
    finished = s.wait()
    record("a1", s.request_id, s.usage)
    restored = kv_del(key)
    r_latency, _, r_seen = poll_until("a1", 200, restored)
    detail = {
        "kv": key, "deny_latency_s": latency, "polled": seen, "refusal": err,
        "in_flight_at_refusal": in_flight_at_refusal,
        "stream_ended_after_refusal_s": (s.ended_at - refused_at) if s.ended_at else None,
        "stream": s.summary(), "a2": a2_status, "a2_error": a2_err, "b1": b_status,
        "restore_latency_s": r_latency, "restore_polled": r_seen,
    }
    problems = []
    if latency is None or latency > BOUND:
        problems.append(f"a1 not refused 402 within {BOUND}s of the write")
    elif not clear_refusal(status, err, 402, want_type):
        problems.append(f"refusal is not a clean 402 {want_type}")
    if not in_flight_at_refusal:
        problems.append("the stream ended before the refusal: nothing was in flight (make LONG_ASK longer)")
    if not finished or s.usage is None or s.error or s.finish not in ("stop", "end_turn", "length", "max_tokens"):
        problems.append("the in-flight stream did not run to completion with usage")
    if a2_refused and (a2_status != 402 or not clear_refusal(a2_status, a2_err, 402, want_type)):
        problems.append(f"a2 (same tenant) got {a2_status}, want a clean 402")
    if not a2_refused and a2_status != 200:
        problems.append(f"a2 (a sibling key) got {a2_status}, want 200")
    if b_status != 200:
        problems.append(f"tenant B got {b_status}, want 200")
    if r_latency is None or r_latency > BOUND:
        problems.append(f"a1 not restored within {BOUND}s of the delete")
    detail["problems"] = problems
    return not problems, detail


def revoke_tenant():
    return revoke("tenant")


def revoke_key():
    return revoke("key")


def exhaust_allowance():
    return revoke("allowance")


def claude_code_revoked():
    """A Claude Code session whose bai_v2 key is denied (blackhole.key.{id}) once its first call has
    been billed: the session must end on its own, promptly, saying why — not hang or loop."""
    import pathlib
    import tempfile
    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    os.environ["VERIFY_KEY"] = KEYS["a1"]["token"]  # harness.py reads it at import
    import harness  # noqa: E402 - the W1 fixture and its isolated environment
    home = pathlib.Path(tempfile.mkdtemp(prefix="verify-ten1-cc-"))
    work = home / "repo"
    work.mkdir()
    (work / "calc.py").write_text(harness.CALC)
    (work / "test_calc.py").write_text(harness.TEST)
    a1 = KEYS["a1"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "CLAUDE", "CODEX"))}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"),
               ANTHROPIC_BASE_URL=BASE, ANTHROPIC_API_KEY=a1["token"], DISABLE_TELEMETRY="1",
               DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, ANTHROPIC_SMALL_FAST_MODEL=MODEL, PWD=str(work))
    cmd = [str(harness.BIN / "claude"), "-p", harness.TASK, "--model", MODEL, "--dangerously-skip-permissions",
           "--output-format", "json"]
    out_f, err_f = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    p = subprocess.Popen(cmd, cwd=work, env=env, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL)
    rows = int(control("ROWS 1").split()[1])
    revoked = kv_put(f"blackhole.key.{a1['key_id']}", "spend")
    hung = False
    try:
        code = p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        hung = True
        p.kill()
        code = p.wait()
    took = time.monotonic() - revoked
    out_f.seek(0)
    err_f.seek(0)
    out, err = out_f.read(), err_f.read()
    result = next((e for e in harness.json_lines(out) if e.get("type") == "result"), {})
    shown = json.dumps(result.get("result", "")) + err[-2000:]
    names_it = any(s in shown for s in ("402", "over limit", "suspended", "access_denied"))
    refusals = int(control("REFUSED 402").split()[1])
    detail = {"rows_before_revoke": rows, "exit": code, "hung": hung, "seconds_after_revoke": round(took, 1),
              "is_error": result.get("is_error"), "subtype": result.get("subtype"),
              "result": str(result.get("result", ""))[:600], "stderr_tail": err[-600:],
              "refusals_402": refusals, "num_turns": result.get("num_turns")}
    problems = []
    if hung:
        problems.append("Claude Code was still running 180s after the revocation")
    if not (result.get("is_error") is True or code != 0):
        problems.append("Claude Code reported success after its key was revoked")
    if not names_it:
        problems.append("Claude Code's output does not say the request was refused (402 / over limit)")
    if refusals == 0:
        problems.append("no request was refused: the revocation never met the session")
    if refusals > 12:
        problems.append(f"{refusals} refused requests: Claude Code is retrying a final 402")
    detail["problems"] = problems
    import shutil
    shutil.rmtree(home, ignore_errors=True)
    return not problems, detail


# --- TEN-2 ---------------------------------------------------------------------------------------

# A scripted session: every turn's body is fixed, so two tenants send byte-identical requests.
SESSION = [
    ([{"role": "user", "content": "Name one primary color. One word."}], False),
    ([{"role": "user", "content": "Name one primary color. One word."},
      {"role": "assistant", "content": "Red"},
      {"role": "user", "content": "Name another one. One word."}], False),
    ([{"role": "user", "content": "Count from 1 to 5, comma separated."}], True),
]


def session_turn(key, msgs, stream):
    """One turn; returns (request_id, usage, cache-status header)."""
    c = client(key)
    if not stream:
        status, rid, usage, err, headers = once(key, msgs, max_tokens=32)
        if status != 200:
            raise RuntimeError(f"{key}: {status} {err}")
        return rid, usage, headers.get("x-beyond-cache-status")
    if is_anthropic():
        s = c.messages.create(model=MODEL, max_tokens=64, stream=True, messages=msgs)
        start = delta = None
        for ev in s:
            if ev.type == "message_start":
                start = ev.message.usage
            elif ev.type == "message_delta":
                delta = ev.usage
        u = messages_usage(start)
        u["output"] = delta.output_tokens
    else:
        s = c.chat.completions.create(model=MODEL, max_tokens=64, stream=True,
                                      stream_options={"include_usage": True}, messages=msgs)
        u = None
        for chunk in s:
            if chunk.usage is not None:
                u = chat_usage(chunk.usage)
    return s.response.headers.get("x-beyond-request-id"), u, s.response.headers.get("x-beyond-cache-status")


def run_session(key):
    """The scripted session; each row must carry the cache hit the client was shown."""
    hits = []
    for msgs, stream in SESSION:
        rid, usage, cache = session_turn(key, msgs, stream)
        hits.append(cache)
        record(key, rid, usage, cache_hit=cache == "hit")
    return hits


def replayed(hits):
    """Every non-streaming turn hit. A streamed turn may miss: openai-py closes the connection at
    [DONE], before the gateway sees the upstream end, and the gateway then skips the fill (D119)."""
    return all(h == "hit" or (stream and h is None) for h, (_, stream) in zip(hits, SESSION))


def cache_isolation():
    """Tenant A fills the cache; then A and B run the identical session concurrently: every A turn
    hits (its own fill), no B turn does (A's fill is not B's); then B's own rerun hits."""
    first = run_session("a1")
    # A fill lands when the gateway finishes the request, which can trail the client's last byte;
    # its billing row is written in the same phase, so wait for the rows before replaying.
    control(f"ROWS {len(calls)}")
    out = {}

    def go(key):
        out[key] = run_session(key)

    ts = [threading.Thread(target=go, args=("a1",)), threading.Thread(target=go, args=("b1",))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    control(f"ROWS {len(calls)}")
    b_again = run_session("b1")
    detail = {"a_fill": first, "a_rerun": out.get("a1"), "b_concurrent": out.get("b1"), "b_rerun": b_again}
    # Isolation is strict: tenant B never replays tenant A's fill, streamed or not.
    ok = (all(h is None for h in first) and out.get("b1") == [None] * len(SESSION)
          and replayed(out.get("a1", [])) and replayed(b_again))
    return ok, detail


def provider_of(key):
    status, rid, usage, err, headers = once(key, max_tokens=8)
    if status != 200:
        raise RuntimeError(f"{key}: {status} {err}")
    p = headers.get("x-beyond-provider")
    record(key, rid, usage, provider=p)
    return p


def pin_isolation():
    """Pooled row (anthropic + openrouter). Each bai_v2 key is its own pin. Find a tenant-A key
    pinned to the provider a new caller is NOT ranked to; tenant B's key with the same vpc_id and
    key_id must still be ranked like a new caller, not steered by A's pin."""
    pinned = {}  # A key name -> provider it was pinned to
    names = sorted(k for k in KEYS if k.startswith("pa"))
    fresh = iter(names)
    tries = []
    for name in fresh:
        pinned[name] = provider_of(name)
        if len(set(pinned.values())) > 1:
            break
    if len(set(pinned.values())) < 2:
        return False, {"why": "every call landed on one provider; the ranker never probed the other", "pinned": pinned}
    for _ in range(3):
        try:
            f1 = next(fresh)
            f2 = next(fresh)
        except StopIteration:
            break
        ranked = provider_of(f1)
        pinned[f1] = ranked
        other = next((k for k, p in pinned.items() if p != ranked and k not in (f1,)), None)
        if other is None:
            continue
        a_follow = provider_of(other)
        twin = "pb" + other[2:]
        b_got = provider_of(twin)
        ranked2 = provider_of(f2)
        pinned[f2] = ranked2
        tries.append({"ranked": ranked, "a_key": other, "a_pinned": pinned[other], "a_followed": a_follow,
                      "b_twin": twin, "b_got": b_got, "ranked_after": ranked2})
        if ranked == ranked2:
            ok = a_follow == pinned[other] and b_got == ranked and b_got != pinned[other]
            return ok, {"pinned": pinned, "tries": tries}
    return False, {"why": "the ranked primary kept flipping; inconclusive", "pinned": pinned, "tries": tries}


def tenant_limit():
    """tenant_max_in_flight = 2. Tenant A holds two streams (two different keys): a third A call is
    refused 429 before the provider; tenant B's two concurrent calls are served; once A's streams
    end, A is served again."""
    s1, s2 = Stream("a1", max_tokens=1500), Stream("a2", max_tokens=1500)
    s1.wait_started()
    s2.wait_started()
    status, rid, usage, err, headers = once("a1")
    record("a1", rid, usage, **({"refused": True} if status != 200 else {}))
    retry_after = headers.get("retry-after") if headers is not None else None
    b = {}

    def go(name):
        b[name] = once("b1")

    ts = [threading.Thread(target=go, args=(n,)) for n in ("x", "y")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    held = not s1.done.is_set() and not s2.done.is_set()
    for st, rid_, u, _, _ in b.values():
        record("b1", rid_, u, **({"refused": True} if st != 200 else {}))
    s1.wait()
    s2.wait()
    record("a1", s1.request_id, s1.usage)
    record("a2", s2.request_id, s2.usage)
    after, rid, usage, _, _ = once("a1")
    record("a1", rid, usage, **({"refused": True} if after != 200 else {}))
    detail = {"a_third": status, "a_third_error": err, "retry_after": retry_after,
              "b": [v[0] for v in b.values()], "a_streams_held_during_b": held,
              "streams": [s1.summary(), s2.summary()], "a_after": after}
    ok = (clear_refusal(status, err, 429) and all(v[0] == 200 for v in b.values()) and held
          and s1.usage is not None and s2.usage is not None and after == 200)
    return ok, detail


def models(c):
    try:
        c.models.list()
        return 200, None
    except Exception as e:  # noqa: BLE001
        resp = getattr(e, "response", None)
        if resp is None:
            raise
        return resp.status_code, {"sdk_error": type(e).__name__, "body": getattr(e, "body", None),
                                  "retry_after": resp.headers.get("retry-after")}


def rate_limit():
    """rate_limit_rps = 3 per credential. A burst on a1 is throttled 429; a2 (same tenant, another
    credential) and b1, sent in the same second, are not. /v1/models: the limit sits in front of it
    and it costs nothing."""
    ca1, ca2, cb1 = client("a1"), client("a2"), client("b1")  # SDK import and setup outside the clock
    t0 = time.monotonic()
    burst = [models(ca1) for _ in range(10)]
    a2, _ = models(ca2)
    b1, _ = models(cb1)
    elapsed = time.monotonic() - t0
    throttled = [e for s, e in burst if s == 429]
    time.sleep(1.2)
    a1_after, _ = models(ca1)
    detail = {"burst": [s for s, _ in burst], "a2": a2, "b1": b1, "elapsed_s": round(elapsed, 3),
              "throttled_error": throttled[0] if throttled else None, "a1_after_window": a1_after}
    # Ten calls inside one second straddle at most two 1s windows: at most 6 are admitted.
    ok = (elapsed < 1.0 and len(throttled) >= 4 and a2 == 200 and b1 == 200 and a1_after == 200
          and clear_refusal(429, throttled[0], 429))
    return ok, detail


def leaked(headers, body):
    """Any 16-character run of a pool key in a response's headers or body."""
    blob = "\n".join(f"{k}: {v}" for k, v in (headers or {}).items()) + "\n" + json.dumps(body, default=str)
    for key in json.loads(os.environ.get("VERIFY_POOL_KEYS") or "[]"):
        if any(key[i:i + 16] in blob for i in range(0, max(len(key) - 15, 1))):
            return True
    return False


def key_rotation():
    """REL-14 / SEC-7. The gateway holds two signing kids, and the provider's pool is mid-rotation:
    its first key is revoked (the provider 401s it), the second is live; the last provider on the
    row has only a revoked key.

    - Tokens from kid 1 (a1, tenant A) and kid 2 (k2, tenant B) are both served, non-streaming and
      streamed, each billed to its own tenant and key: the revoked key costs no request.
    - A kid-2 token signed with kid 1's key, and a kid-3 token, are refused 401 and bill nothing.
    - A call steered (x-beyond-only) to the provider holding only a revoked key gets that provider's
      refusal relayed, and no response anywhere carries a pool key."""
    detail, leaks, ok = {"served": [], "refused": {}}, [], True
    for key in ("a1", "k2", "a1", "k2"):
        status, rid, usage, err, headers = once(key)
        record(key, rid, usage)
        detail["served"].append((key, status))
        ok = ok and status == 200
        if leaked(headers, err):
            leaks.append(key)
    for key in ("a1", "k2"):
        s = Stream(key, SHORT_ASK, max_tokens=16)
        s.done.wait(120)
        record(key, s.request_id, s.usage)
        detail["served"].append((f"{key}:stream", s.status))
        ok = ok and s.status == 200 and s.usage is not None
    for key in ("k2forged", "k3"):
        status, rid, usage, err, headers = once(key)
        record(key, rid, None, refused=True)
        detail["refused"][key] = status
        ok = ok and clear_refusal(status, err, 401)
        if leaked(headers, err):
            leaks.append(key)
    status, rid, usage, err, headers = once("a1", headers={"x-beyond-only": "openrouter"})
    record("a1", rid, usage, **({} if status == 200 else {"refused": True}))
    detail["revoked_only"] = {"status": status, "error": err}
    ok = ok and status in (401, 403) and bool(err)
    if leaked(headers, err):
        leaks.append("revoked_only")
    detail["leaks"] = leaks
    return ok and not leaks, detail


SCENARIOS = {f.__name__: f for f in (revoke_tenant, revoke_key, exhaust_allowance, claude_code_revoked,
                                     cache_isolation, pin_isolation, tenant_limit, rate_limit, key_rotation)}

if __name__ == "__main__":
    try:
        ok, detail = SCENARIOS[sys.argv[1]]()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": calls, "detail": detail}, default=str), flush=True)
