"""The independent reference implementation of the pricing contract, in exact rationals.

It computes every golden vector's `expect` in `verify/pricing_vectors.json` from that vector's
`row` and the compiled rate table, and shares nothing with `crates/providers/src/pricing.rs` but
the rates themselves: the table comes in as decimal strings (`rates-sync dump --json`), and every
rule below is written from the contract (`crates/providers/ARCHITECTURE.md`, "Pricing contract"),
not from the Rust code. A disagreement between the two is a bug in one of them, never a snapshot
to refresh.

    python3 -I verify/pricing_reference.py TABLE_JSON VECTORS_JSON --check   # exit 1 on any diff
    python3 -I verify/pricing_reference.py TABLE_JSON VECTORS_JSON --write   # rewrite every expect

`mise run pricing:vectors` writes; `crates/rates-sync/tests/pricing_reference.rs` checks (CI).
A new vector is a new `{name, why, row}` entry in the vectors file: `--write` fills its `expect`.
"""
import datetime
import json
import sys
from decimal import Decimal
from fractions import Fraction as F

U128 = 1 << 128
U64 = 1 << 64

# The `server_tools` kinds the row may name (the contract's Server tools table).
KINDS = ["web_search", "web_search_preview", "web_search_page", "web_fetch", "code_execution",
         "file_search", "image_generation", "computer_use", "mcp", "shell", "tool_search",
         "x_search", "x_posts", "x_users", "document_search", "sources", "tool_calls"]


class Unpriced(Exception):
    pass


def fr(s):
    return F(Decimal(s))


class Table:
    def __init__(self, spec):
        self.cards = spec["cards"]
        self.openrouter = spec["openrouter"]
        self.pricing = {p["model"]: p for p in spec["pricing"]}
        self.providers = set(spec["providers"])
        self.credit_fee = fr(spec["openrouter_credit_fee"])
        self.rate_version = spec["rate_version"]


def tokens(row):
    """Normalization: every prompt token in exactly one bucket."""
    cw, c1 = row.get("cache_write_tokens", 0), row.get("cache_write_1h_tokens", 0)
    if c1 > cw:
        raise Unpriced("inconsistent_tokens")
    inp, cr = row.get("input_tokens", 0), row.get("cache_read_tokens", 0)
    if row.get("usage_wire", "anthropic") == "openai":
        fresh = inp - cr - cw
        if fresh < 0:
            raise Unpriced("inconsistent_tokens")
    else:
        fresh = inp
    gw = row.get("gateway_cache_write_tokens", 0)
    if gw > fresh:
        raise Unpriced("inconsistent_tokens")
    return dict(fresh=fresh, gw=gw, read=cr, w5=cw - c1, w1=c1, out=row.get("output_tokens", 0))


def served_class(row):
    t = row.get("service_tier")
    if t in (None, "default", "standard", "on_demand", "auto"):
        tier = "standard"
    elif t in ("priority", "fast"):
        tier = "fast"
    elif t == "ultrafast":
        tier = "ultrafast"
    elif t == "flex":
        tier = "flex"
    else:
        raise Unpriced("unknown_class")
    sp = row.get("speed")
    if sp in (None, "standard"):
        speed = "standard"
    elif sp == "fast":
        speed = "fast"
    else:
        raise Unpriced("unknown_class")
    if speed == "standard":
        return tier
    if tier in ("standard", "fast"):
        return "fast"
    raise Unpriced("unknown_class")


def geo(row):
    g = row.get("inference_geo")
    if g in (None, "global", "not_available"):
        return False
    if g == "us":
        return True
    raise Unpriced("unknown_geo")


def off_peak(op, ts):
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    if op["weekdays_only"] and dt.weekday() >= 5:
        return True
    minute = dt.hour * 60 + dt.minute
    if not any(a <= minute < b for a, b in op["peak"]):
        return True
    d = dt.date().isoformat()
    if d < op["covered"][0] or d > op["covered"][1]:
        raise Unpriced("unknown_calendar")
    return d in op["holidays"]


def fits(usd):
    """The contract's arithmetic bound: an attodollar numerator times at most two multipliers in
    basis points, in 128 bits."""
    if usd * 10**18 * 10**8 >= U128:
        raise Unpriced("overflow")


def side(card, t, row, cls, geo_us, fee):
    prompt = t["fresh"] + t["read"] + t["w5"] + t["w1"]
    lc = card.get("long_context")
    long = bool(lc) and (prompt >= lc["threshold"] if lc["inclusive"] else prompt > lc["threshold"])
    if cls == "standard" and "off_peak" in card and not long:
        if off_peak(card["off_peak"], row.get("unix_secs", 0)):
            cls = "off_peak"
    if cls == "off_peak":
        r = card["off_peak"]["rates"]
    elif cls == "standard":
        r = card.get("long" if long else "standard")
    else:
        r = card.get(cls + ("_long" if long else ""))
    if r is None:
        raise Unpriced("no_rate")
    rate = {k: (fr(v) if v is not None else None) for k, v in r.items()}
    m = F(1)
    if geo_us:
        if "geo_us" not in card:
            raise Unpriced("unknown_geo")
        m *= fr(card["geo_us"])
    m *= fee
    per = F(1, 1_000_000)  # rates are USD per million tokens
    if t["w1"] > 0 and rate["cache_write_1h"] is None:
        raise Unpriced("no_rate")
    # The gateway's own cache writes are inside the uncached input and bill as 5-minute writes.
    parts = [
        (t["fresh"] - t["gw"]) * rate["input"],
        t["read"] * rate["cache_read"],
        (t["w5"] + t["gw"]) * rate["cache_write"],
        t["w1"] * (rate["cache_write_1h"] or 0),
        t["out"] * rate["output"],
    ]
    usd = F(0)
    for p in parts:
        fits(p * per)
        usd += p * per * m
    for tool, n in row["server_tools"].items():
        if n == 0 or tool == "tool_calls":
            continue
        fees = card.get("tools", {})
        if tool not in fees:
            raise Unpriced("no_tool_fee")
        fits(n * fr(fees[tool]))
        usd += n * fr(fees[tool]) * fee  # residency multiplies token categories only
    fits(usd)
    return usd, cls, long


def micros(usd):
    q = (usd * 1_000_000 + F(1, 2)).__floor__()  # once per side, half-up
    if q >= U64:
        raise Unpriced("overflow")
    return int(q)


def parse_cost(s):
    try:
        d = Decimal(s)
    except Exception:
        raise Unpriced("bad_reported_cost")
    if (not d.is_finite() or d < 0 or s.strip() != s or s.startswith(("+", "-"))
            or "inf" in s.lower() or "nan" in s.lower()):
        raise Unpriced("bad_reported_cost")
    return F(d)


def parse_tools(text):
    out = {}
    for pair in [x for x in (text or "").split(",") if x]:
        if "=" not in pair:
            raise Unpriced("unknown_tool")
        k, n = pair.split("=", 1)
        if k not in KINDS or not n.isdigit():
            raise Unpriced("unknown_tool")
        out[k] = int(n)
    return out


def price(tb, row):
    row = dict(row, server_tools=parse_tools(row.get("server_tools")))
    if row.get("cache_hit"):
        return dict(status="priced", basis="cache_hit", cost_micros=0, price_micros=0,
                    cost_class="standard", price_class="standard", cost_long=False, price_long=False)
    m = row.get("price_model")
    if m is None:
        raise Unpriced("no_price_model")
    if m not in tb.pricing:
        raise Unpriced("unknown_model")
    p = tb.pricing[m]
    prov = row.get("provider")
    if prov not in tb.providers:
        raise Unpriced("unknown_provider")
    if prov not in p["cost"]:
        raise Unpriced("not_a_candidate")
    v = row.get("price_variant")
    if not (v is None or (prov == "bedrock" and v == "regional")
            or (prov == "openrouter" and v in ("regional", "global"))):
        raise Unpriced("unknown_variant")
    st = row["server_tools"]
    t = tokens(row)
    cls = served_class(row)
    if prov in ("anthropic", "bedrock") and row.get("service_tier") not in (None, "standard"):
        raise Unpriced("unknown_class")
    g = geo(row)
    status = "estimated" if row.get("usage_estimated") or row.get("upstream_may_continue") else "priced"
    cn = p["cost"][prov]
    if cn.startswith("unverified"):
        raise Unpriced("unverified")
    if row.get("upstream_cost_usd") is not None:
        cusd = parse_cost(row["upstream_cost_usd"]) * (tb.credit_fee if prov == "openrouter" else 1)
        fits(cusd)
        ccls, clong, basis = cls, False, "reported"
    elif cn.startswith("openrouter:"):
        if st.get("tool_calls", 0) > st.get("web_search", 0):
            raise Unpriced("no_tool_fee")
        status, basis = "estimated", "dearest_endpoint"
        if g:
            raise Unpriced("unknown_geo")
        host = row.get("served_by")
        best = None
        for e in tb.openrouter[cn.split(":", 1)[1]]:
            if e["cls"] != cls or (host is not None and e["host"].lower() != host.lower()):
                continue
            c = dict(standard=e["standard"])
            if "long" in e:
                c["long"], c["long_context"] = e["long"], e["long_context"]
            if "web_search" in e:
                c["tools"] = {"web_search": e["web_search"]}
            got = side(c, t, row, "standard", False, tb.credit_fee)
            if best is None or got[0] > best[0]:
                best = got
        if best is None:
            raise Unpriced("unknown_host")
        cusd, _, clong = best
        ccls = cls
    else:
        card = tb.cards[p["customer"] if cn == "list" else cn]
        cusd, ccls, clong = side(card, t, row, cls, g, F(1))
        basis = "tokens"
    # Pass-through: the price is the cost, on the serving host's card, fees included.
    c = micros(cusd)
    return dict(status=status, basis=basis, cost_micros=c, price_micros=c,
                cost_class=ccls, price_class=ccls, cost_long=clong, price_long=clong)


def expect(tb, row):
    try:
        return price(tb, row)
    except Unpriced as e:
        return {"unpriced": str(e)}


def main():
    table_path, vectors_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    tb = Table(json.load(open(table_path)))
    raw = open(vectors_path).read()
    doc = json.loads(raw)
    diffs = []
    if doc["rate_version"] != tb.rate_version:
        diffs.append(f"rate_version: vectors {doc['rate_version']}, table {tb.rate_version}")
    for v in doc["vectors"]:
        got = expect(tb, v["row"])
        if v.get("expect") != got:
            diffs.append(f"{v['name']}: recorded {v.get('expect')}, reference {got}")
        v["expect"] = got
    doc["rate_version"] = tb.rate_version
    if mode == "--write":
        open(vectors_path, "w").write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"{len(doc['vectors'])} vectors, {len(diffs)} changed")
        for d in diffs:
            print("  " + d)
    elif mode == "--check":
        if diffs:
            print(f"{len(diffs)} of {len(doc['vectors'])} vectors differ from the reference:")
            for d in diffs:
                print("  " + d)
            sys.exit(1)
        print(f"{len(doc['vectors'])} vectors reproduce exactly")
    else:
        sys.exit("mode is --check or --write")


if __name__ == "__main__":
    main()
