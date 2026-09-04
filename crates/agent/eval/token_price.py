"""One OpenRouter list-price card for Harbor A/B tables.

Harbor's Pi adapter copies billed `usage.cost.total`. Our adapter never sees that
field, so it used to multiply tokens by this card while Pi's `$` column stayed on
OpenRouter's invoice. Compare tokens; when a `$` is needed, price *both* arms here.
Harbor `n_input_tokens` includes cache; fresh input is `n_input - n_cache`.
"""

from __future__ import annotations

# Kimi K3: FrontierHarness freeze (skills/frontierharness-eval/reference.md).
# GLM 5.3: OpenRouter list ($1.40 / $0.26 cache / $4.40).
_KIMI_K3 = (3.00, 0.30, 15.00)
_GLM_53 = (1.40, 0.26, 4.40)


def token_rates(model_name: str | None) -> tuple[float, float, float]:
    """(input, cache-read, output) USD per million tokens."""
    name = (model_name or "").lower()
    if "glm-5.3" in name:
        return _GLM_53
    return _KIMI_K3


def list_usd(
    n_input_tokens: int,
    n_cache_tokens: int,
    n_output_tokens: int,
    model_name: str | None,
) -> tuple[int, float]:
    """Return (fresh_input_tokens, list-price USD). `n_input_tokens` includes cache."""
    fresh = max(int(n_input_tokens) - int(n_cache_tokens), 0)
    inp, cache, out = token_rates(model_name)
    usd = (
        fresh / 1_000_000 * inp
        + int(n_cache_tokens) / 1_000_000 * cache
        + int(n_output_tokens) / 1_000_000 * out
    )
    return fresh, usd
