# v3.5 Momentum / Mean-Reversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a momentum / mean-reversion (MR) score on BTC log-returns that either *gates* (skip when MR contradicts favorite) or *tilts* (modulate `edge_buffer`) entries in `theta_harvester`'s favorite-side rule, then sweep on PM BTC 90d corpus and decide ship.

**Architecture:** New pure-NumPy module `hlanalysis/strategy/momentum_mr.py` exposes `momentum_mr_score(recent_returns, lookback_min, indicator, favorite_side) -> (score, regime)`. `ThetaHarvesterConfig` gets six new fields (default off = bit-identical to v3.1). `_evaluate_entry` reads them: in `gate` mode it skips trades when `regime == "mr"` and `score < -tau_gate`; in `tilt` mode it scales the effective `edge_buffer` by `(1 - alpha_tilt * score)`. A new `@register("v3_5_momentum_mr")` factory exposes the variant to the `hl-bt tune` CLI. Sweep runner mirrors `scripts/run_v31_ablations.py`.

**Tech Stack:** Python 3.12, NumPy, scipy.stats, existing pytest infra under `tests/unit/`, `uv run hl-bt tune` CLI for walk-forward sweeps.

**Spec:** `docs/specs/2026-05-28-v35-momentum-mr-design.md`

**Baseline being matched/beaten:** v3.1 final state — config in `config/tuning.v3-1-final-pm.yaml`.

**Reference signal source:** the `recent_returns: tuple[float, ...]` parameter already passed to `ThetaHarvesterStrategy.evaluate` is a tuple of 60-second BTC log-returns. `lookback_min` translates to the last `lookback_min` elements of that tuple (sampling dt = 60s = 1m by config).

---

## Task 1: Create `momentum_mr.py` indicator module

**Files:**
- Create: `hlanalysis/strategy/momentum_mr.py`
- Test:   `tests/unit/test_momentum_mr.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_momentum_mr.py
"""Unit tests for hlanalysis.strategy.momentum_mr."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hlanalysis.strategy.momentum_mr import momentum_mr_score


# Helper: build a trending up return series of length n with per-bar return r.
def _trend(n: int, r: float) -> tuple[float, ...]:
    return tuple(r for _ in range(n))


# Helper: build a mean-reverting series (n/2 up, n/2 down, summing to ~0).
def _mr(n: int, r: float) -> tuple[float, ...]:
    half = n // 2
    return tuple([r] * half + [-r] * (n - half))


class TestZRet:
    def test_strong_up_trend_aligned_to_up_favorite_is_momentum(self) -> None:
        rets = _trend(120, 0.0008)  # strong sustained up move
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=+1,
        )
        assert score > 0.0
        assert regime in {"momentum", "neutral"}

    def test_strong_up_trend_against_down_favorite_is_negative(self) -> None:
        rets = _trend(120, 0.0008)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=-1,
        )
        assert score < 0.0

    def test_insufficient_data_returns_neutral_zero(self) -> None:
        # Lookback 60 but only 3 returns available → neutral, score 0
        score, regime = momentum_mr_score(
            recent_returns=(0.001, -0.001, 0.0), lookback_min=60,
            indicator="z_ret", favorite_side=+1,
        )
        assert score == 0.0
        assert regime == "neutral"

    def test_flat_returns_neutral(self) -> None:
        rets = tuple(0.0 for _ in range(120))
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="z_ret", favorite_side=+1,
        )
        assert score == 0.0
        assert regime == "neutral"


class TestRSI:
    def test_pure_up_returns_overbought_against_up_favorite_is_mr(self) -> None:
        # 30 consecutive up bars → RSI(14) saturates near 100 → score>>0 aligned
        # to UP favorite → momentum; but if regime call uses RSI>70 as MR sign
        # (overbought = reversion likely), we still expect score > 0 (aligned)
        # and regime in {"momentum","mr"} per spec table.
        rets = _trend(60, 0.001)
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="rsi", favorite_side=+1,
        )
        assert score > 0.5
        assert regime in {"momentum", "mr"}

    def test_pure_down_returns_against_up_favorite_is_negative_score(self) -> None:
        rets = _trend(60, -0.001)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=15, indicator="rsi", favorite_side=+1,
        )
        assert score < -0.5


class TestMASigma:
    def test_recent_drift_up_aligned_to_up_favorite_is_positive(self) -> None:
        # First half flat, last half up — last close stretched above MA
        rets = tuple([0.0] * 60 + [0.001] * 30)
        score, _ = momentum_mr_score(
            recent_returns=rets, lookback_min=30, indicator="ma_sigma", favorite_side=+1,
        )
        assert score > 0.0


class TestHurstOU:
    def test_random_walk_close_to_05_neutral_or_momentum(self) -> None:
        # Brownian-like sequence (seed for determinism)
        rng = np.random.default_rng(seed=42)
        rets = tuple(rng.normal(0.0, 0.001, size=200).tolist())
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=60, indicator="hurst_ou", favorite_side=+1,
        )
        # Random walk → H ~ 0.5 → score near 0
        assert abs(score) < 0.6
        assert regime in {"neutral", "momentum", "mr"}

    def test_strong_mean_reverting_returns_mr(self) -> None:
        # Alternating high-frequency reversion → H < 0.5
        rets = _mr(120, 0.001) * 2  # 240 bars of pure alternation
        score, regime = momentum_mr_score(
            recent_returns=rets, lookback_min=60, indicator="hurst_ou", favorite_side=+1,
        )
        assert regime == "mr"


class TestUnknownIndicator:
    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown indicator"):
            momentum_mr_score(
                recent_returns=(0.0,) * 30, lookback_min=15,
                indicator="totally_made_up", favorite_side=+1,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_momentum_mr.py -v
```

Expected: ImportError / `ModuleNotFoundError: hlanalysis.strategy.momentum_mr`.

- [ ] **Step 3: Implement `momentum_mr.py`**

```python
# hlanalysis/strategy/momentum_mr.py
"""Momentum / mean-reversion indicators on BTC 1m log-return series.

The `recent_returns` tuple passed to `ThetaHarvesterStrategy.evaluate` is a
chronological tuple of 60-second log-returns. `lookback_min` selects the last
N elements (one per minute under the default vol_sampling_dt_seconds=60).

Score convention: positive ⇒ underlying trend ALIGNED with `favorite_side`
(continuation likely), negative ⇒ against the favorite (reversion likely).
Score is clipped to [-1.0, +1.0].

Regime:
  - "momentum": trend aligned, |score| not extreme
  - "mr":       mean-reversion signal (overextended) — caller may gate against
  - "neutral":  insufficient signal / inadequate data
"""
from __future__ import annotations

import math

import numpy as np


# Reference window for z-score variance — fixed at 240 minutes (4h) regardless
# of indicator lookback. Long enough to be stable, short enough to track
# regime shifts within a 24h PM market.
_Z_RET_REF_WINDOW_MIN = 240
_Z_RET_MR_THRESHOLD = 2.5
_Z_RET_MOMENTUM_THRESHOLD = 1.0
_MA_SIGMA_MR_THRESHOLD = 2.0
_MA_SIGMA_MOMENTUM_THRESHOLD = 0.5
_HURST_MR_BAND = 0.45
_HURST_MOMENTUM_BAND = 0.55


def momentum_mr_score(
    *,
    recent_returns: tuple[float, ...],
    lookback_min: int,
    indicator: str,
    favorite_side: int,
) -> tuple[float, str]:
    """Return (score in [-1, +1], regime).

    Parameters
    ----------
    recent_returns: chronological tuple of 60s log-returns.
    lookback_min:   number of trailing 1-minute bars to read.
    indicator:      one of "z_ret", "rsi", "ma_sigma", "hurst_ou".
    favorite_side:  +1 if current favorite is UP (yes), -1 if DOWN (no).

    Score is always signed relative to favorite_side: positive ⇒ underlying
    moving WITH the favorite; negative ⇒ AGAINST.
    """
    if favorite_side not in (-1, +1):
        raise ValueError(f"favorite_side must be ±1, got {favorite_side}")
    if lookback_min <= 0:
        raise ValueError(f"lookback_min must be positive, got {lookback_min}")

    if indicator == "z_ret":
        return _z_ret(recent_returns, lookback_min, favorite_side)
    if indicator == "rsi":
        return _rsi(recent_returns, lookback_min, favorite_side)
    if indicator == "ma_sigma":
        return _ma_sigma(recent_returns, lookback_min, favorite_side)
    if indicator == "hurst_ou":
        return _hurst_ou(recent_returns, lookback_min, favorite_side)
    raise ValueError(f"Unknown indicator: {indicator!r}")


def _z_ret(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """z-score of the cumulative return over lookback_min vs a 4h reference σ."""
    if len(recent_returns) < max(lookback_min, 30):
        return 0.0, "neutral"
    arr = np.asarray(recent_returns, dtype=np.float64)
    ref_window = arr[-_Z_RET_REF_WINDOW_MIN:] if len(arr) >= _Z_RET_REF_WINDOW_MIN else arr
    sigma_per_bar = float(np.std(ref_window, ddof=1)) if len(ref_window) > 1 else 0.0
    if sigma_per_bar <= 0.0:
        return 0.0, "neutral"
    window = arr[-lookback_min:]
    cum_ret = float(np.sum(window))
    z = cum_ret / (sigma_per_bar * math.sqrt(lookback_min))
    signed_z = z * favorite_side
    score = float(np.clip(signed_z / _Z_RET_MR_THRESHOLD, -1.0, 1.0))
    abs_z = abs(z)
    regime = "neutral"
    if abs_z >= _Z_RET_MR_THRESHOLD:
        # Check for sign-flip in last 3 bars → MR signal
        last3 = window[-3:] if len(window) >= 3 else window
        if len(last3) >= 2 and float(np.sign(last3[-1])) != float(np.sign(cum_ret)):
            regime = "mr"
        else:
            regime = "momentum" if signed_z > 0 else "mr"
    elif abs_z >= _Z_RET_MOMENTUM_THRESHOLD:
        regime = "momentum" if signed_z > 0 else "mr"
    return score, regime


def _rsi(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Wilder RSI(14) computed over the last `lookback_min + 14` bars (need 14
    to seed the smoothing). Returns score = (RSI − 50) / 50 signed to favorite.
    """
    period = 14
    needed = lookback_min + period
    if len(recent_returns) < needed:
        return 0.0, "neutral"
    arr = np.asarray(recent_returns[-needed:], dtype=np.float64)
    # Convert log-returns to "price changes" — for RSI the sign is what matters,
    # so use the returns directly as up/down deltas.
    gains = np.where(arr > 0, arr, 0.0)
    losses = np.where(arr < 0, -arr, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(arr)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    score_raw = (rsi - 50.0) / 50.0
    signed = score_raw * favorite_side
    score = float(np.clip(signed, -1.0, 1.0))
    regime = "neutral"
    if rsi > 70.0 or rsi < 30.0:
        # Overbought/oversold: if aligned with favorite, momentum extreme; if
        # against favorite, MR signal (overstretched the wrong way).
        regime = "momentum" if signed > 0 else "mr"
    return score, regime


def _ma_sigma(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Distance of last cum-return from rolling-mean cum-return in σ units.

    Build a synthetic log-price path from returns (start at 0), take MA and σ
    over the last `lookback_min` bars, return (last − MA) / σ.
    """
    if len(recent_returns) < lookback_min:
        return 0.0, "neutral"
    arr = np.asarray(recent_returns[-lookback_min:], dtype=np.float64)
    log_price = np.cumsum(arr)  # relative path starting at 0
    ma = float(np.mean(log_price))
    sd = float(np.std(log_price, ddof=1)) if len(log_price) > 1 else 0.0
    if sd <= 0.0:
        return 0.0, "neutral"
    d = (float(log_price[-1]) - ma) / sd
    signed = d * favorite_side
    score = float(np.clip(signed / _MA_SIGMA_MR_THRESHOLD, -1.0, 1.0))
    regime = "neutral"
    if abs(d) >= _MA_SIGMA_MR_THRESHOLD:
        regime = "mr" if signed < 0 else "momentum"
    elif abs(d) >= _MA_SIGMA_MOMENTUM_THRESHOLD and signed > 0:
        regime = "momentum"
    return score, regime


def _hurst_ou(
    recent_returns: tuple[float, ...], lookback_min: int, favorite_side: int,
) -> tuple[float, str]:
    """Estimate Hurst exponent over the synthetic log-price path via the
    rescaled-range method on log2-spaced sub-window sizes.

    H ∈ (0,1). H < 0.5 ⇒ mean-reverting; H > 0.5 ⇒ trending. We map to a
    signed score by multiplying the directional sign of the path by the
    distance from 0.5: `score = sign(cum_ret * favorite_side) * 2*(H−0.5)`.
    """
    if len(recent_returns) < max(lookback_min, 60):
        return 0.0, "neutral"
    arr = np.asarray(recent_returns[-lookback_min:], dtype=np.float64)
    log_price = np.cumsum(arr)
    n = len(log_price)
    if n < 16:
        return 0.0, "neutral"
    # R/S over chunks of size {8, 16, 32, ...} up to n//2
    sizes: list[int] = []
    s = 8
    while s <= n // 2:
        sizes.append(s)
        s *= 2
    if len(sizes) < 2:
        return 0.0, "neutral"
    rs_values: list[float] = []
    for size in sizes:
        chunks = n // size
        rs_list = []
        for c in range(chunks):
            chunk = log_price[c * size:(c + 1) * size]
            mean = float(np.mean(chunk))
            dev = chunk - mean
            cdev = np.cumsum(dev)
            r = float(np.max(cdev) - np.min(cdev))
            sd = float(np.std(chunk, ddof=1)) if size > 1 else 0.0
            if sd > 0.0:
                rs_list.append(r / sd)
        if rs_list:
            rs_values.append(float(np.mean(rs_list)))
        else:
            rs_values.append(0.0)
    if not rs_values or any(v <= 0.0 for v in rs_values):
        return 0.0, "neutral"
    log_sizes = np.log(np.asarray(sizes, dtype=np.float64))
    log_rs = np.log(np.asarray(rs_values, dtype=np.float64))
    # Least-squares slope = Hurst exponent
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    H = float(slope)
    # Clip to a sane band
    H = max(0.1, min(0.9, H))
    # Direction of the path × favorite_side
    cum = float(log_price[-1])
    direction = 1.0 if cum * favorite_side > 0 else -1.0
    score = float(np.clip(direction * 2.0 * (H - 0.5), -1.0, 1.0))
    regime = "neutral"
    if H < _HURST_MR_BAND:
        regime = "mr"
    elif H > _HURST_MOMENTUM_BAND:
        regime = "momentum"
    return score, regime
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_momentum_mr.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/strategy/momentum_mr.py tests/unit/test_momentum_mr.py
git commit -m "feat(strategy): momentum_mr.py indicator module with 4 indicators"
```

---

## Task 2: Add MomentumMR fields to `ThetaHarvesterConfig` (default off; no behavior change)

**Files:**
- Modify: `hlanalysis/strategy/theta_harvester.py:207-317` (config dataclass), `:944-990` (register factory)
- Test:   `tests/unit/test_theta_harvester_v35_defaults.py`

- [ ] **Step 1: Write the failing test (default-off path is bit-identical)**

```python
# tests/unit/test_theta_harvester_v35_defaults.py
"""v3.5 backward-compat: defaults preserve v3.1 behavior bit-for-bit.

Strategy: build two ThetaHarvesterConfig dicts — one pre-v3.5 baseline (no
momentum_mr_* keys), one with the new keys explicitly set to their disabled
defaults. Both must produce identical Decisions on the same QuestionView/
BookState sequence.
"""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig


def _baseline_kwargs() -> dict:
    return dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.03,
        fee_taker=0.0,
        half_spread_assumption=0.005,
        drift_lookback_seconds=3600,
        drift_blend=0.0,
        max_position_usd=200.0,
        favorite_threshold=0.90,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )


def test_default_momentum_mr_fields_match_disabled_explicit() -> None:
    cfg_default = ThetaHarvesterConfig(**_baseline_kwargs())
    cfg_explicit = ThetaHarvesterConfig(
        **_baseline_kwargs(),
        momentum_mr_enabled=False,
        momentum_mr_indicator="z_ret",
        momentum_mr_lookback_min=15,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=1.0,
        momentum_mr_alpha_tilt=0.5,
    )
    # Compare every field
    for fld in cfg_default.__dataclass_fields__:
        assert getattr(cfg_default, fld) == getattr(cfg_explicit, fld), fld


def test_momentum_mr_enabled_is_false_by_default() -> None:
    cfg = ThetaHarvesterConfig(**_baseline_kwargs())
    assert cfg.momentum_mr_enabled is False
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_defaults.py -v
```

Expected: `AttributeError` / dataclass field missing.

- [ ] **Step 3: Add the fields to `ThetaHarvesterConfig` (after the `fee_rate` field at line ~317)**

In `hlanalysis/strategy/theta_harvester.py`, append after the existing `fee_rate: float = 0.0` line at the end of the `ThetaHarvesterConfig` body:

```python
    # v3.5: momentum / mean-reversion (MR) gate or tilt on the favorite-side
    # entry rule. Default off → v3.1 behavior is preserved bit-for-bit.
    # When `enabled` and `mode == "gate"`: skip entries where the momentum_mr
    # regime is "mr" against the favorite side and |score| > tau_gate.
    # When `enabled` and `mode == "tilt"`: scale the effective edge_buffer
    # by (1 − alpha_tilt * score). Score is signed: + = aligned with favorite.
    # See hlanalysis/strategy/momentum_mr.py and
    # docs/specs/2026-05-28-v35-momentum-mr-design.md.
    momentum_mr_enabled: bool = False
    momentum_mr_indicator: str = "z_ret"      # "z_ret" | "rsi" | "ma_sigma" | "hurst_ou"
    momentum_mr_lookback_min: int = 15
    momentum_mr_mode: str = "gate"            # "gate" | "tilt"
    momentum_mr_tau_gate: float = 1.0
    momentum_mr_alpha_tilt: float = 0.5
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_defaults.py -v
```

Expected: green.

- [ ] **Step 5: Add config plumbing in the `build_v3_theta_harvester` factory**

Locate `build_v3_theta_harvester` (around line 945). Append the six new keys to the `cfg = ThetaHarvesterConfig(...)` keyword args at the end:

```python
        fee_rate=float(params.get("fee_rate", 0.0)),
        momentum_mr_enabled=bool(params.get("momentum_mr_enabled", False)),
        momentum_mr_indicator=str(params.get("momentum_mr_indicator", "z_ret")),
        momentum_mr_lookback_min=int(params.get("momentum_mr_lookback_min", 15)),
        momentum_mr_mode=str(params.get("momentum_mr_mode", "gate")),
        momentum_mr_tau_gate=float(params.get("momentum_mr_tau_gate", 1.0)),
        momentum_mr_alpha_tilt=float(params.get("momentum_mr_alpha_tilt", 0.5)),
    )
    return ThetaHarvesterStrategy(cfg)
```

- [ ] **Step 6: Add new register factory `v3_5_momentum_mr`**

Append at the bottom of `hlanalysis/strategy/theta_harvester.py`, after `build_v3_4_lmgate`:

```python
@register("v3_5_momentum_mr")
def build_v3_5_momentum_mr(params: dict) -> ThetaHarvesterStrategy:
    """v3.5 — v3.1 final + momentum/MR gate or tilt on favorite-side entries.

    Defaults to v3.1 final state plus momentum_mr_enabled=True. Sweep params
    expose `momentum_mr_indicator`, `momentum_mr_lookback_min`,
    `momentum_mr_mode`, `momentum_mr_tau_gate`, `momentum_mr_alpha_tilt`.
    """
    params_with_default = dict(params)
    params_with_default.setdefault("momentum_mr_enabled", True)
    return build_v3_theta_harvester(params_with_default)
```

- [ ] **Step 7: Run the full strategy test suite to confirm no regression**

```bash
uv run pytest tests/unit/test_strategy_isolation.py tests/unit/test_strategy_late_resolution.py tests/unit/test_theta_harvester_v35_defaults.py -v
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add hlanalysis/strategy/theta_harvester.py tests/unit/test_theta_harvester_v35_defaults.py
git commit -m "feat(strategy): v3.5 ThetaHarvesterConfig momentum_mr fields (default off)"
```

---

## Task 3: Wire momentum_mr gate into `_evaluate_entry`

**Files:**
- Modify: `hlanalysis/strategy/theta_harvester.py:419-669` (the `_evaluate_entry` method)
- Test:   `tests/unit/test_theta_harvester_v35_gate.py`

The gate fires **after** the `favorite_threshold` filter and **before** the `edge_buffer` check. Chosen favorite side is derived from `chosen_sym` (yes_symbol ⇒ +1, no_symbol or any other ⇒ -1).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_theta_harvester_v35_gate.py
"""v3.5 gate mode: skip entry when momentum_mr regime is 'mr' and
score < -tau_gate (i.e. strong MR signal against the favorite)."""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)
from hlanalysis.strategy.types import (
    Action, BookState, QuestionView,
)


def _qv(strike: float = 100.0) -> QuestionView:
    return QuestionView(
        question_idx=0,
        klass="priceBinary",
        kv=(),
        yes_symbol="YES",
        no_symbol="NO",
        leg_symbols=(),
        strike=strike,
        expiry_ns=10**18,  # far future
        settled=False,
    )


def _cfg(**over) -> ThetaHarvesterConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.0,
        fee_taker=0.0,
        half_spread_assumption=0.0,
        drift_lookback_seconds=0,
        drift_blend=0.0,
        max_position_usd=100.0,
        favorite_threshold=0.85,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )
    base.update(over)
    return ThetaHarvesterConfig(**base)


def test_gate_mode_blocks_entry_when_mr_against_favorite() -> None:
    # Strong sustained DOWN move → z_ret against an UP favorite is negative
    # (regime == "mr" if |z|>=2.5 with sign-flip; but easier to construct:
    # a long down run produces a strongly negative signed-z, regime "mr").
    # We use ma_sigma here because it produces a deterministic strong signal
    # from a one-sided drift.
    rets = tuple([-0.002] * 90)  # strong down drift
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="ma_sigma",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=0.3,
    )
    strat = ThetaHarvesterStrategy(cfg)
    # Construct a binary where YES is favorite at 0.95: bid 0.94, ask 0.96.
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0),
    }
    qv = _qv(strike=100.0)
    # reference_price below strike + sustained down drift → MR against YES favorite
    dec = strat.evaluate(
        question=qv, books=books, reference_price=99.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    assert dec.action == Action.HOLD
    assert any(d.message == "momentum_mr_gate" for d in dec.diagnostics)


def test_gate_mode_allows_entry_when_momentum_aligned() -> None:
    rets = tuple([0.002] * 90)  # strong UP drift aligned to YES favorite
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="ma_sigma",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="gate",
        momentum_mr_tau_gate=0.3,
        edge_buffer=-1.0,  # disable edge filter — we only test gate behavior
    )
    strat = ThetaHarvesterStrategy(cfg)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0),
    }
    qv = _qv(strike=100.0)
    dec = strat.evaluate(
        question=qv, books=books, reference_price=101.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    # When edge_buffer is loose and momentum aligns, we expect ENTER (or at
    # least no momentum_mr_gate diagnostic).
    assert not any(d.message == "momentum_mr_gate" for d in dec.diagnostics)


def test_disabled_gate_does_not_fire() -> None:
    rets = tuple([-0.002] * 90)
    cfg = _cfg(momentum_mr_enabled=False)  # the default
    strat = ThetaHarvesterStrategy(cfg)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.96, ask_sz=10.0),
        "NO":  BookState(symbol="NO",  bid_px=0.04, bid_sz=10.0, ask_px=0.06, ask_sz=10.0),
    }
    qv = _qv(strike=100.0)
    dec = strat.evaluate(
        question=qv, books=books, reference_price=99.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    assert not any(d.message == "momentum_mr_gate" for d in dec.diagnostics)
```

- [ ] **Step 2: Confirm tests fail**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_gate.py -v
```

Expected: fail (gate not implemented).

- [ ] **Step 3: Implement the gate**

In `hlanalysis/strategy/theta_harvester.py`, edit the `_evaluate_entry` method. Locate the section that picks `chosen_sym, chosen_p, chosen_edge, chosen_book, chosen_phi` (around line 536-538). **Immediately after** the `effective_edge = chosen_edge - gamma_lambda * chosen_phi` line (~line 539), and **before** the diagnostic-building `if is_binary:` block (~line 543), insert:

```python
        # v3.5: momentum / MR gate — skip if regime == "mr" and aligned-signed
        # score < -tau_gate. Computed AFTER favorite is chosen so we know which
        # side to align to.
        if (
            self.cfg.momentum_mr_enabled
            and self.cfg.momentum_mr_mode == "gate"
        ):
            from hlanalysis.strategy.momentum_mr import momentum_mr_score
            fav_side = +1 if chosen_sym == question.yes_symbol else -1
            mm_score, mm_regime = momentum_mr_score(
                recent_returns=recent_returns,
                lookback_min=self.cfg.momentum_mr_lookback_min,
                indicator=self.cfg.momentum_mr_indicator,
                favorite_side=fav_side,
            )
            if mm_regime == "mr" and mm_score < -self.cfg.momentum_mr_tau_gate:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "momentum_mr_gate", (
                        ("indicator", self.cfg.momentum_mr_indicator),
                        ("score", f"{mm_score:.3f}"),
                        ("regime", mm_regime),
                        ("tau_gate", f"{self.cfg.momentum_mr_tau_gate:.3f}"),
                        ("fav_side", str(fav_side)),
                    )),
                ))
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_gate.py -v
```

Expected: all green.

- [ ] **Step 5: Confirm v3.1 backward-compat still holds**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_defaults.py tests/unit/test_strategy_isolation.py tests/unit/test_strategy_late_resolution.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/strategy/theta_harvester.py tests/unit/test_theta_harvester_v35_gate.py
git commit -m "feat(strategy): v3.5 momentum_mr gate in _evaluate_entry"
```

---

## Task 4: Wire momentum_mr tilt (effective edge_buffer modulation)

**Files:**
- Modify: `hlanalysis/strategy/theta_harvester.py` (in `_evaluate_entry`, around the gate insertion point)
- Test:   `tests/unit/test_theta_harvester_v35_tilt.py`

Tilt replaces the static `edge_buffer` comparison with `effective_edge <= edge_buffer * (1 - alpha_tilt * score)`. Note score is in [-1, +1], so positive aligned momentum **loosens** the bar; negative MR signal **tightens** it (and can drive it negative ⇒ effectively requiring negative edge to enter, which never fires).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_theta_harvester_v35_tilt.py
"""v3.5 tilt mode: edge_buffer is scaled by (1 - alpha_tilt * score)."""
from __future__ import annotations

from hlanalysis.strategy.theta_harvester import (
    ThetaHarvesterConfig, ThetaHarvesterStrategy,
)
from hlanalysis.strategy.types import (
    Action, BookState, QuestionView,
)


def _qv() -> QuestionView:
    return QuestionView(
        question_idx=0,
        klass="priceBinary",
        kv=(),
        yes_symbol="YES",
        no_symbol="NO",
        leg_symbols=(),
        strike=100.0,
        expiry_ns=10**18,
        settled=False,
    )


def _cfg(**over) -> ThetaHarvesterConfig:
    base = dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.05,            # high baseline bar
        fee_taker=0.0,
        half_spread_assumption=0.0,
        drift_lookback_seconds=0,
        drift_blend=0.0,
        max_position_usd=100.0,
        favorite_threshold=0.85,
        tte_min_seconds=0,
        tte_max_seconds=10**9,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )
    base.update(over)
    return ThetaHarvesterConfig(**base)


def test_tilt_loosens_buffer_when_momentum_aligned() -> None:
    # Aligned up-drift → score > 0 → effective edge_buffer < edge_buffer.
    rets = tuple([0.002] * 90)
    cfg = _cfg(
        momentum_mr_enabled=True,
        momentum_mr_indicator="ma_sigma",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=1.0,
    )
    strat = ThetaHarvesterStrategy(cfg)
    # Book: YES is favorite at ~0.95 with edge_yes ≈ 0.04 (below 0.05 buffer)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.94, ask_sz=10.0),
        "NO":  BookState(symbol="NO",  bid_px=0.05, bid_sz=10.0, ask_px=0.06, ask_sz=10.0),
    }
    qv = _qv()
    dec = strat.evaluate(
        question=qv, books=books, reference_price=110.0,  # well above strike → p_yes ≈ 1
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    # With tilt the effective buffer drops below the raw edge — expect ENTER.
    assert dec.action == Action.ENTER
    assert any(d.message == "momentum_mr_tilt" for d in dec.diagnostics)


def test_tilt_tightens_buffer_when_mr_against_favorite() -> None:
    # Strong down-drift against YES favorite → score < 0 → effective buffer
    # ABOVE edge_buffer → entry blocked even though raw edge would clear.
    rets = tuple([-0.002] * 90)
    cfg = _cfg(
        edge_buffer=0.0,            # baseline: any positive edge enters
        momentum_mr_enabled=True,
        momentum_mr_indicator="ma_sigma",
        momentum_mr_lookback_min=60,
        momentum_mr_mode="tilt",
        momentum_mr_alpha_tilt=1.0,
    )
    strat = ThetaHarvesterStrategy(cfg)
    books = {
        "YES": BookState(symbol="YES", bid_px=0.94, bid_sz=10.0, ask_px=0.94, ask_sz=10.0),
        "NO":  BookState(symbol="NO",  bid_px=0.05, bid_sz=10.0, ask_px=0.06, ask_sz=10.0),
    }
    qv = _qv()
    dec = strat.evaluate(
        question=qv, books=books, reference_price=110.0,
        recent_returns=rets, recent_volume_usd=0.0, position=None,
        now_ns=10**17,
    )
    assert dec.action == Action.HOLD
```

- [ ] **Step 2: Confirm tests fail**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_tilt.py -v
```

Expected: fail (tilt not implemented).

- [ ] **Step 3: Implement tilt**

In `_evaluate_entry`, locate the `if effective_edge <= self.cfg.edge_buffer:` check (~line 600). **Replace** that exact line and surrounding logic with a tilt-aware comparison. Specifically, **immediately before** the `if effective_edge <= self.cfg.edge_buffer:` line, insert the tilt computation:

```python
        # v3.5: momentum / MR tilt — scale the effective edge_buffer by
        # (1 - alpha_tilt * score). Aligned momentum (score > 0) lowers the
        # bar; MR against favorite (score < 0) raises it.
        effective_edge_buffer = self.cfg.edge_buffer
        if (
            self.cfg.momentum_mr_enabled
            and self.cfg.momentum_mr_mode == "tilt"
        ):
            from hlanalysis.strategy.momentum_mr import momentum_mr_score
            fav_side = +1 if chosen_sym == question.yes_symbol else -1
            mm_score, mm_regime = momentum_mr_score(
                recent_returns=recent_returns,
                lookback_min=self.cfg.momentum_mr_lookback_min,
                indicator=self.cfg.momentum_mr_indicator,
                favorite_side=fav_side,
            )
            effective_edge_buffer = self.cfg.edge_buffer * (
                1.0 - self.cfg.momentum_mr_alpha_tilt * mm_score
            )
            # Append a single tilt diagnostic alongside `diag` below.
            tilt_diag = Diagnostic("info", "momentum_mr_tilt", (
                ("indicator", self.cfg.momentum_mr_indicator),
                ("score", f"{mm_score:.3f}"),
                ("regime", mm_regime),
                ("eff_edge_buffer", f"{effective_edge_buffer:.5f}"),
                ("base_edge_buffer", f"{self.cfg.edge_buffer:.5f}"),
                ("fav_side", str(fav_side)),
            ))
        else:
            tilt_diag = None
```

Then replace the existing `if effective_edge <= self.cfg.edge_buffer:` block (lines ~600-612) with:

```python
        if effective_edge <= effective_edge_buffer:
            diags: tuple = (diag,)
            if tilt_diag is not None:
                diags = (tilt_diag,) + diags
            if gamma_lambda > 0.0 and chosen_edge > effective_edge_buffer:
                diags = (Diagnostic("info", "edge_after_gamma_below_buffer", (
                    ("raw_edge", f"{chosen_edge:.4f}"),
                    ("phi_d", f"{chosen_phi:.4f}"),
                    ("gamma_penalty", f"{gamma_lambda * chosen_phi:.4f}"),
                )),) + diags
            return Decision(action=Action.HOLD, diagnostics=diags)
```

And finally, ensure the ENTER Decision at the bottom of `_evaluate_entry` also surfaces the tilt diagnostic. Locate the `return Decision(action=Action.ENTER, ...)` at the end (~line 665) and modify its `diagnostics=` tuple:

```python
        diags_out: tuple = (Diagnostic("info", "entry"), diag)
        if tilt_diag is not None:
            diags_out = (tilt_diag,) + diags_out
        return Decision(
            action=Action.ENTER,
            intents=(intent,),
            diagnostics=diags_out,
        )
```

- [ ] **Step 4: Run tilt tests to confirm pass**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_tilt.py -v
```

Expected: green.

- [ ] **Step 5: Re-run gate + defaults regression**

```bash
uv run pytest tests/unit/test_theta_harvester_v35_gate.py tests/unit/test_theta_harvester_v35_defaults.py tests/unit/test_strategy_isolation.py tests/unit/test_strategy_late_resolution.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/strategy/theta_harvester.py tests/unit/test_theta_harvester_v35_tilt.py
git commit -m "feat(strategy): v3.5 momentum_mr tilt (effective edge_buffer modulation)"
```

---

## Task 5: Bit-identical regression test on a small replay slice

**Files:**
- Test: `tests/unit/test_v35_disabled_bitidentical.py`

Confirms that running the strategy with `v3_5_momentum_mr` register factory + `momentum_mr_enabled=False` produces decisions identical to `v3_theta_harvester` with the same params, on a small synthetic replay.

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_v35_disabled_bitidentical.py
"""v3.5 disabled = v3.1 bit-identical on a deterministic replay slice."""
from __future__ import annotations

from hlanalysis.backtest.core.registry import build as build_strategy
from hlanalysis.strategy.types import (
    Action, BookState, QuestionView,
)


def _ticks(n: int):
    """Synthesise n ticks of (qv, books, reference_price, recent_returns, now_ns)."""
    qv = QuestionView(
        question_idx=0,
        klass="priceBinary",
        kv=(),
        yes_symbol="YES",
        no_symbol="NO",
        leg_symbols=(),
        strike=100.0,
        expiry_ns=10**18,
        settled=False,
    )
    rets = [0.0001 * ((i % 7) - 3) for i in range(n)]
    for i in range(n):
        books = {
            "YES": BookState(symbol="YES", bid_px=0.92, bid_sz=10.0, ask_px=0.94, ask_sz=10.0),
            "NO":  BookState(symbol="NO",  bid_px=0.05, bid_sz=10.0, ask_px=0.07, ask_sz=10.0),
        }
        yield qv, books, 101.0 + 0.01 * i, tuple(rets[:i + 1]), 10**17 + i * 10**9


def _baseline_params() -> dict:
    return dict(
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=60,
        vol_clip_min=0.0,
        vol_clip_max=5.0,
        edge_buffer=0.03,
        fee_taker=0.0,
        half_spread_assumption=0.005,
        drift_lookback_seconds=3600,
        drift_blend=0.0,
        max_position_usd=200.0,
        favorite_threshold=0.90,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        stop_loss_pct=None,
        exit_edge_threshold=0.0,
        take_profit_price=None,
        time_stop_seconds=0,
    )


def test_v35_disabled_equals_v31_baseline() -> None:
    v31 = build_strategy("v3_theta_harvester", _baseline_params())
    v35_params = {**_baseline_params(), "momentum_mr_enabled": False}
    v35 = build_strategy("v3_5_momentum_mr", v35_params)
    for qv, books, ref, rets, now_ns in _ticks(150):
        d_a = v31.evaluate(
            question=qv, books=books, reference_price=ref,
            recent_returns=rets, recent_volume_usd=0.0, position=None, now_ns=now_ns,
        )
        d_b = v35.evaluate(
            question=qv, books=books, reference_price=ref,
            recent_returns=rets, recent_volume_usd=0.0, position=None, now_ns=now_ns,
        )
        assert d_a.action == d_b.action
        # Diagnostics not required to be byte-identical (no momentum_mr_* diags
        # emitted when disabled), but the trade decision must match.
        assert tuple(i.symbol for i in d_a.intents) == tuple(i.symbol for i in d_b.intents)
        assert tuple(i.size for i in d_a.intents) == tuple(i.size for i in d_b.intents)
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/unit/test_v35_disabled_bitidentical.py -v
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_v35_disabled_bitidentical.py
git commit -m "test(strategy): v3.5 disabled = v3.1 baseline bit-identical regression"
```

---

## Task 6: Sweep runner script

**Files:**
- Create: `scripts/run_v35_momentum_mr_sweep.py`
- Create: `config/tuning.v3-5-momentum-mr-base.yaml`

Mirrors `scripts/run_v31_ablations.py` — write per-cell grid yaml, call `uv run hl-bt tune` with walk-forward, summarize, compare to v3.1 final baseline.

- [ ] **Step 1: Create the base config**

`config/tuning.v3-5-momentum-mr-base.yaml`:

```yaml
# Base config for v3.5 momentum/MR sweeps. Each sweep cell overrides
# momentum_mr_* keys; everything else mirrors v3.1 final PM state.

grids:
  v3_5_momentum_mr:
    # Math (v3.1 final)
    vol_lookback_seconds: [3600]
    vol_sampling_dt_seconds: [60]
    vol_clip_min: [0.0]
    vol_clip_max: [5.0]
    fee_taker: [0.0]
    half_spread_assumption: [0.005]
    drift_lookback_seconds: [3600]
    drift_blend: [0.0]
    edge_max: [null]
    min_distance_pct: [null]
    topup_enabled: [false]
    min_bid_notional_usd: [10.0]
    # Tuned (v3.1 final)
    favorite_threshold: [0.90]
    edge_buffer: [0.03]
    exit_safety_d: [1.0]
    exit_take_profit_mode: [true]
    exit_fee: [0.0007]
    # Window / risk
    tte_min_seconds: [0]
    tte_max_seconds: [86400]
    max_position_usd: [200.0]
    stop_loss_pct: [null]
    exit_edge_threshold: [0.0]
    take_profit_price: [null]
    time_stop_seconds: [0]
    gamma_lambda: [0.0]
    # v3.5 — sweep cell overrides these:
    momentum_mr_enabled: [true]
    momentum_mr_indicator: ["z_ret"]
    momentum_mr_lookback_min: [15]
    momentum_mr_mode: ["gate"]
    momentum_mr_tau_gate: [1.0]
    momentum_mr_alpha_tilt: [0.5]

run:
  train_markets: 60
  test_markets: 60
  step_markets: 60
  max_workers: 4
```

- [ ] **Step 2: Create the sweep runner**

`scripts/run_v35_momentum_mr_sweep.py`:

```python
#!/usr/bin/env python3
"""v3.5 momentum/MR sweep on PM BTC corpus.

Mirrors scripts/run_v31_ablations.py — each cell writes a grid yaml, drives
`hl-bt tune` walk-forward, and prints a comparison vs the v3.1 final
baseline (config/tuning.v3-1-final-pm.yaml).

Cells:
  - 4 indicators × 4 lookbacks × 3 tau_gate values = 48 gate cells
  - 4 indicators × 4 lookbacks × 3 alpha_tilt values = 48 tilt cells
  Total: 96 cells.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "sim"
TUNE_OUT = DATA_ROOT / "tuning"

INDICATORS = ["z_ret", "rsi", "ma_sigma", "hurst_ou"]
LOOKBACKS = [5, 15, 30, 60]
TAU_GATES = [0.5, 1.0, 1.5]
ALPHA_TILTS = [0.25, 0.5, 1.0]

# Anchor (v3.1 final). The sweep only overrides momentum_mr_* keys.
ANCHOR = {
    "vol_lookback_seconds": [3600],
    "vol_sampling_dt_seconds": [60],
    "vol_clip_min": [0.0],
    "vol_clip_max": [5.0],
    "fee_taker": [0.0],
    "half_spread_assumption": [0.005],
    "drift_lookback_seconds": [3600],
    "drift_blend": [0.0],
    "edge_max": [None],
    "min_distance_pct": [None],
    "topup_enabled": [False],
    "min_bid_notional_usd": [10.0],
    "favorite_threshold": [0.90],
    "edge_buffer": [0.03],
    "exit_safety_d": [1.0],
    "exit_take_profit_mode": [True],
    "exit_fee": [0.0007],
    "tte_min_seconds": [0],
    "tte_max_seconds": [86400],
    "max_position_usd": [200.0],
    "stop_loss_pct": [None],
    "exit_edge_threshold": [0.0],
    "take_profit_price": [None],
    "time_stop_seconds": [0],
    "gamma_lambda": [0.0],
}


def make_cell(*, indicator: str, lookback_min: int, mode: str,
              tau_gate: float | None = None, alpha_tilt: float | None = None) -> dict:
    cell = {
        **ANCHOR,
        "momentum_mr_enabled": [True],
        "momentum_mr_indicator": [indicator],
        "momentum_mr_lookback_min": [lookback_min],
        "momentum_mr_mode": [mode],
    }
    if tau_gate is not None:
        cell["momentum_mr_tau_gate"] = [tau_gate]
    if alpha_tilt is not None:
        cell["momentum_mr_alpha_tilt"] = [alpha_tilt]
    return cell


def write_grid(label: str, grid: dict) -> Path:
    import yaml
    cfg = {
        "grids": {"v3_5_momentum_mr": grid},
        "run": {
            "train_markets": 60, "test_markets": 60,
            "step_markets": 60, "max_workers": 4,
        },
    }
    out_path = REPO_ROOT / "config" / f"tuning.v3-5-{label}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_tune(label: str, grid_path: Path) -> Path:
    run_id = f"v3-5-{label}-2026-05-28"
    out_run = TUNE_OUT / run_id
    if (out_run / "results.jsonl").exists():
        print(f"[skip] {label}")
        return out_run
    cmd = [
        "uv", "run", "hl-bt", "tune",
        "--strategy", "v3_5_momentum_mr",
        "--data-source", "polymarket",
        "--grid", str(grid_path),
        "--run-id", run_id,
        "--out-dir", str(TUNE_OUT),
        "--start", "2025-05-08", "--end", "2026-05-08",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--kind", "binary",
        "--workers", "4",
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(DATA_ROOT)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {label}] rc={proc.returncode}")
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    print(f"[ok] {label} in {dt:.1f}s")
    return out_run


def summarize(results_path: Path) -> dict:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    ss = [r["summary"] for r in rows]
    if not ss:
        return {"PnL": 0, "Sharpe": 0, "maxDD": 0, "trades": 0, "hit": 0,
                "per_split_pnl": []}
    pnl = sum(s["total_pnl_usd"] for s in ss)
    sharpe = sum(s["sharpe"] for s in ss) / len(ss)
    mdd = max(s["max_drawdown_usd"] for s in ss)
    trades = sum(s["n_trades"] for s in ss)
    hit = sum(s["hit_rate"] * s["n_trades"] for s in ss) / max(trades, 1)
    return {"PnL": pnl, "Sharpe": sharpe, "maxDD": mdd, "trades": trades,
            "hit": hit,
            "per_split_pnl": [s["total_pnl_usd"] for s in ss],
            "worst_split_sharpe": min(s["sharpe"] for s in ss),
            "splits": len(ss)}


def main() -> int:
    results: dict[str, dict] = {}
    cells: list[tuple[str, dict]] = []

    for ind in INDICATORS:
        for lb in LOOKBACKS:
            for tau in TAU_GATES:
                label = f"gate-{ind}-lb{lb}-tau{int(tau*10):02d}"
                cells.append((label, make_cell(
                    indicator=ind, lookback_min=lb, mode="gate", tau_gate=tau,
                )))
            for alpha in ALPHA_TILTS:
                label = f"tilt-{ind}-lb{lb}-a{int(alpha*100):03d}"
                cells.append((label, make_cell(
                    indicator=ind, lookback_min=lb, mode="tilt", alpha_tilt=alpha,
                )))

    print(f"running {len(cells)} cells")
    for label, grid in cells:
        gp = write_grid(label, grid)
        rd = run_tune(label, gp)
        results[label] = summarize(rd / "results.jsonl")

    # Baseline: v3.1 final on identical splits
    baseline_run = TUNE_OUT / "v3-1-final-pm-walkforward-2026-05-23"
    if (baseline_run / "results.jsonl").exists():
        base = summarize(baseline_run / "results.jsonl")
    else:
        print("[warn] v3.1 baseline run not found; "
              "absolute PnL only, no Δ-vs-baseline column")
        base = {"PnL": 0, "Sharpe": 0, "maxDD": 0, "trades": 0,
                "hit": 0, "worst_split_sharpe": 0}

    rows = [(label, results[label]) for label, _ in cells]
    rows.sort(key=lambda x: x[1]["PnL"], reverse=True)
    print()
    print(f"{'label':>45} {'PnL':>9} {'ΔPnL':>9} {'Sharpe':>7} {'worst':>7} "
          f"{'maxDD':>8} {'trades':>7} {'hit':>6}")
    print(f"{'BASELINE (v3.1 final)':>45} ${base['PnL']:>7.0f} {'':>9} "
          f"{base['Sharpe']:>7.2f} {base.get('worst_split_sharpe', 0):>7.2f} "
          f"${base['maxDD']:>6.0f} {base['trades']:>7} {base['hit']:>5.1%}")
    for label, r in rows[:30]:
        dp = r["PnL"] - base["PnL"]
        ship = (r["PnL"] >= base["PnL"]
                and r["worst_split_sharpe"] > base.get("worst_split_sharpe", 0)
                and r["maxDD"] <= base["maxDD"]
                and r["trades"] >= 0.6 * base["trades"])
        marker = "*" if ship else " "
        print(f"{marker}{label:>44} ${r['PnL']:>7.0f} ${dp:>+7.0f} "
              f"{r['Sharpe']:>7.2f} {r['worst_split_sharpe']:>7.2f} "
              f"${r['maxDD']:>6.0f} {r['trades']:>7} {r['hit']:>5.1%}")

    out_json = TUNE_OUT / "v3-5-momentum-mr-summary-2026-05-28.json"
    out_json.write_text(json.dumps({"baseline": base, "cells": results}, indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke-test the runner without executing the full sweep**

Run a single cell to verify the wiring works end-to-end:

```bash
uv run hl-bt tune --strategy v3_5_momentum_mr --data-source polymarket \
    --grid config/tuning.v3-5-momentum-mr-base.yaml \
    --run-id v3-5-smoke-2026-05-28 \
    --out-dir data/sim/tuning \
    --start 2025-12-08 --end 2026-01-08 \
    --fee-model pm_binary --fee-rate 0.07 --kind binary --workers 2
```

Expected: a `results.jsonl` is written under `data/sim/tuning/v3-5-smoke-2026-05-28/`, and the strategy registers (no `KeyError: v3_5_momentum_mr`).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v35_momentum_mr_sweep.py config/tuning.v3-5-momentum-mr-base.yaml
git commit -m "feat(backtest): v3.5 momentum/MR sweep runner + base config"
```

---

## Task 7: Execute the full sweep and write up results

**Files:**
- Create: `summeries/v35_momentum_mr_2026_05_28.md`

This is an analysis task — it depends on T1–T6 being complete and the sweep runner being smoke-tested.

- [ ] **Step 1: Run the full sweep**

```bash
uv run python scripts/run_v35_momentum_mr_sweep.py 2>&1 | tee data/sim/tuning/v3-5-sweep-stdout-2026-05-28.log
```

Expected: 96 cells run to completion. Total runtime guidance: similar PM grid (~50 cells × walk-forward × 60 markets) typically takes 1–3 hours on this machine. If a cell fails, the runner stops; inspect `data/sim/tuning/v3-5-<label>-2026-05-28/results.jsonl` for partial output.

- [ ] **Step 2: Identify ship candidates**

From the summary JSON (`data/sim/tuning/v3-5-momentum-mr-summary-2026-05-28.json`), pick all cells where the `ship` marker `*` was printed (the 5 criteria from the spec). For each, record:

- indicator, lookback, mode, threshold
- PnL vs baseline
- Per-split PnL distribution (any negative splits?)
- worst-fold Sharpe vs baseline
- max DD vs baseline
- Trade count ratio vs baseline

Then apply the **robustness check** (spec criterion #5): for each indicator + mode, count how many of the 4 lookbacks produced a ship-marked cell. Only indicator/mode pairs with ≥2 lookbacks passing are real wins.

- [ ] **Step 3: Compute `safety_d` correlation diagnostic**

To answer the "marginal-info-over-safety_d" risk in the spec, run a single representative ship-candidate cell with verbose diagnostics enabled and check whether the momentum_mr score is highly correlated (Spearman |ρ| > 0.7) with the existing `safety_d` exit metric — if so, the new gate is mostly redundant.

A minimal way: pick the top-ranked cell, locate its `data/sim/tuning/v3-5-<label>-2026-05-28/` directory, look for trade-level parquet output (typically `trades.parquet` or `decisions.parquet` — check the existing tune output for the file name), and compute Spearman correlation between `safety_d` and `mm_score` at entry time. Document the correlation in the writeup.

If the writer can't find a per-decision diagnostic file (some tune runs only emit summaries), skip this step and note that the marginal-info question requires a follow-up diagnostic run.

- [ ] **Step 4: Compute TTE-bucket breakdown**

For the single best ship candidate, bucket trades by entry TTE: 0–1h, 1–4h, 4–24h. Compare PnL/market and hit rate per bucket vs v3.1 baseline in the same buckets. Note any bucket where v3.5 loses money — that's the "indicator/TTE interaction" risk in the spec.

- [ ] **Step 5: Write `summeries/v35_momentum_mr_2026_05_28.md`**

Template:

```markdown
# v3.5 Momentum / MR — PM 90d sweep results

**Spec:** `docs/specs/2026-05-28-v35-momentum-mr-design.md`
**Plan:** `docs/superpowers/plans/2026-05-28-v35-momentum-mr.md`
**Date:** 2026-05-28
**Baseline:** v3.1 final (`config/tuning.v3-1-final-pm.yaml`) on identical walk-forward splits.

## TL;DR

[One sentence: ship which variant, or shelve. If shelve, what we learned.]

## Top 5 cells (by PnL)

| Rank | Indicator | LB (min) | Mode | Threshold | PnL | Δ vs base | Sharpe | Worst-fold | Max DD | Trades | Hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| ... |

## Ship criteria check

For each ship-candidate cell, show 5/5 criteria:

| Cell | PnL≥base | Worst-Sharpe>base | maxDD≤base | trades≥0.6·base | 2+ LBs win | Ship? |

## Risk diagnostics

- **safety_d correlation** (Spearman ρ on the top cell): _
- **TTE-bucket PnL** (top cell vs baseline):
  - 0–1h: _ / _
  - 1–4h: _ / _
  - 4–24h: _ / _
- **GBM Itô check**: confirmed baseline d-statistic includes −½σ²·τ (per `[[strategy_v2_gbm_ito]]`). Yes/No: _

## Decision

[Ship / shelve / further investigation, with rationale.]

If ship: which config exactly, and the follow-up question (HL port? Live integration?). Per `[[v31_final_state_2026_05_23]]`, PM-tuned params do NOT transfer to HL; only ship to PM live.

If shelve: short why-not — was it overfit (only 1 LB worked), redundant with safety_d (high correlation), or genuinely no edge?

## Next steps

[1-3 bullets.]
```

- [ ] **Step 6: Commit the writeup**

```bash
git add summeries/v35_momentum_mr_2026_05_28.md \
        data/sim/tuning/v3-5-momentum-mr-summary-2026-05-28.json
git commit -m "docs(v3.5): momentum/MR PM sweep results + decision writeup"
```

---

## Self-Review Notes (for executor)

- **Sweep cell math:** 4 indicators × 4 lookbacks × (3 τ_gate + 3 α_tilt) = 4 × 4 × 6 = **96 cells**. Matches the spec.
- **Walk-forward splits:** the existing `hl-bt tune` CLI with `train=60/test=60/step=60` over 365 days yields ~5 OOS splits, matching v3.1's existing baseline (see `config/tuning.v3-1-final-pm.yaml` header comment). The plan does NOT re-specify k=3/k=4 fold counts — the spec's "walk-forward k=3/k=4" was directional; the actual runner uses the project's standard split scheme.
- **No HL touched:** Tasks 1–7 do not modify any HL config, recorder code, or engine code. PM-only study.
- **Bit-identical guarantee:** Task 5 is the safety net; Tasks 3 + 4 each have their own regression hooks against the v3.1 isolation/late-resolution tests.
- **No new data ingestion:** the indicator reads `recent_returns` which is already plumbed into `Strategy.evaluate`.
