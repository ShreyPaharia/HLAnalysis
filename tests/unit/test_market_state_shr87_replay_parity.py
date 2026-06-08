"""SHR-87 bit-identical gate: engine MarketState routed onto the shared core.

SHR-87 makes ``hlanalysis/engine/market_state.py`` a thin adapter over the
shared ``hlanalysis/marketdata/market_state.py`` core (returns / HL bars / σ /
volume / last_mark math). It is a LIVE-MONEY path, so the refactor must be a
pure no-op: the engine's σ / volume / book / returns query outputs must be
**bit-identical pre/post** on a real-engine replay of the recorded HL corpus.

This test pins that. It loads the *pre-refactor* engine MarketState straight
from the base commit (``a7d4d24``, the parent of the SHR-87 change) as an
independent module, feeds it and the current adapter the IDENTICAL event
stream, and asserts every query output is equal at every sampled scan point.
It is the standing regression gate the Spec-2 plan calls for (replaces the
dual-builder harness).

Two streams are exercised:
  * the recorded ~2h HL HIP-4 fixture corpus (mark-sourced σ, the live HL path);
  * synthetic bbo-sourced + multi-cadence streams (the PM σ path + the
    (symbol, dt) fan-out), which the mark-only corpus does not cover.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hlanalysis.engine.market_state import MarketState as NewMarketState
from hlanalysis.events import (
    BboEvent,
    BookDeltaEvent,
    BookSnapshotEvent,
    FundingEvent,
    HealthEvent,
    LiquidationEvent,
    MarketMetaEvent,
    MarkEvent,
    Mechanism,
    NormalizedEvent,
    OpenInterestEvent,
    OracleEvent,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.strategy.vol import bipower_variation_sigma, sample_std_returns
from hlanalysis.strategy._numba.vol import parkinson_sigma_window

_BASE_COMMIT = "a7d4d24"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_GLOB = "tests/fixtures/hl_hip4/**/*.parquet"

_S = 1_000_000_000


# --------------------------------------------------------------------------
# Load the pre-refactor engine MarketState from the base commit as a module.
# --------------------------------------------------------------------------

def _load_old_market_state() -> Any:
    """Return the ``MarketState`` class as it existed at the base commit.

    Extracts ``hlanalysis/engine/market_state.py`` from ``a7d4d24`` via
    ``git show`` and imports it as a throwaway submodule *inside the
    ``hlanalysis.engine`` package* (so its relative imports resolve), letting
    the test run the OLD implementation and the NEW adapter side-by-side on one
    event stream.
    """
    try:
        src = subprocess.check_output(
            ["git", "show", f"{_BASE_COMMIT}:hlanalysis/engine/market_state.py"],
            cwd=_REPO_ROOT,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"cannot load base-commit MarketState ({exc})")

    import hlanalysis.engine as _engine_pkg

    mod_name = "hlanalysis.engine._shr87_old_market_state_tmp"
    tmp = Path(_engine_pkg.__file__).parent / "_shr87_old_market_state_tmp.py"
    tmp.write_text(src)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, tmp)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.MarketState
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop(mod_name, None)


OldMarketState = _load_old_market_state()


# --------------------------------------------------------------------------
# Recorded HL corpus loader (mirrors ReplayRunner.run_parquet's _gen).
# --------------------------------------------------------------------------

_TYPE_MAP: dict[str, type[Any]] = {
    "trade": TradeEvent,
    "book_snapshot": BookSnapshotEvent,
    "book_delta": BookDeltaEvent,
    "bbo": BboEvent,
    "mark": MarkEvent,
    "oracle": OracleEvent,
    "open_interest": OpenInterestEvent,
    "funding": FundingEvent,
    "liquidation": LiquidationEvent,
    "market_meta": MarketMetaEvent,
    "question_meta": QuestionMetaEvent,
    "settlement": SettlementEvent,
    "health": HealthEvent,
}


def _load_corpus_events() -> list[NormalizedEvent]:
    """Load the recorded HL fixture corpus as NormalizedEvents, sorted by
    arrival. Each ``event=<type>`` partition is read separately because the
    recorder writes per-type schemas (e.g. book_snapshot ``bid_px`` is an
    array, mark/bbo ``bid_px`` is a scalar) which a single union-read can't
    reconcile."""
    import duckdb

    base = _REPO_ROOT / "tests" / "fixtures" / "hl_hip4"
    con = duckdb.connect()
    out: list[NormalizedEvent] = []
    for event_dir in sorted(base.rglob("event=*")):
        if not event_dir.is_dir():
            continue
        etype = event_dir.name.split("=", 1)[1]
        cls = _TYPE_MAP.get(etype)
        if cls is None:
            continue
        glob = str(event_dir / "**" / "*.parquet")
        rows = con.execute(
            f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        ).fetchall()
        cols = [d[0] for d in con.description]
        import msgspec

        accepted = {f.name for f in msgspec.structs.fields(cls)}
        for row in rows:
            d = dict(zip(cols, row, strict=False))
            clean = {
                k: v for k, v in d.items() if v is not None and k in accepted
            }
            try:
                out.append(cls(**clean))
            except Exception:  # noqa: BLE001 - skip rows the type can't accept
                continue
    out.sort(key=lambda e: (e.exchange_ts or e.local_recv_ts, getattr(e, "seq", 0) or 0))
    return out


# --------------------------------------------------------------------------
# Query-surface comparison.
# --------------------------------------------------------------------------

def _book_tuple(ms: Any, sym: str) -> Any:
    b = ms.book(sym)
    if b is None:
        return None
    return (
        b.bid_px, b.bid_sz, b.ask_px, b.ask_sz,
        b.last_trade_ts_ns, b.last_l2_ts_ns,
        tuple(b.ask_levels), tuple(b.bid_levels),
    )


def _assert_same_books(old: Any, new: Any, syms: list[str], ctx: str) -> None:
    for sym in syms:
        assert _book_tuple(old, sym) == _book_tuple(new, sym), f"book[{sym}] @ {ctx}"


def _assert_same_reference_reads(
    old: Any,
    new: Any,
    *,
    ref: str,
    legs: list[str],
    now_ns: int,
    dt: int,
    n: int,
    lookback_seconds: int,
    ctx: str,
) -> None:
    # last_mark / last_mark_ts
    assert old.last_mark(ref) == new.last_mark(ref), f"last_mark @ {ctx}"
    assert old.last_mark_ts(ref) == new.last_mark_ts(ref), f"last_mark_ts @ {ctx}"

    # recent_returns / recent_hl_bars — both the COUNT path (legacy) and the
    # TIME-bounded path the live scanner/replay actually use.
    for kwargs in (
        {"n": n, "dt": dt},  # count path (no now_ns)
        {"n": n, "dt": dt, "now_ns": now_ns, "lookback_seconds": lookback_seconds},
        {"n": n},  # dt-less count path
        {"n": n, "now_ns": now_ns, "lookback_seconds": lookback_seconds},
    ):
        o_ret = old.recent_returns(ref, **kwargs)
        n_ret = new.recent_returns(ref, **kwargs)
        assert o_ret == n_ret, f"recent_returns({kwargs}) @ {ctx}: {o_ret!r} != {n_ret!r}"

        o_hl = old.recent_hl_bars(ref, **kwargs)
        n_hl = new.recent_hl_bars(ref, **kwargs)
        assert o_hl == n_hl, f"recent_hl_bars({kwargs}) @ {ctx}"

        # σ derived from the same window must agree bit-for-bit.
        _assert_same_sigma(o_ret, o_hl, n_ret, n_hl, ctx=f"{ctx} {kwargs}")

    # recent_volume_usd per leg.
    for leg in legs:
        assert old.recent_volume_usd(leg, now=now_ns) == new.recent_volume_usd(
            leg, now=now_ns
        ), f"recent_volume_usd[{leg}] @ {ctx}"


def _assert_same_sigma(o_ret, o_hl, n_ret, n_hl, *, ctx: str) -> None:
    o_r = np.asarray(o_ret, dtype=np.float64)
    n_r = np.asarray(n_ret, dtype=np.float64)
    assert float(sample_std_returns(o_r)) == float(sample_std_returns(n_r)), f"stdev σ @ {ctx}"
    assert float(bipower_variation_sigma(o_r)) == float(
        bipower_variation_sigma(n_r)
    ), f"bipower σ @ {ctx}"
    o_h = np.ascontiguousarray(np.asarray(o_hl, dtype=np.float64).reshape(-1, 2)[:, 0]) if o_hl else np.empty(0)
    o_l = np.ascontiguousarray(np.asarray(o_hl, dtype=np.float64).reshape(-1, 2)[:, 1]) if o_hl else np.empty(0)
    n_h = np.ascontiguousarray(np.asarray(n_hl, dtype=np.float64).reshape(-1, 2)[:, 0]) if n_hl else np.empty(0)
    n_l = np.ascontiguousarray(np.asarray(n_hl, dtype=np.float64).reshape(-1, 2)[:, 1]) if n_hl else np.empty(0)
    assert float(parkinson_sigma_window(o_h, o_l, 0.0)) == float(
        parkinson_sigma_window(n_h, n_l, 0.0)
    ), f"parkinson σ @ {ctx}"


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

def test_bit_identical_on_recorded_hl_corpus() -> None:
    """Real-engine replay of the recorded HL HIP-4 corpus: every engine query
    output is bit-identical between the pre-SHR-87 MarketState and the adapter."""
    events = _load_corpus_events()
    assert len(events) > 10_000, "fixture corpus failed to load"

    old = OldMarketState()
    new = NewMarketState()
    for ms in (old, new):
        # Mirror the live HL cadence registration: a default (dt-less) cadence
        # plus an explicit dt=5 series, both sized for a 3600s σ window so the
        # bounded deque never truncates a within-window bar (live invariant:
        # reference_vol_lookback_seconds ≥ the scanner's read lookback).
        ms.set_reference_cadence("BTC", sampling_dt_seconds=60, lookback_seconds=3600)
        ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)

    legs = ["#150", "#151"]
    syms = ["BTC", *legs]
    sample_every = 250
    compared = 0
    for i, ev in enumerate(events):
        old.apply(ev)
        new.apply(ev)
        if i % sample_every != 0:
            continue
        now_ns = ev.exchange_ts or ev.local_recv_ts
        ctx = f"event#{i} {type(ev).__name__}"
        _assert_same_books(old, new, syms, ctx)
        for dt in (60, 5):
            _assert_same_reference_reads(
                old, new, ref="BTC", legs=legs, now_ns=now_ns, dt=dt,
                n=400, lookback_seconds=3600, ctx=ctx,
            )
        compared += 1

    assert compared > 50, "too few comparison points — corpus too small?"

    # Final-state full comparison (every leg + reference series).
    final_ns = (events[-1].exchange_ts or events[-1].local_recv_ts)
    _assert_same_books(old, new, syms, "final")
    for dt in (60, 5):
        _assert_same_reference_reads(
            old, new, ref="BTC", legs=legs, now_ns=final_ns, dt=dt,
            n=10_000, lookback_seconds=3600, ctx="final",
        )


def _mark(symbol: str, px: float, ts: int) -> MarkEvent:
    return MarkEvent(
        venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
        symbol=symbol, exchange_ts=ts, local_recv_ts=ts, mark_px=px,
    )


def _bbo(symbol: str, bid: float, ask: float, ts: int) -> BboEvent:
    return BboEvent(
        venue="binance", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
        symbol=symbol, exchange_ts=ts, local_recv_ts=ts,
        bid_px=bid, bid_sz=5.0, ask_px=ask, ask_sz=5.0,
    )


def test_bit_identical_bbo_sourced_sigma() -> None:
    """PM-style σ path: a bbo-sourced symbol drives σ/OHLC from the BBO mid."""
    old = OldMarketState()
    new = NewMarketState()
    for ms in (old, new):
        ms.set_reference_cadence("BTCUSDT", sampling_dt_seconds=5, lookback_seconds=3600)
        ms.set_reference_source("BTCUSDT", "bbo")

    base = 100 * _S
    events: list[NormalizedEvent] = []
    for i in range(600):
        bid = 80_000.0 + (i % 13) * 4.0 - (i % 5) * 1.5
        events.append(_bbo("BTCUSDT", bid, bid + 2.0, ts=base + i * 1_300_000_000))
        if i % 7 == 0:  # a stray mark for the bbo-sourced symbol must be ignored
            events.append(_mark("BTCUSDT", 999_999.0, ts=base + i * 1_300_000_000 + 1))

    for i, ev in enumerate(events):
        old.apply(ev)
        new.apply(ev)
        if i % 60 and i != len(events) - 1:
            continue
        now_ns = ev.exchange_ts
        ctx = f"bbo event#{i}"
        _assert_same_books(old, new, ["BTCUSDT"], ctx)
        _assert_same_reference_reads(
            old, new, ref="BTCUSDT", legs=[], now_ns=now_ns, dt=5,
            n=10_000, lookback_seconds=3600, ctx=ctx,
        )


def test_bit_identical_multi_cadence_and_volume() -> None:
    """(symbol, dt) fan-out + trade volume: two cadences off one tick stream,
    plus per-leg rolling volume, stay bit-identical."""
    old = OldMarketState()
    new = NewMarketState()
    for ms in (old, new):
        ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
        ms.set_reference_cadence("BTC", sampling_dt_seconds=2, lookback_seconds=3600)

    base = 1_700_000_000 * _S
    events: list[NormalizedEvent] = []
    for i in range(900):
        ts = base + i * 700_000_000  # 0.7s spacing → multiple ticks per bucket
        events.append(_mark("BTC", 80_000.0 + (i % 11) * 3.0, ts))
        if i % 4 == 0:
            events.append(TradeEvent(
                venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
                mechanism=Mechanism.CLOB, symbol="#30",
                exchange_ts=ts, local_recv_ts=ts, price=0.5 + (i % 3) * 0.01,
                size=1.0 + (i % 5), side="buy",
            ))

    for i, ev in enumerate(events):
        old.apply(ev)
        new.apply(ev)
        if i % 50 and i != len(events) - 1:
            continue
        now_ns = ev.exchange_ts
        ctx = f"multicad event#{i}"
        for dt in (5, 2):
            _assert_same_reference_reads(
                old, new, ref="BTC", legs=["#30"], now_ns=now_ns, dt=dt,
                n=10_000, lookback_seconds=3600, ctx=ctx,
            )
        # dt-less default read resolves to the first registered cadence (dt=5).
        assert old.recent_returns("BTC", n=64) == new.recent_returns("BTC", n=64), ctx
