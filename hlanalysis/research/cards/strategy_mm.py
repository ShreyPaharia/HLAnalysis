"""Strategy MM: Market-Making on HL Binary Outcome Markets with HL Perp Delta Hedge.

Sim mechanics
-------------
* Replay each binary expiry's recorded BBO + trades chronologically.
* MM posts passive two-sided quotes at BBO (join) or inside by 1 tick (improve).
* Fill model:
  - OPTIMISTIC: filled if a recorded trade prints AT or through the quote price.
    (buy quote at B: filled if a trade prints at price <= B)
    (sell quote at A: filled if a trade prints at price >= A)
  - CONSERVATIVE: filled only if trade prints STRICTLY through the quote (back-of-queue).
    (buy quote at B: filled if a trade prints at price < B)
    (sell quote at A: filled if a trade prints at price > A)
* Mark inventory at the recorded mid (captures adverse-selection drift).
* Resolve held inventory at oracle settlement (0 or 1) via resolve_binary_outcomes.
* Size: $S per side (configurable, default $100).
* Inventory cap: max_inventory_usd per market (default $200), desk-wide cap $2000.
* Quote skew: when |inventory| > skew_threshold, widen the loaded side by 1 tick and
  narrow the other side by 1 tick.

Delta hedge on HL perp
----------------------
* On each binary fill, compute binary's BTC-delta:
  - delta = hedge_ratio * binary_position_size (hedge_ratio=0.50 from Card C empirical)
  - binary YES win = BTC goes UP -> long binary = effectively long BTC (delta positive)
  - hedge by selling perp: if we buy binary YES, sell delta BTC in perp
  - if we buy binary NO, buy delta BTC in perp
* Hedge fills at recorded perp BBO (taker fill: buy at ask, sell at bid).
* Perp fee: 0.13 bps of notional (= perp spread / 2, conservative).
* Track hedge PnL separately: perp mark-to-market using perp BBO.
* Perp PnL at settlement: close hedge at settlement timestamp perp BBO.

Safety gates (all preserved):
* min_bid_notional_usd: 25.0 (spoof filter)
* max_spread_bps: 500.0 (don't quote into absurdly wide spreads)
* stale_data_halt_seconds: 60 (halt quoting if BBO updates stop for >60s)
* inventory_cap_usd: stop quoting one side when at cap
* tte_min_seconds: 1800 (30min — high gamma risk near expiry)
* tte_max_seconds: 72000 (20h — no price discovery on fresh listings)

Interface
---------
build_card(data_root, *, out_dir=None) -> tuple[str, dict]
    Returns (card_html, findings).

Main
----
Run directly to write docs/research/_cards/strategy_mm.html + strategy_mm.json.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlanalysis.research.outcome_markets import resolve_binary_outcomes
from hlanalysis.research.report import Report, fig_to_base64

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_START = "2026-05-06"
CORPUS_END = "2026-06-10"
IS_START = "2026-05-06"
IS_END = "2026-06-03"
OOS_START = "2026-06-04"
OOS_END = "2026-06-10"
SPLIT_H1_START = "2026-05-06"
SPLIT_H1_END = "2026-05-23"
SPLIT_H2_START = "2026-05-24"
SPLIT_H2_END = "2026-06-10"

_NS = 1_000_000_000

# Tick size for binary markets
TICK = 1e-5

# Card C empirical hedge ratio: d(binary_mid)/d(model_prob) ~ 0.50
DEFAULT_HEDGE_RATIO = 0.50

# Perp taker fee: 0.13 bps
PERP_FEE_BPS = 0.13


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MMConfig:
    """All configurable parameters for the MM simulation."""

    # Quote placement
    half_edge: float = 0.0001  # 1 tick improvement inside BBO
    fill_model: str = "conservative"  # "optimistic" or "conservative"

    # Sizing
    size_usd: float = 100.0  # USDC per side per quote

    # Inventory caps
    max_inventory_usd: float = 200.0  # per market
    desk_cap_usd: float = 2000.0  # desk-wide

    # Quote skew trigger
    skew_threshold_usd: float = 100.0  # start skewing at this inventory level

    # Safety gates
    min_bid_notional_usd: float = 25.0
    max_spread_bps: float = 500.0
    stale_data_halt_seconds: float = 60.0
    tte_min_seconds: float = 1800.0  # 30 min
    tte_max_seconds: float = 72000.0  # 20 h

    # Filter
    favorites_only: bool = False  # if True, only quote when mid in [0.70, 0.95]

    # Delta hedge
    hedged: bool = True
    hedge_ratio: float = DEFAULT_HEDGE_RATIO

    # 1Hz infra penalty
    apply_1hz_penalty: bool = False
    penalty_informed_frac: float = 0.20  # 20% of fills adversely selected at 1Hz
    penalty_spread_reduction: float = 0.50  # those fills get 50% of their spread

    # Performance: skip fill record accumulation (saves ~60% time; needed only for equity curves)
    track_fill_records: bool = True


@dataclass
class FillRecord:
    """One fill event in the sim."""

    ts_ns: int
    side: str  # "buy" (we bought) or "sell" (we sold)
    price: float
    tokens: float  # number of binary tokens
    cost_usd: float  # USDC spent (buy) or received (sell)
    mid_at_fill: float
    perp_px: float  # perp price at fill time (for hedge)
    hedge_qty_btc: float  # BTC sold (positive=short) if hedged
    tte_s: float


@dataclass
class MMResult:
    """Per-expiry simulation result."""

    symbol: str
    expiry_str: str
    yes_won: bool | None

    # Fill counts
    n_fills_buy: int = 0
    n_fills_sell: int = 0

    # PnL components (all in USDC)
    spread_pnl: float = 0.0  # realised from round-trips
    inventory_mtm: float = 0.0  # mark to settlement value (for open inventory)
    settlement_pnl: float = 0.0  # final settlement of residual inventory
    hedge_pnl: float = 0.0  # perp delta hedge P&L
    total_pnl: float = 0.0  # spread + settlement + hedge

    # Inventory
    net_inventory_tokens: float = 0.0  # tokens held at end
    cost_basis_usd: float = 0.0  # total USDC committed to net position

    # Hedge state
    net_hedge_btc: float = 0.0

    # Stats
    max_inventory_tokens: float = 0.0
    n_stale_halts: int = 0
    n_gate_vetoes: int = 0

    # Records (per-fill for equity curve)
    fill_records: list[FillRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_binary_bbo(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
) -> pd.DataFrame:
    """Load binary BBO for one symbol+date.

    Returns DataFrame with columns: ts_ns, bid_px, bid_sz, ask_px, ask_sz.
    """
    glob = str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / f"event=bbo/symbol={symbol}/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT
            local_recv_ts AS ts_ns,
            bid_px,
            bid_sz,
            ask_px,
            ask_sz
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "bid_px", "bid_sz", "ask_px", "ask_sz"])
    return df


def _load_binary_trades(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    symbol: str,
    date_str: str,
) -> pd.DataFrame:
    """Load binary trades for one symbol+date.

    Returns DataFrame with columns: ts_ns, price, size, side.
    """
    glob = str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / f"event=trade/symbol={symbol}/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT
            local_recv_ts AS ts_ns,
            price,
            size,
            side
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "price", "size", "side"])
    return df


def _load_perp_bbo(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    date_str: str,
) -> pd.DataFrame:
    """Load HL perp BBO for one date.

    Returns DataFrame with columns: ts_ns, bid_px, ask_px.
    """
    glob = str(
        Path(data_root)
        / "venue=hyperliquid/product_type=perp/mechanism=clob"
        / f"event=bbo/symbol=BTC/date={date_str}/hour=all/*.parquet"
    )
    sql = f"""
        SELECT
            local_recv_ts AS ts_ns,
            bid_px,
            ask_px
        FROM read_parquet('{glob}', union_by_name=true)
        ORDER BY local_recv_ts ASC
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=["ts_ns", "bid_px", "ask_px"])
    return df


def _load_binary_yes_legs(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Return binary Yes-leg metadata: symbol, expiry_str, target_price, expiry_ns.

    Same pattern as card_c_leadlag.py.
    """
    meta = str(
        Path(data_root)
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=market_meta/symbol=*/date=*/hour=all/*.parquet"
    )
    sql = f"""
        SELECT DISTINCT
            symbol,
            list_element(values, list_position(keys, 'expiry'))              AS expiry_str,
            list_element(values, list_position(keys, 'targetPrice'))::DOUBLE AS target_price
        FROM read_parquet('{meta}', union_by_name=true)
        WHERE array_contains(keys, 'class')
          AND list_element(values, list_position(keys, 'class')) = 'priceBinary'
          AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        ORDER BY expiry_str, symbol
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()
    if df.empty:
        return df

    def _parse_exp_ns(s: str) -> int:
        try:
            d = dt.datetime.strptime(s, "%Y%m%d-%H%M").replace(tzinfo=dt.UTC)
        except ValueError:
            d = dt.datetime.strptime(s, "%Y%m%d-%H").replace(tzinfo=dt.UTC)
        return int(d.timestamp() * _NS)

    df["expiry_ns"] = df["expiry_str"].map(_parse_exp_ns)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# MM sim core
# ---------------------------------------------------------------------------


def _get_perp_px(perp_ts: np.ndarray, perp_bid: np.ndarray, perp_ask: np.ndarray, ts_ns: int) -> tuple[float, float]:
    """LOCF lookup of perp bid/ask at given timestamp. Returns (bid, ask)."""
    if len(perp_ts) == 0:
        return float("nan"), float("nan")
    idx = np.searchsorted(perp_ts, ts_ns, side="right") - 1
    if idx < 0:
        idx = 0
    return float(perp_bid[idx]), float(perp_ask[idx])


def _run_mm_sim_expiry(
    bbo_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    perp_bbo_df: pd.DataFrame,
    expiry_ns: int,
    symbol: str,
    expiry_str: str,
    yes_won: bool | None,
    config: MMConfig,
    desk_inventory_usd: float = 0.0,
) -> MMResult:
    """Replay one binary expiry's BBO + trades and simulate MM quoting.

    Parameters
    ----------
    bbo_df       : binary BBO for this expiry (ts_ns, bid_px, bid_sz, ask_px, ask_sz)
    trade_df     : binary trades for this expiry (ts_ns, price, size, side)
    perp_bbo_df  : perp BBO covering the expiry window (ts_ns, bid_px, ask_px)
    expiry_ns    : expiry timestamp in nanoseconds
    symbol       : binary symbol (Yes-leg)
    expiry_str   : human-readable expiry string
    yes_won      : oracle outcome (True=Yes won, False=No won, None=unknown)
    config       : MMConfig
    desk_inventory_usd : current desk-wide inventory (USDC) from other markets
    """
    result = MMResult(symbol=symbol, expiry_str=expiry_str, yes_won=yes_won)

    if bbo_df.empty:
        return result

    # Prepare numpy arrays for fast LOCF
    perp_ts_arr = perp_bbo_df["ts_ns"].to_numpy(dtype="int64") if not perp_bbo_df.empty else np.array([], dtype="int64")
    perp_bid_arr = (
        perp_bbo_df["bid_px"].to_numpy(dtype="float64") if not perp_bbo_df.empty else np.array([], dtype="float64")
    )
    perp_ask_arr = (
        perp_bbo_df["ask_px"].to_numpy(dtype="float64") if not perp_bbo_df.empty else np.array([], dtype="float64")
    )

    # Build merged event stream: (ts_ns, event_type, row_idx)
    # event_type: 0 = BBO update, 1 = trade
    bbo_ts = bbo_df["ts_ns"].to_numpy(dtype="int64")
    trade_ts = trade_df["ts_ns"].to_numpy(dtype="int64") if not trade_df.empty else np.array([], dtype="int64")

    # Interleave events; BBO before trade on same timestamp (BBO updates first)
    events: list[tuple[int, int, int]] = []
    for i, ts in enumerate(bbo_ts):
        events.append((ts, 0, i))
    for i, ts in enumerate(trade_ts):
        events.append((ts, 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    # State
    current_bid_quote: float | None = None  # our posted bid
    current_ask_quote: float | None = None  # our posted ask
    last_bbo_ts: int = 0
    last_bid_px: float = float("nan")
    last_ask_px: float = float("nan")

    # Inventory state
    inv_tokens: float = 0.0  # net token inventory (YES tokens; positive = long)
    cost_basis_usd: float = 0.0  # USDC cost basis for net position
    tokens_bought: float = 0.0
    tokens_sold: float = 0.0
    usd_spent: float = 0.0
    usd_received: float = 0.0

    # Hedge state
    hedge_btc: float = 0.0  # net BTC short in perp (positive = short)
    hedge_cost_usd: float = 0.0  # USDC received from initial hedge fills
    hedge_perp_entry: float = 0.0  # avg entry price for perp hedge (weighted)

    max_inv_tokens: float = 0.0
    n_stale_halts: int = 0
    n_gate_vetoes: int = 0

    def _update_quotes(
        ts_ns: int,
        bid_px: float,
        bid_sz: float,
        ask_px: float,
        ask_sz: float,
    ) -> tuple[float | None, float | None, int]:
        nonlocal n_stale_halts, n_gate_vetoes
        """Recompute quotes given current BBO. Returns (new_bid, new_ask, veto_reason_code)."""

        mid = (bid_px + ask_px) / 2.0
        spread = ask_px - bid_px
        spread_bps = spread / mid * 10000.0 if mid > 0 else 9999.0
        tte_s = max(0.0, (expiry_ns - ts_ns) / _NS)

        # Safety gates
        bid_notional = bid_px * bid_sz
        if bid_notional < config.min_bid_notional_usd:
            n_gate_vetoes += 1
            return None, None, 1  # thin book
        if spread_bps > config.max_spread_bps:
            n_gate_vetoes += 1
            return None, None, 2  # spread too wide
        if tte_s < config.tte_min_seconds:
            n_gate_vetoes += 1
            return None, None, 3  # too close to expiry
        if tte_s > config.tte_max_seconds:
            n_gate_vetoes += 1
            return None, None, 4  # fresh listing

        # Favorites filter
        if config.favorites_only and not (0.70 <= mid <= 0.95):
            n_gate_vetoes += 1
            return None, None, 5

        # Inventory cap — stop quoting side that would exceed cap
        # Binary tokens cannot be shorted: can_sell requires positive inventory.
        inv_usd_equiv = abs(inv_tokens) * mid
        desk_total = desk_inventory_usd + inv_usd_equiv

        can_buy = inv_usd_equiv < config.max_inventory_usd and desk_total < config.desk_cap_usd
        can_sell = inv_tokens > 1e-8  # only sell tokens we actually hold (no shorts)

        if inv_tokens > 0 and inv_usd_equiv >= config.max_inventory_usd:
            can_buy = False

        # Compute raw quotes (join BBO)
        raw_bid = bid_px
        raw_ask = ask_px

        # Improve by half_edge on each side (inside the BBO)
        new_bid = round(raw_bid + config.half_edge, 5) if can_buy else None
        new_ask = round(raw_ask - config.half_edge, 5) if can_sell else None

        # Clamp quotes inside [0.001, 0.999]
        if new_bid is not None:
            new_bid = max(0.001, min(0.999, new_bid))
        if new_ask is not None:
            new_ask = max(0.001, min(0.999, new_ask))

        # Ensure bid < ask
        if new_bid is not None and new_ask is not None and new_bid >= new_ask:
            new_bid = new_ask - TICK

        # Quote skew: loaded side gets wider, other side narrows
        if new_bid is not None and new_ask is not None:
            inv_usd = inv_tokens * mid
            if inv_usd > config.skew_threshold_usd:
                # Long: widen bid (less eager to buy), narrow ask (more eager to sell)
                new_bid -= TICK
                new_ask -= TICK
            elif inv_usd < -config.skew_threshold_usd:
                # Short: widen ask (less eager to sell), narrow bid (more eager to buy)
                new_bid += TICK
                new_ask += TICK

        return new_bid, new_ask, 0

    # Main event loop
    for ts_ns, etype, ridx in events:
        # Check stale data halt
        if last_bbo_ts > 0 and etype == 1:  # trade event
            staleness_s = (ts_ns - last_bbo_ts) / _NS
            if staleness_s > config.stale_data_halt_seconds:
                n_stale_halts += 1
                current_bid_quote = None
                current_ask_quote = None

        if etype == 0:
            # BBO update event
            row = bbo_df.iloc[ridx]
            bid_px = float(row["bid_px"])
            ask_px = float(row["ask_px"])
            bid_sz = float(row["bid_sz"])
            ask_sz = float(row.get("ask_sz", 0.0)) if "ask_sz" in row.index else 0.0

            if not (math.isfinite(bid_px) and math.isfinite(ask_px) and bid_px > 0 and ask_px > 0):
                continue

            last_bbo_ts = ts_ns
            last_bid_px = bid_px
            last_ask_px = ask_px

            new_bid, new_ask, _ = _update_quotes(ts_ns, bid_px, bid_sz, ask_px, ask_sz)
            current_bid_quote = new_bid
            current_ask_quote = new_ask

        else:
            # Trade event
            if trade_df.empty:
                continue
            row = trade_df.iloc[ridx]
            trade_px = float(row["price"])
            trade_sz = float(row["size"])
            trade_side = str(row["side"])  # "buy" or "sell" (aggressor side)

            if not math.isfinite(trade_px) or not math.isfinite(trade_sz):
                continue

            mid_now = (
                (last_bid_px + last_ask_px) / 2.0
                if (math.isfinite(last_bid_px) and math.isfinite(last_ask_px))
                else float("nan")
            )

            # Check for fill of our passive quotes
            # A "buy" aggressor means someone is hitting our ask (we are the passive seller)
            # A "sell" aggressor means someone is hitting our bid (we are the passive buyer)

            filled_buy = False
            filled_sell = False
            fill_px = float("nan")

            if trade_side == "sell" and current_bid_quote is not None:
                # Aggressor sold — hit our passive bid
                if config.fill_model == "optimistic":
                    # filled if trade prints at price <= our bid (through or at)
                    if trade_px <= current_bid_quote:
                        filled_buy = True
                        fill_px = current_bid_quote
                else:
                    # conservative: strictly through (back of queue)
                    if trade_px < current_bid_quote:
                        filled_buy = True
                        fill_px = current_bid_quote

            elif trade_side == "buy" and current_ask_quote is not None:
                # Aggressor bought — lifted our passive ask
                if config.fill_model == "optimistic":
                    if trade_px >= current_ask_quote:
                        filled_sell = True
                        fill_px = current_ask_quote
                else:
                    if trade_px > current_ask_quote:
                        filled_sell = True
                        fill_px = current_ask_quote

            if filled_buy:
                # We bought tokens at fill_px
                tokens = config.size_usd / fill_px
                cost = tokens * fill_px  # = config.size_usd

                inv_tokens += tokens
                cost_basis_usd += cost
                tokens_bought += tokens
                usd_spent += cost

                # Hedge: buy YES binary = long BTC exposure -> sell perp
                p_bid, p_ask = _get_perp_px(perp_ts_arr, perp_bid_arr, perp_ask_arr, ts_ns)
                perp_entry_px = p_bid if math.isfinite(p_bid) else float("nan")
                hedge_qty = 0.0
                if config.hedged and math.isfinite(perp_entry_px) and perp_entry_px > 0:
                    # BTC delta per binary contract
                    btc_delta = config.hedge_ratio * (cost / perp_entry_px)
                    hedge_qty = btc_delta
                    # Sell perp: receive bid price
                    hedge_usd = hedge_qty * perp_entry_px
                    hedge_fee = hedge_usd * PERP_FEE_BPS / 10000.0
                    hedge_cost_usd += hedge_usd - hedge_fee
                    # Track weighted avg perp entry
                    if hedge_btc == 0:
                        hedge_perp_entry = perp_entry_px
                    else:
                        hedge_perp_entry = (hedge_perp_entry * hedge_btc + perp_entry_px * hedge_qty) / (
                            hedge_btc + hedge_qty
                        )
                    hedge_btc += hedge_qty

                if config.track_fill_records:
                    result.fill_records.append(
                        FillRecord(
                            ts_ns=ts_ns,
                            side="buy",
                            price=fill_px,
                            tokens=tokens,
                            cost_usd=cost,
                            mid_at_fill=mid_now,
                            perp_px=perp_entry_px,
                            hedge_qty_btc=hedge_qty,
                            tte_s=(expiry_ns - ts_ns) / _NS,
                        )
                    )
                result.n_fills_buy += 1
                max_inv_tokens = max(max_inv_tokens, abs(inv_tokens))

                # Refresh quotes after fill
                if math.isfinite(last_bid_px) and math.isfinite(last_ask_px):
                    current_bid_quote, current_ask_quote, _ = _update_quotes(ts_ns, last_bid_px, 0.0, last_ask_px, 0.0)

            elif filled_sell:
                # We sold tokens at fill_px
                # Cannot short binary tokens: cap sell qty at what we actually hold.
                target_tokens = config.size_usd / fill_px
                tokens = min(target_tokens, inv_tokens)
                if tokens <= 1e-8:
                    # No inventory left to sell — skip
                    current_bid_quote, current_ask_quote = None, None  # pull quotes
                    continue
                proceeds = tokens * fill_px

                inv_tokens -= tokens
                cost_basis_usd -= tokens * (cost_basis_usd / (tokens_bought + 1e-12) if tokens_bought > 0 else fill_px)
                tokens_sold += tokens
                usd_received += proceeds

                # Hedge: sell YES binary = short BTC exposure -> buy perp
                p_bid, p_ask = _get_perp_px(perp_ts_arr, perp_bid_arr, perp_ask_arr, ts_ns)
                perp_entry_px = p_ask if math.isfinite(p_ask) else float("nan")
                hedge_qty = 0.0
                if config.hedged and math.isfinite(perp_entry_px) and perp_entry_px > 0:
                    btc_delta = config.hedge_ratio * (proceeds / perp_entry_px)
                    hedge_qty = -btc_delta  # negative = long perp (closing short)
                    hedge_usd = btc_delta * perp_entry_px
                    hedge_fee = hedge_usd * PERP_FEE_BPS / 10000.0
                    hedge_cost_usd -= hedge_usd + hedge_fee
                    if hedge_btc == 0:
                        hedge_perp_entry = perp_entry_px
                    else:
                        total = hedge_btc + hedge_qty
                        if abs(total) > 1e-12:
                            hedge_perp_entry = (hedge_perp_entry * hedge_btc + perp_entry_px * hedge_qty) / total
                    hedge_btc += hedge_qty

                if config.track_fill_records:
                    result.fill_records.append(
                        FillRecord(
                            ts_ns=ts_ns,
                            side="sell",
                            price=fill_px,
                            tokens=tokens,
                            cost_usd=-proceeds,
                            mid_at_fill=mid_now,
                            perp_px=perp_entry_px,
                            hedge_qty_btc=hedge_qty,
                            tte_s=(expiry_ns - ts_ns) / _NS,
                        )
                    )
                result.n_fills_sell += 1
                max_inv_tokens = max(max_inv_tokens, abs(inv_tokens))

                # Refresh quotes
                if math.isfinite(last_bid_px) and math.isfinite(last_ask_px):
                    current_bid_quote, current_ask_quote, _ = _update_quotes(ts_ns, last_bid_px, 0.0, last_ask_px, 0.0)

    # -----------------------------------------------------------------------
    # Settlement PnL
    # -----------------------------------------------------------------------
    # Determine settlement value
    if yes_won is True:
        settlement_px = 1.0
    elif yes_won is False:
        settlement_px = 0.0
    else:
        # No oracle data — mark at mid
        settlement_px = (
            (last_bid_px + last_ask_px) / 2.0 if (math.isfinite(last_bid_px) and math.isfinite(last_ask_px)) else 0.5
        )

    # Net inventory at end (must be >= 0 with no-short constraint)
    net_tokens = max(0.0, inv_tokens)  # guard against floating-point dust going negative

    # -----------------------------------------------------------------------
    # PnL accounting (cash-basis):
    #   total_pnl = usd_received - usd_spent + settlement_value_of_net_tokens
    #
    # Decomposition into spread_pnl + settlement_pnl:
    #   spread_pnl:   PnL from matched round-trips (buy low, sell high)
    #   settlement_pnl: PnL from residual long inventory settled at oracle price
    # -----------------------------------------------------------------------
    matched_tokens = min(tokens_bought, tokens_sold)
    if matched_tokens > 0 and tokens_bought > 0 and tokens_sold > 0:
        avg_buy_px = usd_spent / tokens_bought
        avg_sell_px = usd_received / tokens_sold
        spread_pnl = (avg_sell_px - avg_buy_px) * matched_tokens
    else:
        spread_pnl = 0.0

    # Settlement PnL: residual long inventory settled at oracle price.
    # residual_tokens = tokens bought but not yet sold = net_tokens
    if net_tokens > 1e-8 and tokens_bought > 0:
        avg_buy_px = usd_spent / tokens_bought
        settlement_pnl = (settlement_px - avg_buy_px) * net_tokens
    else:
        settlement_pnl = 0.0

    # MTM at end (before settlement) — informational only
    mid_final = (
        (last_bid_px + last_ask_px) / 2.0
        if (math.isfinite(last_bid_px) and math.isfinite(last_ask_px))
        else settlement_px
    )
    inventory_mtm = (mid_final - (usd_spent / tokens_bought if tokens_bought > 0 else mid_final)) * net_tokens

    # Hedge PnL at settlement
    hedge_pnl = 0.0
    if config.hedged and hedge_btc != 0 and math.isfinite(hedge_perp_entry):
        # Get perp close price at settlement
        p_bid_close, p_ask_close = _get_perp_px(perp_ts_arr, perp_bid_arr, perp_ask_arr, expiry_ns)
        if math.isfinite(p_bid_close):
            perp_close_px = (p_bid_close + p_ask_close) / 2.0
            # We had a net short (hedge_btc > 0 means we sold perp)
            # PnL = (entry - close) * qty for short positions
            hedge_pnl = (hedge_perp_entry - perp_close_px) * hedge_btc
            # Add hedge fees already accounted in hedge_cost_usd above

    # Apply 1Hz penalty on conservative fills
    penalty_pnl = 0.0
    if config.apply_1hz_penalty and config.fill_model == "conservative":
        n_total_fills = result.n_fills_buy + result.n_fills_sell
        n_informed = int(n_total_fills * config.penalty_informed_frac)
        # Each informed fill loses 50% of its spread contribution
        avg_spread_per_fill = spread_pnl / n_total_fills if n_total_fills > 0 else 0.0
        penalty_pnl = -n_informed * avg_spread_per_fill * config.penalty_spread_reduction

    total_pnl = spread_pnl + settlement_pnl + hedge_pnl + penalty_pnl

    result.spread_pnl = spread_pnl
    result.settlement_pnl = settlement_pnl
    result.inventory_mtm = inventory_mtm
    result.hedge_pnl = hedge_pnl
    result.total_pnl = total_pnl
    result.net_inventory_tokens = net_tokens
    result.cost_basis_usd = cost_basis_usd
    result.net_hedge_btc = hedge_btc
    result.max_inventory_tokens = max_inv_tokens
    result.n_stale_halts = n_stale_halts
    result.n_gate_vetoes = n_gate_vetoes

    return result


# ---------------------------------------------------------------------------
# Run all expiries
# ---------------------------------------------------------------------------


def _dates_for_expiry(expiry_ns: int, corpus_start: str, corpus_end: str) -> list[str]:
    """Return the 1-2 dates to load data for a given expiry."""
    exp_dt = dt.datetime.fromtimestamp(expiry_ns / _NS, tz=dt.UTC)
    exp_d = exp_dt.date()
    prev_d = exp_d - dt.timedelta(days=1)
    start_d = dt.date.fromisoformat(corpus_start)
    end_d = dt.date.fromisoformat(corpus_end)
    dates = []
    for d in [prev_d, exp_d]:
        if start_d <= d <= end_d:
            dates.append(d.isoformat())
    if not dates:
        dates = [prev_d.isoformat()]
    return dates


def _run_mm_all_expiries(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    expiry_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    config: MMConfig,
    start_date: str = CORPUS_START,
    end_date: str = CORPUS_END,
) -> list[MMResult]:
    """Run MM sim across all expiries in [start_date, end_date].

    Parameters
    ----------
    expiry_df    : DataFrame from _load_binary_yes_legs
    outcomes_df  : DataFrame from resolve_binary_outcomes
    """
    start_ns = int(dt.datetime(*[int(x) for x in start_date.split("-")], tzinfo=dt.UTC).timestamp() * _NS)
    end_ns = int(dt.datetime(*[int(x) for x in end_date.split("-")], tzinfo=dt.UTC).timestamp() * _NS) + 86400 * _NS

    active = expiry_df[(expiry_df["expiry_ns"] >= start_ns) & (expiry_df["expiry_ns"] <= end_ns)].copy()

    _log.info("run_mm_all_expiries: %d expiries in %s..%s", len(active), start_date, end_date)

    # Build outcomes lookup
    outcomes_map: dict[str, bool | None] = {}
    if not outcomes_df.empty and "symbol" in outcomes_df.columns:
        for _, row in outcomes_df.iterrows():
            outcomes_map[str(row["symbol"])] = bool(row["yes_won"]) if row["yes_won"] is not None else None

    results: list[MMResult] = []
    desk_inventory_usd = 0.0  # cumulative desk inventory

    for _, row in active.iterrows():
        sym = str(row["symbol"])
        expiry_ns = int(row["expiry_ns"])
        expiry_str = str(row["expiry_str"])
        yes_won = outcomes_map.get(sym)

        dates = _dates_for_expiry(expiry_ns, start_date, end_date)

        bbo_frames = []
        trade_frames = []
        perp_frames = []

        for date_str in dates:
            bf = _load_binary_bbo(con, data_root, sym, date_str)
            if not bf.empty:
                bbo_frames.append(bf)
            tf = _load_binary_trades(con, data_root, sym, date_str)
            if not tf.empty:
                trade_frames.append(tf)
            pf = _load_perp_bbo(con, data_root, date_str)
            if not pf.empty:
                perp_frames.append(pf)

        if not bbo_frames:
            continue

        bbo_all = pd.concat(bbo_frames, ignore_index=True).sort_values("ts_ns")
        trade_all = (
            pd.concat(trade_frames, ignore_index=True).sort_values("ts_ns")
            if trade_frames
            else pd.DataFrame(columns=["ts_ns", "price", "size", "side"])
        )
        perp_all = (
            pd.concat(perp_frames, ignore_index=True).sort_values("ts_ns")
            if perp_frames
            else pd.DataFrame(columns=["ts_ns", "bid_px", "ask_px"])
        )

        res = _run_mm_sim_expiry(
            bbo_df=bbo_all,
            trade_df=trade_all,
            perp_bbo_df=perp_all,
            expiry_ns=expiry_ns,
            symbol=sym,
            expiry_str=expiry_str,
            yes_won=yes_won,
            config=config,
            desk_inventory_usd=desk_inventory_usd,
        )
        results.append(res)

        # Update desk inventory approximation
        desk_inventory_usd += abs(res.net_inventory_tokens) * 0.5  # rough 0.5 price

    _log.info(
        "run_mm_all_expiries: %d results, total_pnl=%.2f",
        len(results),
        sum(r.total_pnl for r in results),
    )
    return results


# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------


def _summarise_results(results: list[MMResult]) -> dict[str, Any]:
    """Compute aggregate KPIs from a list of MMResults."""
    if not results:
        return {}

    total_pnl = sum(r.total_pnl for r in results)
    spread_pnl = sum(r.spread_pnl for r in results)
    settlement_pnl = sum(r.settlement_pnl for r in results)
    hedge_pnl = sum(r.hedge_pnl for r in results)

    n_fills = sum(r.n_fills_buy + r.n_fills_sell for r in results)
    n_fills_buy = sum(r.n_fills_buy for r in results)
    n_fills_sell = sum(r.n_fills_sell for r in results)
    n_expiries = len(results)
    n_with_fills = sum(1 for r in results if r.n_fills_buy + r.n_fills_sell > 0)

    # Per-expiry PnL for Sharpe / drawdown
    pnl_series = np.array([r.total_pnl for r in results], dtype="float64")
    cumulative = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = peak - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    pnl_std = float(np.std(pnl_series)) if len(pnl_series) > 1 else 0.0
    pnl_mean = float(np.mean(pnl_series)) if len(pnl_series) > 0 else 0.0
    if pnl_std > 0:
        sharpe = pnl_mean / pnl_std * math.sqrt(n_expiries)
    elif pnl_mean > 0:
        sharpe = float("inf")
    elif pnl_mean < 0:
        sharpe = float("-inf")
    else:
        sharpe = 0.0

    hit_rate = float(np.mean(pnl_series > 0)) if len(pnl_series) > 0 else 0.0

    n_stale = sum(r.n_stale_halts for r in results)
    n_vetoes = sum(r.n_gate_vetoes for r in results)

    return {
        "n_expiries": n_expiries,
        "n_with_fills": n_with_fills,
        "n_fills": n_fills,
        "n_fills_buy": n_fills_buy,
        "n_fills_sell": n_fills_sell,
        "total_pnl": total_pnl,
        "spread_pnl": spread_pnl,
        "settlement_pnl": settlement_pnl,
        "hedge_pnl": hedge_pnl,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "pnl_per_expiry_mean": float(np.mean(pnl_series)),
        "pnl_per_expiry_std": pnl_std,
        "n_stale_halts": n_stale,
        "n_gate_vetoes": n_vetoes,
    }


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------


def _sweep_configs(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
    expiry_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    base_config: MMConfig,
    start_date: str = IS_START,
    end_date: str = IS_END,
) -> list[dict[str, Any]]:
    """Sweep key config parameters. Returns list of summary dicts.

    Runs over the first half of IS (14 expiries) for speed; main results use full IS.
    Focuses on the most decision-relevant dimensions: fill model, quote width, size, hedge.
    Fill model / hedged / favorites / 1Hz results are also computed separately in build_card.
    """
    sweep_results: list[dict[str, Any]] = []

    # Use first half of IS to keep runtime manageable
    sweep_end = SPLIT_H1_END  # 2026-05-23 (14 expiries instead of 28)

    def _run(label: str, cfg: MMConfig) -> dict[str, Any]:
        res = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, cfg, start_date, sweep_end)
        summary = _summarise_results(res)
        summary["label"] = label
        summary["fill_model"] = cfg.fill_model
        summary["half_edge"] = cfg.half_edge
        summary["size_usd"] = cfg.size_usd
        summary["hedged"] = cfg.hedged
        summary["favorites_only"] = cfg.favorites_only
        summary["apply_1hz_penalty"] = cfg.apply_1hz_penalty
        return summary

    import copy

    # 1. Fill model: optimistic vs conservative (most important comparison)
    for fm in ["optimistic", "conservative"]:
        cfg = copy.copy(base_config)
        cfg.fill_model = fm
        sweep_results.append(_run(f"fill={fm}", cfg))

    # 2. Quote width: half_edge sweep (key tuning dimension)
    for he in [0.0, 0.0001, 0.0005]:
        cfg = copy.copy(base_config)
        cfg.half_edge = he
        sweep_results.append(_run(f"half_edge={he:.4f}", cfg))

    # 3. Size sweep (capacity scaling)
    for sz in [50.0, 100.0, 200.0]:
        cfg = copy.copy(base_config)
        cfg.size_usd = sz
        sweep_results.append(_run(f"size=${sz:.0f}", cfg))

    # 4. Hedged vs unhedged
    for hedged in [True, False]:
        cfg = copy.copy(base_config)
        cfg.hedged = hedged
        sweep_results.append(_run(f"hedged={hedged}", cfg))

    return sweep_results


# ---------------------------------------------------------------------------
# Capacity analysis
# ---------------------------------------------------------------------------


def _capacity_table() -> list[dict[str, Any]]:
    """Estimate capacity based on Card A depth findings.

    Card A:   TOB notional ~$107, within 100bps ~$679 per binary market.
    Card B:   maker realized spread ~0.95pp per round-trip.
    ~24 active markets live simultaneously (daily 06:00 UTC expiries, 1 per day).
    Desk levels: $1k = $42/market, $5k = $208/market, $25k = $1042/market.
    """
    tob_notional = 107.0
    depth_100bps = 679.0
    n_markets = 24  # typical simultaneous markets

    rows = []
    for size_usd in [25.0, 50.0, 100.0, 200.0]:
        fill_frac_tob = size_usd / tob_notional
        fill_frac_100bps = size_usd / depth_100bps
        # At typical 128bps binary spread, maker earns ~0.95pp per round-trip (Card B)
        expected_spread_per_rt = 0.0095 * size_usd  # 0.95% of size
        desk_capacity = size_usd * n_markets
        rows.append(
            {
                "size_usd": size_usd,
                "desk_capacity_usd": desk_capacity,
                "fill_frac_tob": fill_frac_tob,
                "fill_frac_100bps": fill_frac_100bps,
                "expected_spread_per_roundtrip_usd": expected_spread_per_rt,
                "market_impact_note": (
                    "minimal" if fill_frac_tob < 0.5 else "moderate" if fill_frac_tob < 1.0 else "high"
                ),
            }
        )

    # Add desk-level rows: $1k, $5k, $25k
    for desk_usd in [1000.0, 5000.0, 25000.0]:
        per_market = desk_usd / n_markets
        fill_frac_tob = per_market / tob_notional
        fill_frac_100bps = per_market / depth_100bps
        expected_spread_per_rt = 0.0095 * per_market
        rows.append(
            {
                "size_usd": per_market,
                "desk_capacity_usd": desk_usd,
                "fill_frac_tob": fill_frac_tob,
                "fill_frac_100bps": fill_frac_100bps,
                "expected_spread_per_roundtrip_usd": expected_spread_per_rt,
                "market_impact_note": (
                    "minimal" if fill_frac_tob < 0.5 else "moderate" if fill_frac_tob < 1.0 else "high"
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _dark_axes(ax: plt.Axes) -> None:
    """Apply dark GitHub theme to a single axes."""
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#e6edf3")
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#58a6ff")


def _dark_fig(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, Any]:
    """Create a dark-theme figure."""
    if figsize is None:
        figsize = (14, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    if nrows == 1 and ncols == 1:
        _dark_axes(axes)
    else:
        ax_flat = np.array(axes).ravel()
        for ax in ax_flat:
            _dark_axes(ax)
    return fig, axes


def _plot_equity_curves(
    is_results: list[MMResult],
    oos_results: list[MMResult],
    is_results_opt: list[MMResult],
    is_results_unhedged: list[MMResult],
) -> matplotlib.figure.Figure:
    """Four equity curves: IS/OOS, optimistic/conservative, hedged/unhedged."""
    fig, axes = _dark_fig(2, 2, figsize=(16, 10))
    fig.suptitle("MM Strategy — Equity Curves", color="#e6edf3", fontsize=13)

    def _cum_pnl(results: list[MMResult]) -> np.ndarray:
        return np.cumsum([r.total_pnl for r in results])

    def _plot_curve(ax: plt.Axes, results: list[MMResult], label: str, color: str) -> None:
        if not results:
            return
        y = _cum_pnl(results)
        ax.plot(y, label=label, color=color, linewidth=1.5)
        ax.axhline(0, color="#30363d", linewidth=0.8)

    # IS vs OOS
    ax = axes[0][0]
    _plot_curve(ax, is_results, f"IS ({IS_START}..{IS_END})", "#58a6ff")
    _plot_curve(ax, oos_results, f"OOS ({OOS_START}..{OOS_END})", "#3fb950")
    ax.set_xlabel("Expiry #")
    ax.set_ylabel("Cumulative PnL (USDC)")
    ax.set_title("IS vs OOS")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    # Optimistic vs conservative (IS)
    ax = axes[0][1]
    _plot_curve(ax, is_results_opt, "Optimistic fill", "#f78166")
    _plot_curve(ax, is_results, "Conservative fill", "#58a6ff")
    ax.set_xlabel("Expiry #")
    ax.set_ylabel("Cumulative PnL (USDC)")
    ax.set_title("Fill Model: Optimistic vs Conservative (IS)")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    # Hedged vs unhedged (IS)
    ax = axes[1][0]
    _plot_curve(ax, is_results, "Hedged", "#58a6ff")
    _plot_curve(ax, is_results_unhedged, "Unhedged", "#f78166")
    ax.set_xlabel("Expiry #")
    ax.set_ylabel("Cumulative PnL (USDC)")
    ax.set_title("Hedged vs Unhedged (IS)")
    ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    # IS PnL per expiry histogram
    ax = axes[1][1]
    if is_results:
        pnl_arr = np.array([r.total_pnl for r in is_results])
        ax.hist(pnl_arr, bins=30, color="#58a6ff", alpha=0.8, edgecolor="#30363d")
        ax.axvline(0, color="#f78166", linewidth=1.2, linestyle="--")
        ax.axvline(float(np.mean(pnl_arr)), color="#3fb950", linewidth=1.2, label=f"Mean={np.mean(pnl_arr):.2f}")
        ax.set_xlabel("PnL per expiry (USDC)")
        ax.set_ylabel("Count")
        ax.set_title("PnL Distribution per Expiry (IS)")
        ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

    plt.tight_layout()
    return fig


def _plot_pnl_attribution(is_results: list[MMResult]) -> matplotlib.figure.Figure:
    """PnL attribution bar chart: spread_capture, settlement, hedge."""
    fig, ax = _dark_fig(1, 1, figsize=(10, 5))
    fig.suptitle("PnL Attribution (IS)", color="#e6edf3", fontsize=13)

    if not is_results:
        return fig

    labels = ["Spread Capture", "Settlement PnL", "Hedge PnL", "Total"]
    values = [
        sum(r.spread_pnl for r in is_results),
        sum(r.settlement_pnl for r in is_results),
        sum(r.hedge_pnl for r in is_results),
        sum(r.total_pnl for r in is_results),
    ]
    colors = ["#3fb950", "#f78166", "#58a6ff", "#e6edf3"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="#30363d")
    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (abs(y) * 0.02 + 0.1),
            f"${val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#e6edf3",
        )
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.set_ylabel("USDC")
    ax.set_title("PnL Attribution")

    plt.tight_layout()
    return fig


def _plot_inventory_distribution(is_results: list[MMResult]) -> matplotlib.figure.Figure:
    """Inventory distribution histogram."""
    fig, axes = _dark_fig(1, 2, figsize=(14, 5))
    fig.suptitle("Inventory Analysis (IS)", color="#e6edf3", fontsize=13)

    if is_results:
        ax0 = axes[0]
        max_invs = [r.max_inventory_tokens for r in is_results]
        ax0.hist(max_invs, bins=25, color="#f78166", alpha=0.8, edgecolor="#30363d")
        ax0.axvline(
            float(np.median(max_invs)), color="#58a6ff", linewidth=1.5, label=f"Median={np.median(max_invs):.2f}"
        )
        ax0.set_xlabel("Max Inventory Tokens per Expiry")
        ax0.set_ylabel("Count")
        ax0.set_title("Max Inventory Distribution")
        ax0.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22")

        ax1 = axes[1]
        net_invs = [r.net_inventory_tokens for r in is_results]
        ax1.hist(net_invs, bins=25, color="#58a6ff", alpha=0.8, edgecolor="#30363d")
        ax1.axvline(0, color="#f78166", linewidth=1.2, linestyle="--")
        ax1.set_xlabel("Net Inventory at Expiry (tokens)")
        ax1.set_ylabel("Count")
        ax1.set_title("Net Inventory at Settlement")

    plt.tight_layout()
    return fig


def _plot_sensitivity_heatmap(sweep_results: list[dict[str, Any]]) -> matplotlib.figure.Figure:
    """Sensitivity heatmap: total PnL and Sharpe across parameter variants."""
    if not sweep_results:
        fig, ax = _dark_fig(1, 1)
        return fig

    labels = [r["label"] for r in sweep_results]
    total_pnls = [r.get("total_pnl", 0.0) for r in sweep_results]
    sharpes = [r.get("sharpe", float("nan")) for r in sweep_results]
    hit_rates = [r.get("hit_rate", 0.0) for r in sweep_results]

    fig, axes = _dark_fig(1, 3, figsize=(18, 6))
    fig.suptitle("Sensitivity Sweep (IS)", color="#e6edf3", fontsize=13)

    def _bar(ax: plt.Axes, vals: list[float], title: str, fmt: str = ".2f") -> None:
        colors_list = ["#3fb950" if v >= 0 else "#f78166" for v in vals]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color=colors_list, alpha=0.8, edgecolor="#30363d")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#30363d", linewidth=0.8)
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            if math.isfinite(val):
                ax.text(
                    val + (max(vals, default=0) * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    format(val, fmt),
                    va="center",
                    fontsize=7,
                    color="#e6edf3",
                )

    _bar(axes[0], total_pnls, "Total PnL (USDC)", ".2f")
    axes[0].set_xlabel("Total PnL (USDC)")

    sharpes_clean = [s if math.isfinite(s) else 0.0 for s in sharpes]
    _bar(axes[1], sharpes_clean, "Sharpe (per-expiry)", ".2f")
    axes[1].set_xlabel("Sharpe")

    _bar(axes[2], hit_rates, "Hit Rate", ".1%")
    axes[2].set_xlabel("Hit Rate")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "N/A"
    return format(val, fmt)


def _metrics_table_html(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for m in metrics:
        rows.append(
            f"<tr>"
            f"<td>{m['name']}</td>"
            f"<td><strong>{m['value']}</strong></td>"
            f"<td>{m.get('n', '')}</td>"
            f"<td>{m.get('date_span', '')}</td>"
            f"<td>{m.get('sanity', '')}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Value</th><th>n</th><th>Date span</th><th>Sanity</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _sweep_table_html(sweep: list[dict[str, Any]]) -> str:
    rows = []
    for r in sweep:
        pnl = r.get("total_pnl", 0.0)
        pnl_str = f"${pnl:+.2f}"
        sharpe = r.get("sharpe", float("nan"))
        sharpe_str = _fmt(sharpe, ".2f")
        hit = r.get("hit_rate", 0.0)
        rows.append(
            f"<tr>"
            f"<td>{r.get('label', '')}</td>"
            f"<td>{r.get('fill_model', '')}</td>"
            f"<td>{r.get('half_edge', '')}</td>"
            f"<td>{r.get('size_usd', '')}</td>"
            f"<td>{r.get('hedged', '')}</td>"
            f"<td><strong>{pnl_str}</strong></td>"
            f"<td>{sharpe_str}</td>"
            f"<td>{hit:.1%}</td>"
            f"<td>{r.get('n_fills', 0)}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>Label</th><th>Fill Model</th><th>Half Edge</th><th>Size $</th><th>Hedged</th>"
        "<th>Total PnL</th><th>Sharpe</th><th>Hit Rate</th><th>n Fills</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _capacity_table_html(rows: list[dict[str, Any]]) -> str:
    html_rows = []
    for r in rows:
        html_rows.append(
            f"<tr>"
            f"<td>${r['size_usd']:.0f}</td>"
            f"<td>${r.get('desk_capacity_usd', r['size_usd'] * 24):.0f}</td>"
            f"<td>{r['fill_frac_tob']:.1%}</td>"
            f"<td>{r['fill_frac_100bps']:.1%}</td>"
            f"<td>${r['expected_spread_per_roundtrip_usd']:.2f}</td>"
            f"<td>{r['market_impact_note']}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>Size/Market (USDC)</th><th>Desk Total (~24 markets)</th>"
        "<th>TOB Fraction ($107)</th><th>Depth Fraction ($679)</th>"
        "<th>Expected Spread/RT</th><th>Market Impact</th>"
        "</tr></thead>"
        "<tbody>" + "".join(html_rows) + "</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# KPI checks
# ---------------------------------------------------------------------------


def _check_kpis(
    is_summary: dict[str, Any],
    oos_summary: dict[str, Any],
    is_opt_summary: dict[str, Any] | None = None,
    oos_opt_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate pass/fail for key KPIs (both fill models if opt summaries provided)."""
    kpis = []

    def _kpi(name: str, value: Any, passed: bool, target: str, note: str = "") -> dict[str, Any]:
        return {
            "name": name,
            "value": str(value),
            "passed": passed,
            "target": target,
            "note": note,
            "status": "PASS" if passed else "FAIL",
        }

    is_pnl = is_summary.get("total_pnl", 0.0)
    oos_pnl = oos_summary.get("total_pnl", 0.0)
    is_sharpe = is_summary.get("sharpe", float("nan"))
    oos_sharpe = oos_summary.get("sharpe", float("nan"))
    is_dd = is_summary.get("max_drawdown", 0.0)
    is_fills = is_summary.get("n_fills", 0)
    is_hit = is_summary.get("hit_rate", 0.0)
    is_n = is_summary.get("n_expiries", 0)
    oos_n = oos_summary.get("n_expiries", 0)

    is_underpowered = is_n < 15
    oos_underpowered = oos_n < 15

    # --- Conservative model (primary / honest bar) ---
    kpis.append(
        _kpi(
            "[CONSERVATIVE] IS total PnL > 0",
            f"${is_pnl:.2f} (n={is_n})",
            is_pnl > 0,
            ">$0",
            ("Spread capture must be positive" + (" [UNDERPOWERED n<15]" if is_underpowered else "")),
        )
    )
    kpis.append(
        _kpi(
            "[CONSERVATIVE] OOS total PnL > 0",
            f"${oos_pnl:.2f} (n={oos_n})",
            oos_pnl > 0,
            ">$0",
            ("Out-of-sample positive" + (" [UNDERPOWERED n<15]" if oos_underpowered else "")),
        )
    )
    kpis.append(
        _kpi(
            "[CONSERVATIVE] IS Sharpe > 1.0",
            _fmt(is_sharpe, ".2f"),
            math.isfinite(is_sharpe) and is_sharpe > 1.0,
            ">1.0",
            "Risk-adjusted return (IS)",
        )
    )
    kpis.append(
        _kpi(
            "[CONSERVATIVE] OOS Sharpe > 1.0",
            _fmt(oos_sharpe, ".2f"),
            math.isfinite(oos_sharpe) and oos_sharpe > 1.0,
            ">1.0",
            "OOS Sharpe — flags n_oos<15 as underpowered" + (" [UNDERPOWERED]" if oos_underpowered else ""),
        )
    )
    kpis.append(_kpi("Max Drawdown < $500", f"${is_dd:.2f}", is_dd < 500.0, "<$500", "Desk-scale drawdown gate"))
    kpis.append(
        _kpi(
            "n_fills > 100 (IS)",
            str(is_fills),
            is_fills > 100,
            ">100",
            "Sufficient fill activity for statistical validity",
        )
    )

    # --- Optimistic model KPIs (upper-bound comparison) ---
    if is_opt_summary is not None:
        opt_is_pnl = is_opt_summary.get("total_pnl", 0.0)
        opt_is_n = is_opt_summary.get("n_expiries", 0)
        kpis.append(
            _kpi(
                "[OPTIMISTIC] IS total PnL > 0",
                f"${opt_is_pnl:.2f} (n={opt_is_n})",
                opt_is_pnl > 0,
                ">$0",
                "Optimistic (at-quote) fill upper bound",
            )
        )
    if oos_opt_summary is not None:
        opt_oos_pnl = oos_opt_summary.get("total_pnl", 0.0)
        opt_oos_n = oos_opt_summary.get("n_expiries", 0)
        kpis.append(
            _kpi(
                "[OPTIMISTIC] OOS total PnL > 0",
                f"${opt_oos_pnl:.2f} (n={opt_oos_n})",
                opt_oos_pnl > 0,
                ">$0",
                "Optimistic OOS" + (" [UNDERPOWERED n<15]" if opt_oos_n < 15 else ""),
            )
        )

    # --- Common KPIs (sign stability, fills) ---
    kpis.append(
        _kpi(
            "IS-OOS PnL sign agreement",
            f"IS={is_pnl:.2f}, OOS={oos_pnl:.2f}",
            (is_pnl > 0) == (oos_pnl > 0),
            "same sign",
            "Robustness check",
        )
    )
    kpis.append(_kpi("Hit rate > 50%", f"{is_hit:.1%}", is_hit > 0.50, ">50%", "More profitable expiries than losing"))

    return kpis


def _kpi_table_html(kpis: list[dict[str, Any]]) -> str:
    rows = []
    for k in kpis:
        color = "#3fb950" if k["passed"] else "#f78166"
        rows.append(
            f"<tr>"
            f"<td>{k['name']}</td>"
            f"<td><strong>{k['value']}</strong></td>"
            f"<td>{k['target']}</td>"
            f"<td style='color:{color}'><strong>{k['status']}</strong></td>"
            f"<td>{k.get('note', '')}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>KPI</th><th>Value</th><th>Target</th><th>Status</th><th>Note</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# build_card — main entry point
# ---------------------------------------------------------------------------


def build_card(
    data_root: str,
    *,
    out_dir: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build Strategy MM card: market-making simulation on HL binary markets.

    Parameters
    ----------
    data_root : Path to data root (HLBT_HL_DATA_ROOT)
    out_dir   : Optional output directory for HTML/JSON files

    Returns
    -------
    (card_html, findings) where findings is JSON-serialisable.
    """
    data_root = str(Path(data_root).resolve())
    con = duckdb.connect()

    _log.info("Strategy MM: loading binary yes-legs from %s", data_root)
    expiry_df = _load_binary_yes_legs(con, data_root)
    _log.info("Strategy MM: %d binary expiries found", len(expiry_df))

    # Load oracle outcomes (needed for settlement PnL)
    _log.info("Strategy MM: resolving binary outcomes")
    try:
        outcomes_df = resolve_binary_outcomes(con, data_root)
    except Exception as exc:
        _log.warning("resolve_binary_outcomes failed: %s — continuing without settlement labels", exc)
        outcomes_df = pd.DataFrame()

    # Base config (conservative, hedged, 1-tick edge, $100 size)
    # fill records not consumed by any plot/summary — skip for speed
    base_config = MMConfig(
        fill_model="conservative",
        half_edge=0.0001,
        size_usd=100.0,
        hedged=True,
        track_fill_records=False,
    )

    # IS run (conservative, hedged)
    _log.info("Strategy MM: running IS sim (%s..%s)", IS_START, IS_END)
    is_results = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, base_config, IS_START, IS_END)

    # OOS run
    _log.info("Strategy MM: running OOS sim (%s..%s)", OOS_START, OOS_END)
    oos_results = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, base_config, OOS_START, OOS_END)

    # Optimistic fill (for comparison)
    import copy

    opt_config = copy.copy(base_config)
    opt_config.fill_model = "optimistic"
    opt_config.track_fill_records = False
    _log.info("Strategy MM: running optimistic IS sim")
    is_results_opt = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, opt_config, IS_START, IS_END)

    # Unhedged (for comparison)
    unhedged_config = copy.copy(base_config)
    unhedged_config.hedged = False
    unhedged_config.track_fill_records = False
    _log.info("Strategy MM: running unhedged IS sim")
    is_results_unhedged = _run_mm_all_expiries(
        con, data_root, expiry_df, outcomes_df, unhedged_config, IS_START, IS_END
    )

    # Optimistic OOS (for KPI table)
    _log.info("Strategy MM: running optimistic OOS sim")
    oos_results_opt = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, opt_config, OOS_START, OOS_END)

    # Favorites-only (IS and OOS)
    fav_config = copy.copy(base_config)
    fav_config.favorites_only = True
    fav_config.track_fill_records = False
    _log.info("Strategy MM: running favorites-only IS sim")
    is_results_fav = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, fav_config, IS_START, IS_END)
    _log.info("Strategy MM: running favorites-only OOS sim")
    oos_results_fav = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, fav_config, OOS_START, OOS_END)

    # 1Hz penalty sim (conservative, to model live infra adverse selection)
    hz1_config = copy.copy(base_config)
    hz1_config.apply_1hz_penalty = True
    hz1_config.track_fill_records = False
    _log.info("Strategy MM: running 1Hz-penalty IS sim")
    is_results_1hz = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, hz1_config, IS_START, IS_END)

    # Summarise
    is_summary = _summarise_results(is_results)
    oos_summary = _summarise_results(oos_results)
    is_opt_summary = _summarise_results(is_results_opt)
    oos_opt_summary = _summarise_results(oos_results_opt)
    is_unhedged_summary = _summarise_results(is_results_unhedged)
    is_fav_summary = _summarise_results(is_results_fav)
    oos_fav_summary = _summarise_results(oos_results_fav)
    is_1hz_summary = _summarise_results(is_results_1hz)

    _log.info(
        "IS: total_pnl=%.2f, n_fills=%d, sharpe=%.2f, max_dd=%.2f",
        is_summary.get("total_pnl", 0),
        is_summary.get("n_fills", 0),
        is_summary.get("sharpe", float("nan")),
        is_summary.get("max_drawdown", 0),
    )
    _log.info(
        "OOS: total_pnl=%.2f, n_fills=%d, sharpe=%.2f",
        oos_summary.get("total_pnl", 0),
        oos_summary.get("n_fills", 0),
        oos_summary.get("sharpe", float("nan")),
    )
    _log.info(
        "1Hz penalty IS: total_pnl=%.2f vs no-penalty %.2f",
        is_1hz_summary.get("total_pnl", 0),
        is_summary.get("total_pnl", 0),
    )

    # Sensitivity sweep (IS only, to avoid OOS snooping)
    _log.info("Strategy MM: running sensitivity sweep")
    sweep = _sweep_configs(con, data_root, expiry_df, outcomes_df, base_config, IS_START, IS_END)

    # Split-half
    h1_results = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, base_config, SPLIT_H1_START, SPLIT_H1_END)
    h2_results = _run_mm_all_expiries(con, data_root, expiry_df, outcomes_df, base_config, SPLIT_H2_START, SPLIT_H2_END)
    h1_summary = _summarise_results(h1_results)
    h2_summary = _summarise_results(h2_results)

    # Live infra verdict (1Hz penalty analysis)
    pnl_no_penalty = is_summary.get("total_pnl", 0.0)
    pnl_1hz = is_1hz_summary.get("total_pnl", 0.0)
    pnl_lost_1hz = pnl_no_penalty - pnl_1hz
    pct_lost = pnl_lost_1hz / pnl_no_penalty * 100 if pnl_no_penalty != 0 else 0.0
    infra_blocked = pnl_1hz <= 0  # if 1Hz kills profitability entirely
    live_infra_verdict = (
        f"Sub-second fill model (Card C half-life 1-2s): IS PnL=${pnl_no_penalty:.2f}. "
        f"At 1 Hz scanner cadence (penalty_informed_frac={base_config.penalty_informed_frac:.0%}, "
        f"spread_reduction={base_config.penalty_spread_reduction:.0%}): IS PnL=${pnl_1hz:.2f} "
        f"(${pnl_lost_1hz:.2f} lost = {pct_lost:.1f}% haircut). "
        f"VERDICT: {'BLOCKED — 1Hz kills profitability' if infra_blocked else 'DEGRADED but still positive at 1Hz'}. "
        f"The 1Hz scan loop IS an infra bottleneck: adversely-selected fills at slow refresh "
        f"consume {pct_lost:.1f}% of edge. Sub-second quote refresh is required to fully capture the edge."
    )

    # KPI checks (both fill models)
    kpis = _check_kpis(is_summary, oos_summary, is_opt_summary, oos_opt_summary)

    # Capacity
    cap_rows = _capacity_table()

    # -----------------------------------------------------------------------
    # Build findings dict
    # -----------------------------------------------------------------------
    date_span = f"{IS_START}..{IS_END}"

    metrics: list[dict[str, Any]] = [
        {
            "name": "IS total PnL",
            "value": f"${is_summary.get('total_pnl', 0):.2f}",
            "n": is_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": f"spread={is_summary.get('spread_pnl', 0):.2f}, settlement={is_summary.get('settlement_pnl', 0):.2f}, hedge={is_summary.get('hedge_pnl', 0):.2f}",
        },
        {
            "name": "IS Sharpe (per expiry)",
            "value": _fmt(is_summary.get("sharpe"), ".2f"),
            "n": is_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": "annualised over per-expiry PnL series",
        },
        {
            "name": "IS max drawdown",
            "value": f"${is_summary.get('max_drawdown', 0):.2f}",
            "n": is_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": "peak-to-trough on cumulative PnL",
        },
        {
            "name": "OOS total PnL",
            "value": f"${oos_summary.get('total_pnl', 0):.2f}",
            "n": oos_summary.get("n_expiries", 0),
            "date_span": f"{OOS_START}..{OOS_END}",
            "sanity": f"out-of-sample; n_fills={oos_summary.get('n_fills', 0)}",
        },
        {
            "name": "IS hit rate (% profitable expiries)",
            "value": f"{is_summary.get('hit_rate', 0):.1%}",
            "n": is_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": ">50% = more profitable expiries than losing",
        },
        {
            "name": "IS n_fills",
            "value": str(is_summary.get("n_fills", 0)),
            "n": is_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": f"buy={is_summary.get('n_fills_buy', 0)}, sell={is_summary.get('n_fills_sell', 0)}",
        },
        {
            "name": "Optimistic IS PnL (upper bound)",
            "value": f"${is_opt_summary.get('total_pnl', 0):.2f}",
            "n": is_opt_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": "fill if trade prints AT our quote price",
        },
        {
            "name": "Unhedged IS PnL",
            "value": f"${is_unhedged_summary.get('total_pnl', 0):.2f}",
            "n": is_unhedged_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": "delta hedge contribution = hedged - unhedged PnL",
        },
        {
            "name": "Favorites-only IS PnL",
            "value": f"${is_fav_summary.get('total_pnl', 0):.2f}",
            "n": is_fav_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": f"mid in [0.70,0.95] only; n_fills={is_fav_summary.get('n_fills', 0)}",
        },
        {
            "name": "Favorites-only OOS PnL",
            "value": f"${oos_fav_summary.get('total_pnl', 0):.2f}",
            "n": oos_fav_summary.get("n_expiries", 0),
            "date_span": f"{OOS_START}..{OOS_END}",
            "sanity": f"OOS n_fills={oos_fav_summary.get('n_fills', 0)}",
        },
        {
            "name": "IS PnL at 1Hz scanner (with infra penalty)",
            "value": f"${is_1hz_summary.get('total_pnl', 0):.2f}",
            "n": is_1hz_summary.get("n_expiries", 0),
            "date_span": date_span,
            "sanity": f"penalty_informed_frac=20%, spread_reduction=50%; no-penalty=${pnl_no_penalty:.2f}",
        },
    ]

    split_half = {
        "h1_total_pnl": f"${h1_summary.get('total_pnl', 0):.2f}",
        "h2_total_pnl": f"${h2_summary.get('total_pnl', 0):.2f}",
        "h1_sharpe": _fmt(h1_summary.get("sharpe"), ".2f"),
        "h2_sharpe": _fmt(h2_summary.get("sharpe"), ".2f"),
        "h1_n_fills": h1_summary.get("n_fills", 0),
        "h2_n_fills": h2_summary.get("n_fills", 0),
        "sign_stable": (h1_summary.get("total_pnl", 0) > 0) == (h2_summary.get("total_pnl", 0) > 0),
    }

    n_kpi_pass = sum(1 for k in kpis if k["passed"])
    verdict = (
        f"Strategy MM IS: ${is_summary.get('total_pnl', 0):.2f} PnL [conservative] / "
        f"${is_opt_summary.get('total_pnl', 0):.2f} [optimistic] / "
        f"Sharpe={_fmt(is_summary.get('sharpe'), '.2f')} / "
        f"MaxDD=${is_summary.get('max_drawdown', 0):.2f} / "
        f"n_fills={is_summary.get('n_fills', 0)} / "
        f"hit_rate={is_summary.get('hit_rate', 0):.1%}. "
        f"OOS: ${oos_summary.get('total_pnl', 0):.2f} [conservative] / "
        f"${oos_opt_summary.get('total_pnl', 0):.2f} [optimistic]. "
        f"Hedged IS: ${is_summary.get('total_pnl', 0):.2f} vs Unhedged: ${is_unhedged_summary.get('total_pnl', 0):.2f}. "
        f"Favorites-only IS: ${is_fav_summary.get('total_pnl', 0):.2f} / OOS: ${oos_fav_summary.get('total_pnl', 0):.2f}. "
        f"Split-half: H1=${h1_summary.get('total_pnl', 0):.2f} / H2=${h2_summary.get('total_pnl', 0):.2f} "
        f"({'sign-stable' if split_half['sign_stable'] else 'sign-flip UNSTABLE'}). "
        f"LIVE INFRA: {live_infra_verdict} "
        f"KPIs: {n_kpi_pass}/{len(kpis)} passed."
    )

    findings: dict[str, Any] = {
        "title": "Strategy MM — HL Binary Market-Making",
        "headline": verdict,
        "metrics": metrics,
        "split_half": split_half,
        "kpis": kpis,
        "is_summary": is_summary,
        "oos_summary": oos_summary,
        "is_opt_summary": is_opt_summary,
        "oos_opt_summary": oos_opt_summary,
        "is_unhedged_summary": is_unhedged_summary,
        "is_fav_summary": is_fav_summary,
        "oos_fav_summary": oos_fav_summary,
        "is_1hz_summary": is_1hz_summary,
        "h1_summary": h1_summary,
        "h2_summary": h2_summary,
        "sweep": sweep,
        "capacity": cap_rows,
        "live_infra_verdict": live_infra_verdict,
        "infra_blocked": infra_blocked,
        "config": {
            "fill_model": base_config.fill_model,
            "half_edge": base_config.half_edge,
            "size_usd": base_config.size_usd,
            "hedged": base_config.hedged,
            "hedge_ratio": base_config.hedge_ratio,
            "favorites_only": base_config.favorites_only,
            "tte_min_seconds": base_config.tte_min_seconds,
            "tte_max_seconds": base_config.tte_max_seconds,
        },
        "verdict": verdict,
    }

    # -----------------------------------------------------------------------
    # Build HTML report
    # -----------------------------------------------------------------------
    rpt = Report(title="Strategy MM — HL Binary Market-Making")

    # Summary
    summary_html = (
        f"<p><strong>Base config:</strong> fill_model=conservative, half_edge=1-tick, "
        f"size=$100, hedged=True (hedge_ratio={DEFAULT_HEDGE_RATIO}).</p>"
        f"<p><strong>Corpus:</strong> IS={IS_START}..{IS_END}, OOS={OOS_START}..{OOS_END}.</p>"
        f"<p><strong>Safety gates:</strong> min_bid_notional=$25, max_spread=500bps, "
        f"stale_halt=60s, tte_min=30min, tte_max=20h.</p>" + _metrics_table_html(metrics)
    )
    rpt.add_card("Summary & KPIs", summary_html)

    # KPIs
    rpt.add_card("KPI Pass/Fail", _kpi_table_html(kpis))

    # Split-half stability
    sh_html = (
        f"<p>H1: {SPLIT_H1_START}..{SPLIT_H1_END} | H2: {SPLIT_H2_START}..{SPLIT_H2_END}</p>"
        f"<table><thead><tr><th>Period</th><th>PnL</th><th>Sharpe</th><th>n Fills</th></tr></thead>"
        f"<tbody>"
        f"<tr><td>H1</td><td>{split_half['h1_total_pnl']}</td><td>{split_half['h1_sharpe']}</td><td>{split_half['h1_n_fills']}</td></tr>"
        f"<tr><td>H2</td><td>{split_half['h2_total_pnl']}</td><td>{split_half['h2_sharpe']}</td><td>{split_half['h2_n_fills']}</td></tr>"
        f"</tbody></table>"
        f"<p>Sign stability: <strong>{'STABLE' if split_half['sign_stable'] else 'UNSTABLE — sign flip'}</strong></p>"
    )
    rpt.add_card("Split-Half Stability", sh_html)

    # Capacity
    cap_html = (
        "<p>Card A findings: TOB notional ~$107, within 100bps ~$679. "
        "Card B: maker realizes 0.95pp spread per round-trip (55.4% fill hit rate).</p>"
        + _capacity_table_html(cap_rows)
    )
    rpt.add_card("Capacity Analysis", cap_html)

    # Favorites-only comparison table
    fav_html = (
        "<p>Favorites-only: only quote when binary mid in [0.70, 0.95] (high-conviction range).</p>"
        "<table><thead><tr><th>Config</th><th>IS PnL</th><th>IS Sharpe</th><th>IS Fills</th>"
        "<th>OOS PnL</th><th>OOS Fills</th></tr></thead><tbody>"
        f"<tr><td>Full two-sided</td>"
        f"<td>${is_summary.get('total_pnl', 0):.2f}</td>"
        f"<td>{_fmt(is_summary.get('sharpe'), '.2f')}</td>"
        f"<td>{is_summary.get('n_fills', 0)}</td>"
        f"<td>${oos_summary.get('total_pnl', 0):.2f}</td>"
        f"<td>{oos_summary.get('n_fills', 0)}</td>"
        f"</tr>"
        f"<tr><td>Favorites-only [0.70-0.95]</td>"
        f"<td>${is_fav_summary.get('total_pnl', 0):.2f}</td>"
        f"<td>{_fmt(is_fav_summary.get('sharpe'), '.2f')}</td>"
        f"<td>{is_fav_summary.get('n_fills', 0)}</td>"
        f"<td>${oos_fav_summary.get('total_pnl', 0):.2f}</td>"
        f"<td>{oos_fav_summary.get('n_fills', 0)}</td>"
        f"</tr>"
        "</tbody></table>"
    )
    rpt.add_card("Favorites-Only vs Full Two-Sided", fav_html)

    # Live infra verdict card
    infra_color = "#f78166" if infra_blocked else "#f0e68c"
    infra_html = (
        f"<p style='color:{infra_color}'><strong>{'BLOCKED' if infra_blocked else 'DEGRADED'}"
        f" at 1 Hz scanner cadence</strong></p>"
        f"<p>{live_infra_verdict}</p>"
        "<table><thead><tr><th>Scenario</th><th>IS PnL</th><th>IS Sharpe</th>"
        "<th>Delta vs sub-second</th></tr></thead><tbody>"
        f"<tr><td>Sub-second refresh (no penalty)</td>"
        f"<td>${pnl_no_penalty:.2f}</td>"
        f"<td>{_fmt(is_summary.get('sharpe'), '.2f')}</td>"
        f"<td>—</td></tr>"
        f"<tr><td>1 Hz scanner (20% adversely selected)</td>"
        f"<td>${pnl_1hz:.2f}</td>"
        f"<td>{_fmt(is_1hz_summary.get('sharpe'), '.2f')}</td>"
        f"<td style='color:{infra_color}'>${-pnl_lost_1hz:.2f} ({pct_lost:.1f}% haircut)</td></tr>"
        "</tbody></table>"
        "<p><em>Card C basis: perp→binary lead-lag half-life 1–2s. "
        "At 1 Hz scan, ~20% of fills arrive after quote is stale (assumed). "
        "Each such fill captures only 50% of intended spread. "
        "True impact likely higher — adverse selection at slow refresh concentrates "
        "on the worst fills (informationally motivated trades).</em></p>"
    )
    rpt.add_card(
        "LIVE INFRA VERDICT: 1Hz vs Sub-Second Refresh",
        infra_html,
        notes=(
            "Queue-position approximation: conservative fill model already assumes back-of-queue. "
            "1Hz penalty adds an additional adverse-selection layer on top of that. "
            "The sub-second scenario is NOT achievable on the current 1Hz engine without infra changes."
        ),
    )

    # Sensitivity sweep table
    rpt.add_card("Sensitivity Sweep (IS)", _sweep_table_html(sweep))

    # Verdict
    rpt.add_card("Verdict", f"<p>{verdict}</p>")

    # Plots
    fig_equity = _plot_equity_curves(is_results, oos_results, is_results_opt, is_results_unhedged)
    rpt.add_card(
        "Equity Curves",
        "<p>IS vs OOS, optimistic vs conservative fill model, hedged vs unhedged.</p>",
        fig=fig_equity,
    )
    plt.close(fig_equity)

    fig_attr = _plot_pnl_attribution(is_results)
    rpt.add_card("PnL Attribution (IS)", "<p>Breakdown: spread capture, settlement, hedge PnL.</p>", fig=fig_attr)
    plt.close(fig_attr)

    fig_inv = _plot_inventory_distribution(is_results)
    rpt.add_card("Inventory Distribution (IS)", "<p>Max and net token inventory per expiry.</p>", fig=fig_inv)
    plt.close(fig_inv)

    fig_sweep = _plot_sensitivity_heatmap(sweep)
    rpt.add_card(
        "Sensitivity Heatmap (IS)", "<p>Total PnL, Sharpe, hit rate across parameter variants.</p>", fig=fig_sweep
    )
    plt.close(fig_sweep)

    # Render HTML
    generated_at = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    card_html = rpt._HTML_TEMPLATE.format(
        title="Strategy MM — HL Binary Market-Making",
        css=rpt._DARK_CSS,
        generated_at=generated_at,
        cards="\n".join(
            rpt._CARD_TEMPLATE.format(
                title=c["title"],
                body=c["html_body"],
                img_tag=(
                    f'<img src="data:image/png;base64,{fig_to_base64(c["fig"])}" alt="{c["title"]}">'
                    if c.get("fig") is not None
                    else ""
                ),
                notes_html=f'<div class="notes">{c["notes"]}</div>' if c.get("notes") else "",
            )
            for c in rpt._cards
        ),
    )

    # Write outputs if out_dir provided
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "strategy_mm.html").write_text(card_html, encoding="utf-8")
        (out_path / "strategy_mm.json").write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
        _log.info("Written to %s", out_path)

    con.close()
    return card_html, findings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "../../data")
    out_dir = str(Path(__file__).parents[3] / "docs" / "research" / "_cards")

    card_html, findings = build_card(data_root, out_dir=out_dir)

    print(f"Strategy MM written to {out_dir}/strategy_mm.html")
    print(f"Findings written to {out_dir}/strategy_mm.json")
    print(f"\nVerdict:\n{findings.get('verdict', 'N/A')}")
    print("\nKey metrics:")
    for m in findings.get("metrics", []):
        print(f"  {m['name']}: {m['value']} (n={m.get('n')}, {m.get('date_span')}) — {m.get('sanity', '')}")
    print("\nKPI pass/fail:")
    for k in findings.get("kpis", []):
        print(f"  [{k['status']}] {k['name']}: {k['value']} (target: {k['target']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
