from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, Position

from .data.binance_klines import Kline
from .data.schemas import PMMarket, PMTrade
from .diagnostics import DiagnosticRow, build_row, write_diagnostics
from .fills import Fill, FillModelConfig, simulate_fill
from .hftbt_adapter import EventKind, build_event_stream
from .market_state import SimMarketState
from .question_builder import build_question_view


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    scanner_interval_seconds: int
    fill_model: FillModelConfig
    synthetic_half_spread: float
    synthetic_depth: float
    day_open_btc: float


@dataclass(slots=True)
class RunResult:
    fills: list[Fill] = field(default_factory=list)
    n_decisions: int = 0
    realized_pnl_usd: float | None = None
    diagnostics_count: int = 0


_STOP_DISABLED_SENTINEL = -1.0  # bid_px is always ≥ 0 for PM, so this never trips


def _strategy_stop_loss_pct(strategy: Strategy) -> float | None:
    """Sim-only contract: v1/v2 strategies expose `cfg.stop_loss_pct`. None
    disables the stop. Returns None when disabled, else the percentage.
    """
    raw = getattr(getattr(strategy, "cfg", None), "stop_loss_pct", None)
    if raw is None:
        return None
    # v1 historically used a large sentinel float to mean "no stop"; treat the
    # same as None so pure-Python None and the sentinel converge.
    if float(raw) >= 1e8:
        return None
    return float(raw)


def _stop_price(fill_price: float, stop_pct: float | None) -> float:
    if stop_pct is None:
        return _STOP_DISABLED_SENTINEL
    return max(0.0, fill_price * (1.0 - stop_pct / 100.0))


def run_one_market(
    strategy: Strategy,
    market: PMMarket,
    klines: list[Kline],
    trades: list[PMTrade],
    cfg: RunnerConfig,
    *,
    diagnostics_dir: Path | None = None,
) -> RunResult:
    state = SimMarketState()
    result = RunResult()
    pos: Position | None = None
    last_scan_ns = market.start_ts_ns
    scan_interval_ns = cfg.scanner_interval_seconds * 1_000_000_000
    diag_rows: list[DiagnosticRow] = []

    events = list(build_event_stream(
        trades=trades, klines=klines,
        half_spread=cfg.synthetic_half_spread, depth=cfg.synthetic_depth,
        yes_token_id=market.yes_token_id, no_token_id=market.no_token_id,
    ))

    stop_pct = _strategy_stop_loss_pct(strategy)

    for ev in events:
        if ev.kind == EventKind.L2:
            state.apply_l2(ev.payload)
        elif ev.kind == EventKind.TRADE_TS:
            state.apply_trade_ts(ev.payload.token_id, ev.ts_ns)
        elif ev.kind == EventKind.KLINE:
            state.apply_kline(ev.payload)

        if ev.ts_ns - last_scan_ns < scan_interval_ns:
            continue

        qv = build_question_view(market, day_open_btc=cfg.day_open_btc, now_ns=ev.ts_ns)
        books = {market.yes_token_id: state.book(market.yes_token_id),
                 market.no_token_id:  state.book(market.no_token_id)}
        books = {k: v for k, v in books.items() if v is not None}
        recent_returns = state.recent_returns(now_ns=ev.ts_ns, lookback_seconds=86_400)
        ref_close = state.latest_btc_close() or cfg.day_open_btc

        decision = strategy.evaluate(
            question=qv, books=books,
            reference_price=float(ref_close),
            recent_returns=recent_returns,
            recent_volume_usd=0.0,
            position=pos, now_ns=ev.ts_ns,
        )
        result.n_decisions += 1
        result.diagnostics_count += len(decision.diagnostics)

        # Collect diagnostics row (only when a diagnostics_dir was requested)
        if diagnostics_dir is not None:
            yes_book = books.get(market.yes_token_id)
            no_book = books.get(market.no_token_id)
            diag_rows.append(build_row(
                ts_ns=ev.ts_ns,
                condition_id=market.condition_id,
                question_idx=qv.question_idx,
                decision=decision,
                ref_price=float(ref_close),
                yes_bid=yes_book.bid_px if yes_book is not None else None,
                yes_ask=yes_book.ask_px if yes_book is not None else None,
                no_bid=no_book.bid_px if no_book is not None else None,
                no_ask=no_book.ask_px if no_book is not None else None,
            ))

        if decision.action == Action.ENTER and decision.intents:
            intent = decision.intents[0]
            book = state.book(intent.symbol)
            if book is not None:
                fill = simulate_fill(intent, book, cfg.fill_model)
                if fill.size > 0:
                    result.fills.append(fill)
                    stop_px = _stop_price(fill.price, stop_pct)
                    pos = Position(
                        question_idx=qv.question_idx, symbol=intent.symbol,
                        qty=fill.size, avg_entry=fill.price,
                        stop_loss_price=stop_px, last_update_ts_ns=ev.ts_ns,
                    )
        elif decision.action == Action.EXIT and decision.intents:
            intent = decision.intents[0]
            book = state.book(intent.symbol)
            if book is not None:
                fill = simulate_fill(intent, book, cfg.fill_model)
                if fill.size > 0:
                    result.fills.append(fill)
                    pos = None
        last_scan_ns = ev.ts_ns

    # Write diagnostics parquet for this market (after event loop, before return)
    if diagnostics_dir is not None and diag_rows:
        write_diagnostics(diag_rows, diagnostics_dir / f"{market.condition_id}.parquet")

    if pos is not None:
        is_yes_pos = pos.symbol == market.yes_token_id
        won = ((market.resolved_outcome == "yes" and is_yes_pos)
               or (market.resolved_outcome == "no" and not is_yes_pos))
        settle_px = 1.0 if won else 0.0
        result.fills.append(Fill(
            cloid="settle", symbol=pos.symbol,
            side="sell" if pos.qty > 0 else "buy",
            price=settle_px, size=abs(pos.qty), fee=0.0, partial=False,
        ))
        pos = None

    realized = 0.0
    for f in result.fills:
        notional = f.price * f.size
        if f.side == "buy":
            realized -= notional
        else:
            realized += notional
        realized -= f.fee
    result.realized_pnl_usd = realized
    return result
