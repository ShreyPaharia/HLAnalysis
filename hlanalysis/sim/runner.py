from __future__ import annotations

from dataclasses import dataclass, field

from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, Position

from .data.binance_klines import Kline
from .data.schemas import PMMarket, PMTrade
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


def _strategy_stop_loss_pct(strategy: Strategy) -> float:
    """Sim-only contract: v1/v2 strategies expose `cfg.stop_loss_pct`. None or
    sentinel-large value disables the stop. Used to set Position.stop_loss_price
    after a fill — the runner needs *some* stop level for stop-loss simulation.
    """
    raw = getattr(getattr(strategy, "cfg", None), "stop_loss_pct", None)
    if raw is None:
        return 1e9
    return float(raw)


def run_one_market(
    strategy: Strategy,
    market: PMMarket,
    klines: list[Kline],
    trades: list[PMTrade],
    cfg: RunnerConfig,
) -> RunResult:
    state = SimMarketState(vol_sampling_dt_seconds=60)
    result = RunResult()
    pos: Position | None = None
    last_scan_ns = market.start_ts_ns
    scan_interval_ns = cfg.scanner_interval_seconds * 1_000_000_000

    events = list(build_event_stream(
        trades=trades, klines=klines,
        half_spread=cfg.synthetic_half_spread, depth=cfg.synthetic_depth,
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

        if decision.action == Action.ENTER and decision.intents:
            intent = decision.intents[0]
            book = state.book(intent.symbol)
            if book is not None:
                fill = simulate_fill(intent, book, cfg.fill_model)
                if fill.size > 0:
                    result.fills.append(fill)
                    stop_px = max(0.0, fill.price * (1.0 - stop_pct / 100.0))
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
