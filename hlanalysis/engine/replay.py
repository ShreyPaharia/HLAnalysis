from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from ..events import NormalizedEvent
from ..strategy.base import Strategy
from ..strategy.types import Action, Decision, Position
from .market_state import MarketState


class ReplayRunner:
    """Deterministic offline replay of NormalizedEvents through a Strategy.

    Walks events in arrival order, maintains MarketState, and triggers
    `strategy.evaluate()` once whenever any of:
      - a BBO/Trade/Mark event lands on a known question's leg, or
      - the BTC reference price moves,
    so we don't run the full scan loop for every event.

    Phase 1A scope: no order routing, no risk gate, no persistence. Logging-only.
    Caller can pass `position_lookup` to simulate held positions.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        reference_symbol: str = "BTC",
        position_lookup: dict[int, Position] | None = None,
        sampling_dt_seconds: int = 60,
        reference_sigma_source: str = "mark",
    ) -> None:
        self._strategy = strategy
        self._ref = reference_symbol
        self._positions = position_lookup or {}
        self._market = MarketState()
        # Honor the same mark-bucketing cadence as the live engine so replay /
        # paper validation sees the σ window the strategy assumes. Default 60s
        # preserves legacy replay behaviour (32 one-minute bars).
        self._sampling_dt_seconds = int(sampling_dt_seconds)
        self._market.set_reference_cadence(
            reference_symbol, sampling_dt_seconds=self._sampling_dt_seconds,
        )
        # σ/OHLC source for the reference symbol, mirroring the live engine.
        # "mark" (default) preserves legacy replay; "bbo" sources σ from the
        # dense BBO mid (parity with the live PM bbo path + the backtest
        # `_load_binance_bbo_reference`).
        self._market.set_reference_source(reference_symbol, reference_sigma_source)
        # Hold the same wall-clock σ window as the legacy 32×60s default,
        # scaled to the configured cadence (32 bars at 60s; 384 bars at 5s).
        self._recent_returns_n = max(32, (32 * 60) // self._sampling_dt_seconds)

    def run_iter(self, events: Iterable[NormalizedEvent]) -> Iterator[Decision]:
        for ev in events:
            self._market.apply(ev)
            ref = self._market.last_mark(self._ref)
            if ref is None:
                continue
            now_ns = ev.exchange_ts or ev.local_recv_ts
            for q in self._market.all_questions():
                books = {}
                if q.yes_symbol:
                    b = self._market.book(q.yes_symbol)
                    if b is not None:
                        books[q.yes_symbol] = b
                if q.no_symbol:
                    b = self._market.book(q.no_symbol)
                    if b is not None:
                        books[q.no_symbol] = b
                if not books:
                    continue
                d = self._strategy.evaluate(
                    question=q,
                    books=books,
                    reference_price=ref,
                    recent_returns=self._market.recent_returns(
                        self._ref, n=self._recent_returns_n,
                    ),
                    recent_hl_bars=self._market.recent_hl_bars(
                        self._ref, n=self._recent_returns_n,
                    ),
                    recent_volume_usd=self._market.recent_volume_usd(
                        q.yes_symbol, now=now_ns
                    ) + self._market.recent_volume_usd(
                        q.no_symbol, now=now_ns
                    ),
                    position=self._positions.get(q.question_idx),
                    now_ns=now_ns,
                )
                if d.action is not Action.HOLD:
                    logger.info(
                        "replay decision question_idx={} action={} intents={}",
                        q.question_idx, d.action.value, len(d.intents),
                    )
                yield d

    def run_parquet(self, parquet_glob: Path) -> Iterator[Decision]:
        """Load events from parquet via duckdb, ordered by exchange_ts.

        Expects parquet files written by the recorder (one file per partition).
        Yields decisions in arrival order.
        """
        import duckdb  # local import to keep strategy/ unaffected
        from ..events import (
            BboEvent, BookSnapshotEvent, BookDeltaEvent, FundingEvent,
            HealthEvent, LiquidationEvent, MarketMetaEvent, MarkEvent,
            OpenInterestEvent, OracleEvent, QuestionMetaEvent,
            SettlementEvent, TradeEvent,
        )

        TYPE_MAP: dict[str, type[Any]] = {
            "trade": TradeEvent, "book_snapshot": BookSnapshotEvent,
            "book_delta": BookDeltaEvent, "bbo": BboEvent, "mark": MarkEvent,
            "oracle": OracleEvent, "open_interest": OpenInterestEvent,
            "funding": FundingEvent, "liquidation": LiquidationEvent,
            "market_meta": MarketMetaEvent, "question_meta": QuestionMetaEvent,
            "settlement": SettlementEvent, "health": HealthEvent,
        }
        con = duckdb.connect()
        rows = con.execute(
            f"SELECT * FROM read_parquet('{parquet_glob}') ORDER BY exchange_ts"
        ).fetchall()
        cols = [d[0] for d in con.description]

        def _gen() -> Iterator[NormalizedEvent]:
            for row in rows:
                d = dict(zip(cols, row, strict=False))
                cls = TYPE_MAP.get(d.get("event_type"))
                if cls is None:
                    continue
                # Drop nulls; pydantic validators handle defaults.
                clean = {k: v for k, v in d.items() if v is not None}
                yield cls(**clean)

        yield from self.run_iter(_gen())


def _cli() -> None:
    """python -m hlanalysis.engine.replay --parquet 'data/.../*.parquet'"""
    import argparse
    from pathlib import Path

    from .config import load_strategy_config, match_question  # noqa: F401
    from .runtime import build_late_resolution_config
    from ..strategy.late_resolution import LateResolutionStrategy

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="parquet glob")
    p.add_argument("--strategy-config", default="config/strategy.yaml")
    args = p.parse_args()

    cfg = load_strategy_config(Path(args.strategy_config))
    runtime_cfg = build_late_resolution_config(cfg)
    runner = ReplayRunner(strategy=LateResolutionStrategy(runtime_cfg))
    n = 0
    for _ in runner.run_parquet(Path(args.parquet)):
        n += 1
    logger.info("replay finished — {} decisions emitted", n)


if __name__ == "__main__":
    _cli()
