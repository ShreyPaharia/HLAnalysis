from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from ..adapters.binance import BinanceAdapter
from ..adapters.composite import CompositeAdapter
from ..adapters.hyperliquid import HyperliquidAdapter
from ..adapters.polymarket import PolymarketAdapter
from ..config import RecorderConfig, Subscription, load_config
from ..events import Mechanism, ProductType
from .config import load_deploy_config, load_strategies_config
from .runtime import EngineRuntime

# Venues whose symbols.yaml subscriptions the engine consumes wholesale. Binance
# is deliberately excluded here: the engine ingests only a single dedicated,
# code-constructed bbo reference feed (see `binance_spot_reference_subscription`),
# NOT the heavier recorder-side binance entries (trades/book/mark/funding).
_ENGINE_VENUES_FROM_SYMBOLS = ("hyperliquid", "polymarket")


def binance_spot_reference_subscription() -> Subscription:
    """The engine's dedicated Binance SPOT BTCUSDT `bbo` reference feed.

    This is the price/σ/strike reference for the PM up/down slots: PM resolves
    against the Binance spot BTC/USDT 1m candle close, so the live price and the
    captured strike must both be spot (no perp/spot basis in the edge). Remapped
    on ingest to internal symbol ``BTCUSDT_SPOT`` (see EngineRuntime._ingest_loop) so
    it never collides with the perp ``BTCUSDT`` book key. spot bookTicker is not
    geo-blocked from the EC2/Tokyo IP.
    """
    return Subscription(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        channels=("bbo",),
    )


def build_engine_subscriptions(sym_cfg: RecorderConfig) -> list[Subscription]:
    """Subscriptions feeding the engine's MarketState: the HL + PM entries from
    symbols.yaml verbatim, plus the dedicated Binance SPOT bbo reference feed.

    Binance entries in symbols.yaml are skipped (recorder-only); the spot
    reference feed is appended explicitly so it's obvious this is the engine's
    lean, bbo-only reference — not the recorder's full binance ingest.

    PM slots use ``reference_symbol: BTCUSDT_SPOT`` (no perp/spot basis in
    price, σ, or strike). The perp BTCUSDT feed was removed 2026-06-01."""
    subs = [
        s for s in sym_cfg.subscriptions
        if s.venue in _ENGINE_VENUES_FROM_SYMBOLS
    ]
    subs.append(binance_spot_reference_subscription())
    return subs


def build_engine_adapter() -> CompositeAdapter:
    """One merged stream over HL + PM + the binance reference feed. Each child
    adapter only receives its own venue's subs (CompositeAdapter filters by
    `venue`), so the BinanceAdapter sees only the bbo reference sub."""
    return CompositeAdapter(
        [HyperliquidAdapter(), PolymarketAdapter(), BinanceAdapter()]
    )


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru so adapter / SDK logs land
    in the same stream as engine logs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def main() -> None:
    p = argparse.ArgumentParser(description="HLAnalysis MM engine (Phase 1)")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument(
        "--symbols-config", type=Path, default=Path("config/symbols.yaml"),
        help="Subscriptions feeding MarketState (HIP-4 binaries + BTC spot bbo reference).",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--log-format", choices=["json", "pretty"], default="json",
        help=(
            "Log output format. 'json' (default) emits one JSON object per line, "
            "suitable for journald / SSM / jq pipelines. "
            "'pretty' emits human-readable coloured text for local dev runs."
        ),
    )
    args = p.parse_args()

    logger.remove()
    if args.log_format == "json":
        # Structured JSON sink: every line is a self-contained JSON object with
        # `time`, `level`, `message`, and `extra.{alias, kind, question_idx, ...}`
        # as top-level queryable fields. This is the deployed (journald) default so
        # that `journalctl -u engine | jq ...` works without grep archaeology.
        logger.add(sys.stderr, serialize=True, level=args.log_level)
    else:
        # Human-readable coloured output for local dev / debugging.
        logger.add(sys.stderr, level=args.log_level)
    # Route stdlib logging (used by adapters/hyperliquid.py + SDK + asyncio)
    # into loguru. Without this, adapter INFO logs are silently dropped and
    # only WARNING+ leaks to stderr via Python's lastResort handler.
    logging.basicConfig(handlers=[_InterceptHandler()], level=args.log_level.upper(), force=True)

    strategies_cfg = load_strategies_config(args.strategy_config)
    deploy_cfg = load_deploy_config(args.deploy_config)
    sym_cfg = load_config(args.symbols_config)

    # Engine consumes HL + PM subscriptions plus a dedicated, lean Binance
    # BTCUSDT SPOT bbo reference feed (the PM slots' price/σ/strike reference).
    # The heavier binance entries in symbols.yaml stay recorder-only.
    engine_subs = build_engine_subscriptions(sym_cfg)

    runtime = EngineRuntime(
        strategies=strategies_cfg.strategies,
        deploy_cfg=deploy_cfg,
        adapter_factory=build_engine_adapter,
        subscriptions=engine_subs,
    )
    for s in strategies_cfg.strategies:
        if s.paper_mode:
            logger.warning(
                "PAPER MODE alias={} ({}) — no real orders will be placed",
                s.account_alias, s.strategy_type,
            )
        else:
            # warning, not error: this is a startup banner, not a failure
            # condition. Logging at ERROR pollutes journalctl ERROR filters
            # and any alerting hooked off the loguru level.
            logger.warning(
                "LIVE MODE alias={} ({}) — real money at stake; "
                "ensure caps in strategy.yaml are correct",
                s.account_alias, s.strategy_type,
            )

    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()
