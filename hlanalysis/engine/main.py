from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from ..adapters.hyperliquid import HyperliquidAdapter
from ..config import load_config
from .config import load_deploy_config, load_strategy_config
from .runtime import EngineRuntime


def main() -> None:
    p = argparse.ArgumentParser(description="HLAnalysis MM engine (Phase 1)")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument(
        "--symbols-config", type=Path, default=Path("config/symbols.yaml"),
        help="Subscriptions feeding MarketState (HIP-4 binaries + BTC perp).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logger.remove()
    logger.add(lambda m: print(m, end=""), level=args.log_level)

    strategy_cfg = load_strategy_config(args.strategy_config)
    deploy_cfg = load_deploy_config(args.deploy_config)
    sym_cfg = load_config(args.symbols_config)

    # Engine consumes only Hyperliquid subs. Binance is recorder-only in Phase 1.
    hl_subs = [s for s in sym_cfg.subscriptions if s.venue == "hyperliquid"]

    runtime = EngineRuntime(
        strategy_cfg=strategy_cfg,
        deploy_cfg=deploy_cfg,
        adapter_factory=HyperliquidAdapter,
        subscriptions=hl_subs,
    )
    if strategy_cfg.paper_mode:
        logger.warning("PAPER MODE — no real orders will be placed")
    else:
        logger.error("LIVE MODE — real money at stake; ensure caps in strategy.yaml are correct")

    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()
