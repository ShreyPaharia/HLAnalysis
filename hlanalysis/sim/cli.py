from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from .data import polymarket
from .data.binance_klines import Kline, fetch_klines
from .data.cache import Cache
from .data.schemas import PMMarket, PMTrade
from .fills import FillModelConfig
from .metrics import summarise_run
from .report import write_single_run_report, write_tuning_report
from .runner import RunnerConfig, run_one_market
from .tuning import TuningJob
from .tuning_config import load_tuning_yaml
from .v1_factory import build_v1_strategy_from_params


def _build_v2_factory():
    # Lazy import so the v2 factory module can live next to v1 without circular concerns.
    from .v2_factory import build_v2_strategy_from_params
    return build_v2_strategy_from_params


def _factory_for(strategy: str):
    if strategy == "v2":
        return _build_v2_factory()
    return build_v1_strategy_from_params


def cmd_fetch(args: argparse.Namespace) -> None:
    cache = Cache(root=Path(args.cache_root))
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    days: list[str] = []
    cur = start
    while cur < end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    logger.info(f"Fetching {len(days)} days from PM Gamma + Binance")
    markets = polymarket.discover_btc_updown_markets(days)
    logger.info(f"Found {len(markets)} candidate markets")
    for m in markets:
        if cache.read_trades(m.condition_id) and not args.refresh:
            continue
        trades = polymarket.fetch_trades(m.condition_id)
        # Spec §4.1 per-market floor filter — skip illiquid markets.
        # Gamma's `volume` is sometimes 0 even for active markets (unreliable source),
        # so only enforce the volume floor when Gamma actually reports a positive value.
        vol_below_floor = (m.total_volume_usd > 0) and (m.total_volume_usd < args.min_volume_usd)
        if len(trades) < args.min_trades or vol_below_floor:
            logger.info(
                f"PM {m.condition_id}: skip (trades={len(trades)} vol=${m.total_volume_usd:.0f})"
            )
            continue
        # Backfill n_trades from the actual fetched count and persist canonical metadata
        # so the cache is self-contained for downstream loading.
        m_persist = m.model_copy(update={"n_trades": len(trades)})
        cache.write_trades(m.condition_id, trades)
        cache.update_manifest(
            condition_id=m.condition_id,
            n_rows=len(trades),
            last_pull_ts_ns=int(datetime.now(timezone.utc).timestamp() * 1e9),
            market=m_persist,
        )
        logger.info(f"PM {m.condition_id}: {len(trades)} trades")
    klines = fetch_klines(
        start_ts_ms=int(start.timestamp() * 1000),
        end_ts_ms=int(end.timestamp() * 1000),
    )
    klines_dir = Path(args.cache_root) / "btc_klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    klines_path = klines_dir / f"{args.start}_to_{args.end}.json"
    klines_path.write_text(json.dumps([asdict(k) for k in klines]))
    logger.info(f"BTC klines: {len(klines)} → {klines_path}")


def _load_jobs_from_cache(cache_root: Path) -> list[TuningJob]:
    cache = Cache(root=cache_root)
    klines_dir = cache_root / "btc_klines"
    klines: list[Kline] = []
    if klines_dir.exists():
        for f in sorted(klines_dir.glob("*.json")):
            klines.extend(Kline(**k) for k in json.loads(f.read_text()))
        klines.sort(key=lambda k: k.ts_ns)
    if not klines:
        return []

    jobs: list[TuningJob] = []
    for cond_id in cache.manifest_keys():
        trades = cache.read_trades(cond_id)
        if not trades:
            continue
        m = cache.get_market(cond_id)
        if m is None:
            # Legacy cache without persisted metadata. Refuse to fabricate canonical
            # YES/NO ordering from trades — sorted() doesn't match PM's clobTokenIds
            # order, and an unknown outcome silently settles every position to 0.
            # Re-fetch with `hl-sim fetch --refresh` to populate the manifest.
            logger.warning(
                f"PM {cond_id}: missing manifest metadata (legacy cache); skipping. "
                f"Re-run `hl-sim fetch --refresh` to populate."
            )
            continue
        market_klines = [k for k in klines if m.start_ts_ns <= k.ts_ns <= m.end_ts_ns]
        if not market_klines:
            continue
        jobs.append(TuningJob(
            market=m, klines=market_klines, trades=trades,
            day_open_btc=market_klines[0].open,
        ))
    jobs.sort(key=lambda j: j.market.start_ts_ns)
    return jobs


def cmd_run(args: argparse.Namespace) -> None:
    jobs = _load_jobs_from_cache(Path(args.cache_root))
    if not jobs:
        logger.error("No jobs in cache; run `hl-sim fetch` first.")
        return
    factory = _factory_for(args.strategy)
    params = json.loads(Path(args.config).read_text())
    strat = factory(params)
    fill_cfg = FillModelConfig(
        slippage_bps=args.slippage_bps, fee_taker=args.fee_taker,
        book_depth_assumption=args.depth,
    )
    pnl: list[float] = []
    n_trades = 0
    for j in jobs:
        rcfg = RunnerConfig(
            scanner_interval_seconds=60, fill_model=fill_cfg,
            synthetic_half_spread=args.half_spread, synthetic_depth=args.depth,
            day_open_btc=j.day_open_btc,
        )
        res = run_one_market(strat, j.market, j.klines, j.trades, rcfg)
        pnl.append(res.realized_pnl_usd or 0.0)
        n_trades += len(res.fills)
    summary = summarise_run(pnl, n_trades=n_trades)
    out_dir = Path(args.out_dir)
    write_single_run_report(
        out_dir=out_dir, strategy_name=args.strategy,
        config_summary=params, per_market_pnl=pnl, summary=summary,
    )
    logger.info(f"Report → {out_dir}/report.md")


def cmd_tune(args: argparse.Namespace) -> None:
    from .tuning import run_tuning_parallel  # imports parallel runner from Task 28
    jobs = _load_jobs_from_cache(Path(args.cache_root))
    if not jobs:
        logger.error("No jobs in cache; run `hl-sim fetch` first.")
        return
    tcfg = load_tuning_yaml(Path(args.grid))
    grid = tcfg.v2_grid if args.strategy == "v2" else tcfg.v1_grid
    factory = _factory_for(args.strategy)
    fill_cfg = FillModelConfig(
        slippage_bps=args.slippage_bps, fee_taker=args.fee_taker,
        book_depth_assumption=args.depth,
    )
    base_rcfg = RunnerConfig(
        scanner_interval_seconds=60, fill_model=fill_cfg,
        synthetic_half_spread=args.half_spread, synthetic_depth=args.depth,
        day_open_btc=0.0,
    )
    out_dir = Path(args.out_dir) / args.run_id
    rows = list(run_tuning_parallel(
        grid=grid, strategy_factory=factory,
        runner_cfg_factory=lambda p: base_rcfg, jobs=jobs,
        train=tcfg.run.get("train_markets", 60),
        test=tcfg.run.get("test_markets", 15),
        step=tcfg.run.get("step_markets", 15),
        out_dir=out_dir, n_workers=args.workers,
    ))
    write_tuning_report(out_dir=out_dir, strategy_name=args.strategy, rows=rows, top_k=10)
    logger.info(f"Tuning report → {out_dir}/report.md")


def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.run_dir)
    log_path = out_dir / "results.jsonl"
    if not log_path.exists():
        logger.error(f"No results.jsonl at {log_path}")
        return
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    write_tuning_report(out_dir=out_dir, strategy_name=args.strategy, rows=rows, top_k=args.top_k)


def main() -> None:
    p = argparse.ArgumentParser(prog="hl-sim")
    sp = p.add_subparsers(dest="cmd", required=True)

    pf = sp.add_parser("fetch", help="Fetch PM markets + Binance klines into cache")
    pf.add_argument("--start", required=True)
    pf.add_argument("--end", required=True)
    pf.add_argument("--cache-root", default="data/sim")
    pf.add_argument("--refresh", action="store_true")
    pf.add_argument("--min-trades", type=int, default=30,
                    help="Skip markets with fewer trades (spec §4.1 floor)")
    pf.add_argument("--min-volume-usd", type=float, default=1000.0,
                    help="Skip markets with notional volume below this (spec §4.1 floor)")
    pf.set_defaults(func=cmd_fetch)

    pr = sp.add_parser("run", help="Run one strategy config across cached markets")
    pr.add_argument("--strategy", choices=["v1", "v2"], required=True)
    pr.add_argument("--config", required=True, help="JSON file of param dict")
    pr.add_argument("--cache-root", default="data/sim")
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--slippage-bps", type=float, default=5.0)
    pr.add_argument("--fee-taker", type=float, default=0.0,
                    help="Polymarket CLOB takers pay 0%; HL HIP-4 ~3-5bps")
    pr.add_argument("--half-spread", type=float, default=0.005)
    pr.add_argument("--depth", type=float, default=10000.0)
    pr.set_defaults(func=cmd_run)

    pt = sp.add_parser("tune", help="Grid + walk-forward tuning")
    pt.add_argument("--strategy", choices=["v1", "v2"], required=True)
    pt.add_argument("--grid", required=True)
    pt.add_argument("--cache-root", default="data/sim")
    pt.add_argument("--out-dir", default="data/sim/tuning")
    pt.add_argument("--run-id", required=True)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--slippage-bps", type=float, default=5.0)
    pt.add_argument("--fee-taker", type=float, default=0.0,
                    help="Polymarket CLOB takers pay 0%; HL HIP-4 ~3-5bps")
    pt.add_argument("--half-spread", type=float, default=0.005)
    pt.add_argument("--depth", type=float, default=10000.0)
    pt.set_defaults(func=cmd_tune)

    pp = sp.add_parser("report", help="Re-render tuning report from results.jsonl")
    pp.add_argument("--strategy", choices=["v1", "v2"], required=True)
    pp.add_argument("--run-dir", required=True)
    pp.add_argument("--top-k", type=int, default=10)
    pp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
