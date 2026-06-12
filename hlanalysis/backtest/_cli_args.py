"""Argparse construction helpers for the ``hl-bt`` CLI.

Contains :func:`_add_run_config_args` (the shared run-config flag group) and
per-subcommand parser builders (:func:`_build_run_parser`,
:func:`_build_tune_parser`, etc.).  No command-handler logic lives here.
"""
from __future__ import annotations

import argparse

from ._cli_commands import (
    cmd_fetch,
    cmd_report,
    cmd_run,
    cmd_strategies,
    cmd_trace,
    cmd_tune,
)
from .core.source_config import PM_FLAVORS


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    """Declare the run-config args shared verbatim by ``run`` and ``tune``
    (fee / tick / lot / slippage / scan-cadence). Consumed by
    :func:`_run_config_from_args`."""
    parser.add_argument("--scanner-interval-seconds", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--fee-taker", type=float, default=0.0)
    parser.add_argument(
        "--fee-model",
        choices=["flat", "pm_binary"],
        default="flat",
        help="Binary-leg fee model. 'flat' uses --fee-taker as constant %% of "
        "notional (HL/synthetic). 'pm_binary' uses Polymarket's curve "
        "fee = qty * --fee-rate * p * (1-p).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.07,
        help="feeRate for --fee-model pm_binary. PM crypto = 0.07; sports = 0.03; "
        "politics/tech/finance/mentions = 0.04; econ/culture/weather = 0.05.",
    )
    parser.add_argument("--tick-size", type=float, default=0.001)
    parser.add_argument("--lot-size", type=float, default=1.0)
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Optional explicit fill-size cap in lots (SHR-79). Default=None "
        "(unlimited — the real recorded book governs fill size via "
        "partial_fill_exchange). Use e.g. --depth 500 to hard-cap fills.",
    )
    parser.add_argument(
        "--order-latency-ms",
        type=float,
        default=50.0,
        help="Constant order round-trip latency in milliseconds (SHR-79). "
        "Default=50 ms (empirical HL HIP-4 median). Set to 0 for legacy "
        "zero-latency behaviour.",
    )
    # SHR-95: event-driven scan mode (mirrors live engine cadence).
    parser.add_argument(
        "--scan-mode",
        choices=["fixed", "event"],
        default=None,
        dest="scan_mode",
        help="Scan cadence mode. Unset → 'fixed' for ad-hoc runs (legacy "
        "behaviour) but 'event' for --slot runs (mirror the live engine's "
        "cadence). 'fixed' evaluates every --scanner-interval-seconds. "
        "'event' evaluates on each book/reference update, clamped between "
        "--scan-min-interval-seconds (floor) and --scan-max-interval-seconds "
        "(ceiling). Pass explicitly to override the --slot default.",
    )
    parser.add_argument(
        "--scan-min-interval-seconds",
        type=float,
        default=0.2,
        dest="scan_min_interval_seconds",
        help="(event scan mode only) Minimum time between consecutive scans "
        "in seconds. Mirrors live scan_min_interval_seconds=0.2. Default=0.2.",
    )
    parser.add_argument(
        "--scan-max-interval-seconds",
        type=float,
        default=2.0,
        dest="scan_max_interval_seconds",
        help="(event scan mode only) Maximum time between consecutive scans "
        "(idle-backoff ceiling) in seconds. Mirrors live scan_max_interval_seconds=2.0. "
        "Default=2.0.",
    )
    parser.add_argument(
        "--min-inter-order-seconds",
        type=float,
        default=0.0,
        dest="min_inter_order_seconds",
        help="(SHR-79/SHR-89) Minimum inter-order re-fire floor in seconds: after "
        "dispatching an IOC for a leg, suppress further orders on that leg until "
        "this many seconds elapse (live serializes one order per leg in flight). "
        "Throttles wide-bucket exit churn to live cadence. Measured HL floor "
        "~0.75s (live min 0.73s). Default=0.0 disables it (legacy no-floor A/B arm).",
    )
    parser.add_argument(
        "--ioc-fleeting-persistence-seconds",
        type=float,
        default=0.0,
        dest="ioc_fleeting_persistence_seconds",
        help="(SHR-89b) Wall-clock persistence IOC marketability re-check. When > 0 "
        "and ioc_marketability_recheck is enabled, veto IOC submits where the ask/bid "
        "level just appeared in the current snapshot (absent in the prior one) AND has "
        "been present for fewer than this many seconds. Catches HL's burst-sampled book "
        "where the SHR-94 latency window never fires (snapshots are seconds apart). "
        "Default=0.0 disables (legacy behaviour).",
    )


def _build_strategies_parser(sp) -> None:
    ps = sp.add_parser("strategies", help="List registered strategy ids")
    ps.set_defaults(func=cmd_strategies)


def _build_run_parser(sp) -> None:
    pr = sp.add_parser("run", help="Run one strategy config across discovered questions")
    pr.add_argument("--strategy", default=None,
                    help="Registry strategy id. Required unless --slot is given "
                         "(--slot derives it from the live config).")
    pr.add_argument(
        "--data-source",
        required=True,
        choices=["synthetic", "polymarket", "hl_hip4", "pm_nba"],
    )
    pr.add_argument("--config", default=None,
                    help="JSON file of param dict. Mutually exclusive with --slot.")
    pr.add_argument(
        "--slot", default=None,
        help="(SHR-99) Run from a live strategy.yaml slot by account_alias "
             "(e.g. v31). Sources the EXACT live decision config via the engine "
             "config builders — no hand-written JSON to drift. Sets --strategy.",
    )
    pr.add_argument(
        "--slot-config", default="config/strategy.yaml", dest="slot_config",
        help="Path to the live strategy config for --slot (default "
             "config/strategy.yaml).",
    )
    pr.add_argument(
        "--slot-class", default=None, dest="slot_class",
        help="(--slot only) Select a per-class override config (e.g. priceBucket) "
             "matching the engine's per-question.klass resolution. Omit for the "
             "slot default. Run one class per invocation (one σ cadence).",
    )
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--start", default=None)
    pr.add_argument("--end", default=None)
    _add_run_config_args(pr)
    pr.add_argument("--skip-markets", type=int, default=0)
    pr.add_argument("--max-markets", type=int, default=None)
    pr.add_argument(
        "--cache-root",
        default=None,
        help="Override cache/data root (env: HLBT_PM_CACHE_ROOT / HLBT_HL_DATA_ROOT).",
    )
    pr.add_argument(
        "--pm-flavor",
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
    )
    pr.add_argument(
        "--hedge-data-path",
        default=None,
        help="Path to Binance perp/spot kline JSON for the hedge leg (v5 only). "
        "When omitted, hedge_enabled in --config is ignored.",
    )
    pr.add_argument(
        "--kind",
        choices=["binary", "bucket", "both"],
        default="both",
        help="Filter discovery to this question class. "
        "For hl_hip4: binary→priceBinary, bucket→priceBucket. "
        "For polymarket: pass-through.",
    )
    pr.add_argument(
        "--ref-source",
        choices=["hl_perp", "binance_perp"],
        default="hl_perp",
        help="(hl_hip4 only) Reference-price feed for σ + p_model. "
        "`hl_perp` reads HL BBO/mark; `binance_perp` reads Binance perp BBO/mark.",
    )
    pr.add_argument(
        "--ref-event",
        choices=["bbo", "mark"],
        default=None,
        help="(hl_hip4 only) Which reference price stream feeds σ/p_model. "
        "`mark` (live-faithful default, SHR-97) matches the LIVE engine "
        "(reference_sigma_source default = mark). `bbo` reads bid/ask mid "
        "(bounces more → higher σ). Default: config-derived (mark).",
    )
    pr.add_argument(
        "--reference-ticks",
        choices=["bars", "raw"],
        default=None,
        dest="reference_ticks",
        help="(hl_hip4 only) Reference-tick mode (SHR-93). `raw` (live-faithful "
        "default, SHR-97) emits one event per recorded tick and lets the shared "
        "MarketState bucket them, making last_mark the instantaneous raw tick "
        "price (live-parity). `bars` pre-buckets raw ticks into OHLC bars "
        "(legacy A/B override). Default: config-derived (raw).",
    )
    pr.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo", "klines_1s"],
        default="klines",
        help="(polymarket only) Reference-feed source. `klines` (default) reads "
        "cached 1m Binance klines (12-month corpus). `binance_bbo` reads "
        "recorded Binance BBO ticks and buckets to vol_sampling_dt_seconds "
        "(BBO-overlap window only). `klines_1s` pulls genuine Binance 1s klines "
        "(cached under btc_klines_1s/) and buckets to vol_sampling_dt_seconds — "
        "the on-demand-klines counterpart to binance_bbo for the ref-equivalence "
        "experiment.",
    )
    pr.add_argument(
        "--pm-binance-bbo-product-type",
        choices=["perp", "spot"],
        default="perp",
        dest="pm_binance_bbo_product_type",
        help="(polymarket binance_bbo only) Binance product type for the BBO "
        "reference feed. `perp` (default) reads Binance PERP BBO ticks. `spot` "
        "reads Binance SPOT BBO ticks, matching PM's settlement instrument "
        "(Binance SPOT 1m close). Spot ticks use local_recv_ts since Binance "
        "SPOT bookTicker does not provide exchange_ts.",
    )
    pr.add_argument(
        "--pm-book-source",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="(polymarket only) Fill-book source. `synthetic` (default) builds "
        "a flat 1-level book per trade print + `1−p` parity. `recorded` feeds "
        "the real multi-level L2 `book_snapshot` parquet per leg (HL parity; "
        "coverage from 2026-05-27).",
    )
    pr.add_argument(
        "--pm-liquidity-profile",
        default=None,
        dest="pm_liquidity_profile",
        help="(polymarket synthetic mode only) Path to a JSON liquidity-profile "
        "produced by scripts/calibrate_pm_liquidity.py. When supplied, the "
        "synthetic book builder uses per-price-bucket half_spread/depth from "
        "the profile instead of the flat 0.005/10000 defaults.",
    )
    pr.add_argument(
        "--reference-warmup-seconds",
        type=int,
        default=None,
        dest="reference_warmup_seconds",
        help="(hl_hip4 only) Reference warm-up prefix in seconds (SHR-92). "
        "When > 0, pre-loads reference rows from [start - N, start) into "
        "MarketState so σ is warm at market open. Default: auto-derived from "
        "max(vol_lookback_seconds) across param classes. Pass 0 to disable.",
    )
    pr.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for independent markets "
                         "(default 1 = serial). Use up to #cores for big runs.")
    pr.add_argument("--no-cache", "--fresh", dest="no_cache", action="store_true",
                    help="Disable the event-array cache for this run (it is "
                         "default-ON) and force a fresh build. Use when you "
                         "suspect a stale/poisoned cached entry.")
    pr.add_argument("--cache-event-arrays", action="store_true",
                    help="(deprecated, no-op) The event-array cache is now "
                         "default-ON; this flag is kept for back-compat.")
    pr.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore cached event arrays and rebuild them once "
                         "(then repopulate the cache).")
    pr.set_defaults(func=cmd_run)


def _build_fetch_parser(sp) -> None:
    pf = sp.add_parser("fetch", help="Populate polymarket cache from Gamma + CLOB")
    pf.add_argument(
        "--data-source",
        choices=["polymarket"],
        default="polymarket",
        help="Only polymarket supports fetch; HL HIP-4 is consumed from the recorder directly.",
    )
    pf.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    pf.add_argument("--end", required=True, help="ISO date YYYY-MM-DD")
    pf.add_argument("--kind", choices=["binary", "bucket", "both"], default="both")
    pf.add_argument("--cache-root", default=None, help="Override env HLBT_PM_CACHE_ROOT")
    pf.add_argument(
        "--pm-flavor",
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
    )
    pf.add_argument("--min-trades", type=int, default=30)
    pf.add_argument("--min-volume-usd", type=float, default=1000.0)
    pf.add_argument("--refresh", action="store_true")
    pf.set_defaults(func=cmd_fetch)


def _build_tune_parser(sp) -> None:
    pt = sp.add_parser("tune", help="Walk-forward parallel grid sweep")
    pt.add_argument("--strategy", required=True)
    pt.add_argument(
        "--data-source",
        required=True,
        choices=["polymarket", "hl_hip4", "pm_nba"],
        help="Synthetic is not supported by tune.",
    )
    pt.add_argument("--grid", required=True, help="YAML file with `grids.<strategy>` + `run.*`")
    pt.add_argument("--out-dir", default="data/sim/tuning")
    pt.add_argument("--run-id", required=True)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--cache-root", default=None)
    pt.add_argument(
        "--pm-flavor",
        choices=sorted(PM_FLAVORS),
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load. "
        "Default: btc_updown.",
    )
    pt.add_argument("--start", default=None)
    pt.add_argument("--end", default=None)
    _add_run_config_args(pt)
    pt.add_argument("--skip-markets", type=int, default=0)
    pt.add_argument("--max-markets", type=int, default=None)
    pt.add_argument("--top-k", type=int, default=10)
    pt.add_argument("--no-cache", "--fresh", dest="no_cache", action="store_true",
                    help="Disable the event-array cache for this sweep (it is "
                         "default-ON) and force fresh builds.")
    pt.add_argument("--cache-event-arrays", action="store_true",
                    help="(deprecated, no-op) The event-array cache is now "
                         "default-ON; kept for back-compat.")
    pt.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore cached event arrays and rebuild them once "
                         "(then reuse across the sweep).")
    pt.add_argument(
        "--kind",
        choices=["binary", "bucket", "both"],
        default="both",
        help="(polymarket) Filter discovery to this question kind. "
        "Avoids mixing binary and bucket markets in one walk-forward.",
    )
    pt.add_argument(
        "--pm-liquidity-profile",
        default=None,
        dest="pm_liquidity_profile",
        help="(polymarket synthetic mode only) Path to a JSON liquidity-profile "
        "produced by scripts/calibrate_pm_liquidity.py. When supplied, the "
        "synthetic book builder uses per-price-bucket half_spread/depth from "
        "the profile instead of the flat 0.005/10000 defaults.",
    )
    pt.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo", "klines_1s"],
        default="klines",
        help="(polymarket only) Reference-feed source for the sweep. `klines` "
        "(default) = cached 1m Binance klines. `klines_1s` = genuine Binance 1s "
        "klines (cached under <asset>_klines_1s/) bucketed to the per-cell "
        "vol_sampling_dt_seconds — use with a dt<60 grid for live-parity. "
        "`binance_bbo` = recorded Binance BBO ticks.",
    )
    pt.add_argument(
        "--pm-book-source",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="(polymarket only) Fill-book source for the sweep. `synthetic` "
        "(default) builds a flat 1-level book per trade print + `1−p` parity "
        "(calibratable via --pm-liquidity-profile). `recorded` feeds real L2.",
    )
    pt.add_argument(
        "--reference-warmup-seconds",
        type=int,
        default=None,
        dest="reference_warmup_seconds",
        help="(hl_hip4 only) Reference warm-up prefix in seconds (SHR-92). "
        "Override the per-cell auto-derivation from vol_lookback_seconds. "
        "Pass 0 to disable warm-up entirely.",
    )
    pt.add_argument(
        "--reference-ticks",
        choices=["bars", "raw"],
        default=None,
        dest="reference_ticks",
        help="(hl_hip4 only) Reference-tick mode (SHR-93). `raw` (live-faithful "
        "default, SHR-97) emits one event per recorded tick for live-parity. "
        "`bars` pre-buckets raw ticks — the legacy A/B override. "
        "Default: config-derived (raw).",
    )
    # Hedge leg flags (v5_delta_hedged only; safe to pass for other strategies
    # since hedge_enabled defaults to False when --hedge-data-path is omitted).
    pt.add_argument(
        "--hedge-data-path",
        default=None,
        help="Path to Binance perp/spot kline JSON for the hedge leg (v5 only).",
    )
    pt.add_argument("--hedge-symbol", default="BTC-PERP")
    pt.add_argument("--hedge-tick-size", type=float, default=0.1)
    pt.add_argument("--hedge-lot-size", type=float, default=0.001)
    pt.add_argument("--hedge-slippage-bps", type=float, default=15.0)
    pt.add_argument("--hedge-fee-bps", type=float, default=1.0)
    pt.add_argument("--hedge-half-spread-bps", type=float, default=1.0)
    pt.set_defaults(func=cmd_tune)


def _build_trace_parser(sp) -> None:
    ptr = sp.add_parser("trace", help="Per-question diagnostic trace plot")
    ptr.add_argument("--run-dir", required=True)
    ptr.add_argument("--question-id", required=True)
    ptr.add_argument("--out", default=None, help="Defaults to <run-dir>/traces/<question-id>.html")
    ptr.set_defaults(func=cmd_trace)


def _build_report_parser(sp) -> None:
    prp = sp.add_parser("report", help="Re-render tuning report from results.jsonl")
    prp.add_argument("--strategy", required=True)
    prp.add_argument("--run-dir", required=True)
    prp.add_argument("--top-k", type=int, default=10)
    prp.set_defaults(func=cmd_report)
