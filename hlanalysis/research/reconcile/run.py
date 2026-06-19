"""CLI: python -m hlanalysis.research.reconcile.run"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _parse_expiry(s: str) -> int:
    """Parse '20260613-0600' or an integer ns timestamp.

    Parameters
    ----------
    s:
        Either an integer nanosecond string or a datetime in ``YYYYMMDD-HHMM`` format.

    Returns
    -------
    Expiry timestamp in nanoseconds.
    """
    try:
        return int(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y%m%d-%H%M").replace(tzinfo=UTC)
        return int(dt.timestamp() * 1e9)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the sim-vs-live reconciliation CLI.

    Parameters
    ----------
    argv:
        Argument list; defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        description="Sim-vs-live reconciliation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hlanalysis.research.reconcile.run \\
    --question-idx 4010 \\
    --expiry 20260613-0600 \\
    --sim-trace /tmp/trace.jsonl \\
    --sim-fills /tmp/fills.parquet \\
    --live-cache-dir /tmp/live_cache \\
    --pull
""",
    )
    parser.add_argument(
        "--question-idx",
        type=int,
        required=True,
        help="HL question/market index",
    )
    parser.add_argument(
        "--expiry",
        required=True,
        help="YYYYMMDD-HHMM or integer ns timestamp",
    )
    parser.add_argument(
        "--sim-trace",
        required=True,
        help="Path to sim decision_trace.jsonl",
    )
    parser.add_argument(
        "--sim-fills",
        required=True,
        help="Path to sim fills.parquet or fills.json",
    )
    parser.add_argument(
        "--live-cache-dir",
        required=True,
        help="Directory to cache/read pulled live data",
    )
    parser.add_argument(
        "--instance-id",
        default="i-0dc4c0abec85a9eda",
        help="EC2 instance ID for SSM pulls",
    )
    parser.add_argument(
        "--strategy-id",
        default="v31",
        help="Engine slot to reconcile in the unified DB (e.g. v31, v1)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull fresh live data from EC2 (requires AWS CLI + credentials)",
    )
    parser.add_argument(
        "--data-root",
        default="../../data",
        help="Path to HL recorded data root (for book parity checks)",
    )
    parser.add_argument(
        "--out",
        help="Write markdown report to this file (also printed to stdout)",
    )
    args = parser.parse_args(argv)

    expiry_ns = _parse_expiry(args.expiry)
    question_idx: int = args.question_idx
    cache_dir = Path(args.live_cache_dir)
    data_root = Path(args.data_root)
    instance_id: str = args.instance_id
    strategy_id: str = args.strategy_id

    # -- Load sim data --
    sim_trace_path = Path(args.sim_trace)
    if not sim_trace_path.exists():
        print(f"ERROR: sim trace not found: {sim_trace_path}", file=sys.stderr)
        sys.exit(1)

    sim_trace_rows: list[dict] = []
    with open(sim_trace_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("question_idx") == question_idx:
                    sim_trace_rows.append(obj)
            except json.JSONDecodeError:
                continue
    sim_trace = pd.DataFrame(sim_trace_rows) if sim_trace_rows else pd.DataFrame()

    sim_fills_path = Path(args.sim_fills)
    if not sim_fills_path.exists():
        print(f"ERROR: sim fills not found: {sim_fills_path}", file=sys.stderr)
        sys.exit(1)

    if sim_fills_path.suffix == ".parquet":
        sim_fills = pd.read_parquet(sim_fills_path)
    elif sim_fills_path.suffix in (".json", ".jsonl"):
        sim_fills = pd.read_json(sim_fills_path, lines=sim_fills_path.suffix == ".jsonl")
    else:
        print(f"ERROR: unsupported sim fills format: {sim_fills_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Filter sim fills to this question
    if "question_idx" in sim_fills.columns:
        sim_fills = sim_fills[sim_fills["question_idx"] == question_idx].copy()

    # -- Pull or load live data --
    from hlanalysis.research.reconcile.pull_live import (  # noqa: PLC0415
        pull_config_hash,
        pull_halts_rejects,
        pull_live_fills,
        pull_live_trace,
        pull_settlement,
    )

    if args.pull:
        print(f"Pulling live data for #{question_idx} ({strategy_id}) from {instance_id}...")
        live_fills = pull_live_fills(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
        live_trace = pull_live_trace(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
        live_settlement = pull_settlement(
            question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id
        )
        _halts = pull_halts_rejects(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
        live_config_hash = pull_config_hash(strategy_id, instance_id=instance_id)
    else:
        # Try to load from cache
        def _load_cache(kind: str) -> str | None:
            p = cache_dir / f"q{question_idx}_{strategy_id}_{kind}.json"
            return p.read_text() if p.exists() else None

        fills_raw = _load_cache("fills")
        trace_raw = _load_cache("trace")
        settlement_raw = _load_cache("settlement")

        if fills_raw is None or trace_raw is None:
            print(
                "ERROR: No cached live data found and --pull not specified.\n"
                f"  Expected cache dir: {cache_dir}\n"
                "  Re-run with --pull to fetch from EC2.",
                file=sys.stderr,
            )
            sys.exit(1)

        fills_rows = json.loads(fills_raw)
        live_fills = pd.DataFrame(fills_rows) if fills_rows else pd.DataFrame()
        trace_rows = json.loads(trace_raw)
        live_trace = pd.DataFrame(trace_rows) if trace_rows else pd.DataFrame()
        live_settlement = json.loads(settlement_raw) if settlement_raw else {}
        live_config_hash = None

    # -- Sim resolved outcome (from sim trace or sim fills) --
    sim_resolved: dict = {}
    if "winner_side" in sim_trace.columns and not sim_trace.empty:
        last = sim_trace.iloc[-1]
        sim_resolved = {
            "winner_side": last.get("winner_side"),
            "resolved_outcome": last.get("resolved_outcome"),
        }

    # -- Run reconciliation --
    from hlanalysis.research.reconcile.reconcile import run_reconcile  # noqa: PLC0415
    from hlanalysis.research.reconcile.report import render_markdown  # noqa: PLC0415

    result = run_reconcile(
        question_idx=question_idx,
        expiry_ns=expiry_ns,
        live_fills=live_fills,
        live_trace=live_trace,
        live_settlement=live_settlement,
        live_config_hash=live_config_hash,
        sim_fills=sim_fills,
        sim_trace=sim_trace,
        sim_resolved=sim_resolved,
        data_root=data_root if data_root.exists() else None,
    )

    md = render_markdown(result)
    print(md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\nReport written to: {out_path}", file=sys.stderr)

    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
