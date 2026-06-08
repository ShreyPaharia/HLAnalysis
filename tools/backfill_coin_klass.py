"""Backfill the coin_klass table in each HL slot's state.db from the recorder's
recorded HL question-metadata parquet.

BACKGROUND (SHR-77)
--------------------
The engine's daily report splits each HL slot's PnL/fills by market class
(priceBinary vs priceBucket) by joining venue fills (which carry only coin "#N")
to the ``coin_klass`` table in state.db.  That table is stamped FORWARD-ONLY at
QuestionMetaEvent ingest — so fills that settled BEFORE the SHR-77 feature
deployed have no mapping and show as "unknown".

SOURCE
------
The recorder persists HL question metadata as Hive-partitioned parquet at:

    <data_root>/venue=hyperliquid/product_type=prediction_binary/mechanism=clob
               /event=question_meta/symbol=Q{question_idx}/**/*.parquet

Each parquet row is a QuestionMetaEvent snapshot.  The fields we consume are:
  - question_idx: int
  - named_outcome_idxs: list[int]
  - keys/values: parallel string arrays; values[keys.index("class")] = klass

The coin encoding is deterministic and identical to the live engine's formula
(market_state.py line 325-327):

    for o in sorted(named_outcome_idxs):
        for s in (0, 1):
            coin = f"#{10 * o + s}"

This is COMPLETE and DURABLE: the recorder re-emits the metadata on every
restart, the parquet is never pruned, and the formula is embedded in the venue
adapter.  No numeric heuristic is needed or permitted.

SCOPE
-----
Only HL slots are processed (venue == "hyperliquid").  PM fills are binary by
construction and the PM leg tokens are CLOB token-ids, not "#N" coins; they
don't need this table.

USAGE
-----
Dry run (default) — prints the plan, writes nothing:
  uv run python tools/backfill_coin_klass.py

Execute:
  uv run python tools/backfill_coin_klass.py --apply

On the EC2 box (run inside the venv with env sourced for creds):
  set -a; . /etc/hl-engine/env
  .venv/bin/python tools/backfill_coin_klass.py --data-root /opt/hl-recorder/data --apply
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hlanalysis.engine.state import StateDAL


# ---------------------------------------------------------------------------
# Pure core — no IO, fully unit-testable
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Question:
    question_idx: int
    named_outcome_idxs: list[int]
    klass: str


def coin_klass_rows(
    questions,
) -> list[tuple[str, str, int]]:
    """Expand each question into (coin, klass, question_idx) triples.

    ``questions`` is any iterable of objects with:
      - .question_idx: int
      - .named_outcome_idxs: list[int]
      - .klass: str

    Skips questions with empty klass or empty named_outcome_idxs — neither
    produces a useful row and both would write un-joinable data.

    Returns a flat list; order is question-order then outcome-order then
    side-order (0=YES, 1=NO).
    """
    out: list[tuple[str, str, int]] = []
    for q in questions:
        klass = q.klass
        if not klass:
            continue
        outcomes = sorted(q.named_outcome_idxs)
        if not outcomes:
            continue
        for o in outcomes:
            for s in (0, 1):
                out.append((f"#{10 * o + s}", klass, q.question_idx))
    return out


def backfill_from_questions(questions, dal: "StateDAL") -> int:
    """Write all (coin, klass, question_idx) rows to ``dal`` via set_coin_klass.

    Idempotent: set_coin_klass is an upsert, so re-running is safe.
    Returns the number of (coin, klass) pairs written (including re-writes).
    """
    rows = coin_klass_rows(questions)
    for coin, klass, question_idx in rows:
        dal.set_coin_klass(coin=coin, klass=klass, question_idx=question_idx)
    return len(rows)


# ---------------------------------------------------------------------------
# Parquet reader — converts recorder parquet into question-like objects
# ---------------------------------------------------------------------------

def parse_question_meta_parquet(glob_pattern: str) -> list[_Question]:
    """Read HL question_meta parquet files matching ``glob_pattern``.

    Deduplicates by question_idx (the recorder re-emits on every engine
    restart), keeping the first-seen row.  Rows without a "class" key in
    their keys/values arrays are skipped — they can't be mapped to a klass.

    Returns a list of _Question objects ready for coin_klass_rows().
    """
    import pyarrow.parquet as pq

    paths = sorted(_glob.glob(glob_pattern, recursive=True))
    if not paths:
        return []

    # Read each file independently and select only the columns we need. A
    # whole-corpus concat (recorder.read.read_recorded) raises ArrowTypeError
    # because the recorder's parquet schemas drift across files — e.g.
    # fallback_outcome_idx is int32 in some files and int64 in others, and
    # pyarrow's promote_options="default" won't merge those. A per-file read of
    # a fixed, type-stable column subset (question_idx + the list columns we
    # actually consume) sidesteps the merge entirely.
    want = ("question_idx", "named_outcome_idxs", "keys", "values")
    seen: set[int] = set()
    out: list[_Question] = []
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
            cols = [c for c in want if c in set(pf.schema_arrow.names)]
            if "question_idx" not in cols:
                continue
            d = pf.read(columns=cols).to_pydict()
        except Exception:  # noqa: BLE001 — a single unreadable file must not abort the scan
            continue
        n = len(d.get("question_idx", []))
        for i in range(n):
            qidx = int(d["question_idx"][i])
            if qidx in seen:
                continue
            keys_i = d.get("keys", [None] * n)[i]
            values_i = d.get("values", [None] * n)[i]
            keys_raw = list(keys_i) if keys_i is not None else []
            values_raw = list(values_i) if values_i is not None else []
            kv = dict(zip(keys_raw, values_raw, strict=False))
            # The recorder stores "class" directly; some older rows bury it
            # inside the question_description field (pipe-delimited). Check both.
            klass = kv.get("class", "")
            if not klass:
                desc = kv.get("question_description", "")
                for part in desc.split("|"):
                    if part.startswith("class:"):
                        klass = part[len("class:"):]
                        break
            if not klass:
                continue
            named_i = d.get("named_outcome_idxs", [None] * n)[i]
            named = [int(x) for x in (named_i if named_i is not None else [])]
            seen.add(qidx)
            out.append(_Question(question_idx=qidx, named_outcome_idxs=named, klass=klass))
    return out


# ---------------------------------------------------------------------------
# IO shell — config loading and per-slot DB writing
# ---------------------------------------------------------------------------

_HL_QM_GLOB = (
    "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
    "/event=question_meta/symbol=Q*/**/*.parquet"
)


def _hl_slot_aliases(deploy_cfg, strategies_cfg) -> list[str]:
    """Return account_aliases for HL slots only (venue == hyperliquid)."""
    from hlanalysis.engine.config import HyperliquidAccount
    hl = {
        alias for alias, acct in deploy_cfg.accounts.items()
        if isinstance(acct, HyperliquidAccount)
    }
    return [s.account_alias for s in strategies_cfg.strategies if s.account_alias in hl]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--strategy-config", type=Path, default=Path("config/strategy.yaml"),
        help="path to strategy.yaml (default: config/strategy.yaml)",
    )
    ap.add_argument(
        "--deploy-config", type=Path, default=Path("config/deploy.yaml"),
        help="path to deploy.yaml (default: config/deploy.yaml)",
    )
    ap.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="recorder data root containing venue=hyperliquid/... partitions (default: data)",
    )
    ap.add_argument(
        "--alias", type=str, default=None,
        help="restrict to a specific HL slot alias; default processes all HL slots",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="write to the DB (default: dry run — prints the plan, writes nothing)",
    )
    args = ap.parse_args(argv)

    from hlanalysis.engine.config import load_deploy_config, load_strategies_config
    from hlanalysis.engine.state import StateDAL

    try:
        deploy_cfg = load_deploy_config(args.deploy_config)
    except ValueError as e:
        # env vars not set in dry-run / offline context: only a blocker if --apply
        if args.apply:
            print(f"ERROR: could not load deploy config: {e}", file=sys.stderr)
            return 1
        # In dry-run mode, still show what we'd do — load strategies only.
        print(f"WARNING: deploy config env vars missing ({e}); showing source scan only", file=sys.stderr)
        deploy_cfg = None

    strategies_cfg = load_strategies_config(args.strategy_config)

    # -- discover HL question-meta parquet --
    glob_pattern = str(args.data_root / _HL_QM_GLOB)
    print(f"scanning: {glob_pattern}", file=sys.stderr)
    questions = parse_question_meta_parquet(glob_pattern)
    print(f"found {len(questions)} distinct HL questions in recorder data", file=sys.stderr)

    if not questions:
        print("no question_meta parquet found — nothing to backfill", file=sys.stderr)
        print(
            "  (check --data-root; expected path:\n"
            f"   {glob_pattern})",
            file=sys.stderr,
        )
        return 0

    # Show coin expansion summary
    rows = coin_klass_rows(questions)
    by_klass: dict[str, int] = {}
    for _, klass, _ in rows:
        by_klass[klass] = by_klass.get(klass, 0) + 1
    print(f"expands to {len(rows)} (coin, klass) pairs: {by_klass}", file=sys.stderr)

    if not args.apply:
        print("\n(dry run — pass --apply to write to state.db)", file=sys.stderr)
        # Show a sample of what would be written
        sample = rows[:10]
        for coin, klass, qidx in sample:
            print(f"  would write: coin={coin!r} klass={klass!r} question_idx={qidx}", file=sys.stderr)
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more", file=sys.stderr)
        return 0

    if deploy_cfg is None:
        print("ERROR: --apply requires a loadable deploy config (env vars must be set)", file=sys.stderr)
        return 1

    # Determine which HL slots to process
    all_hl = _hl_slot_aliases(deploy_cfg, strategies_cfg)
    if args.alias:
        if args.alias not in all_hl:
            print(f"ERROR: alias {args.alias!r} is not an HL slot (known HL: {all_hl})", file=sys.stderr)
            return 1
        aliases = [args.alias]
    else:
        aliases = all_hl

    print(f"writing to HL slots: {aliases}", file=sys.stderr)
    total = 0
    for alias in aliases:
        db_path = Path(deploy_cfg.state_db_path_for(alias))
        if not db_path.exists():
            print(f"  [{alias}] state.db not found at {db_path} — skipping", file=sys.stderr)
            continue
        dal = StateDAL(db_path)
        dal.run_migrations()
        n = backfill_from_questions(questions, dal)
        total += n
        print(f"  [{alias}] wrote {n} coin_klass rows to {db_path}", file=sys.stderr)

    print(f"\ndone: {total} total coin_klass rows written across {len(aliases)} slot(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
