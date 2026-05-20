"""Compare two fills.parquet outputs ignoring random cloid (UUID per run)."""
from __future__ import annotations

import argparse
import sys

import pyarrow.parquet as pq


_IGNORE = {"cloid"}
_STR_COLS = {"side", "symbol", "question_id", "resolved_outcome", "entry_edge_chosen_side"}


def compare(a: str, b: str) -> tuple[bool, list[str]]:
    ta = pq.read_table(a)
    tb = pq.read_table(b)
    diffs: list[str] = []
    if ta.num_rows != tb.num_rows:
        diffs.append(f"row count: {ta.num_rows} vs {tb.num_rows}")
        return False, diffs
    if set(ta.column_names) != set(tb.column_names):
        diffs.append(f"columns differ: {set(ta.column_names) ^ set(tb.column_names)}")
        return False, diffs
    for col in ta.column_names:
        if col in _IGNORE:
            continue
        la = ta[col].to_pylist()
        lb = tb[col].to_pylist()
        if col in _STR_COLS:
            if la != lb:
                # Locate first diff
                for i, (x, y) in enumerate(zip(la, lb)):
                    if x != y:
                        diffs.append(f"{col} row {i}: {x!r} vs {y!r}")
                        break
            continue
        # Numeric: tolerate NaN/None and small fp error
        for i, (x, y) in enumerate(zip(la, lb)):
            if x is None and y is None:
                continue
            if x is None or y is None:
                diffs.append(f"{col} row {i}: {x} vs {y}")
                break
            xf, yf = float(x), float(y)
            if xf != xf and yf != yf:  # NaN == NaN
                continue
            if abs(xf - yf) > 1e-6:
                diffs.append(f"{col} row {i}: {xf} vs {yf} (diff {xf-yf})")
                break
    return (len(diffs) == 0), diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    ok, diffs = compare(args.a, args.b)
    if ok:
        print(f"EQUIVALENT: {args.a} == {args.b} (cloid ignored)")
        return 0
    print(f"DIFFER: {args.a} vs {args.b}")
    for d in diffs:
        print(f"  {d}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
