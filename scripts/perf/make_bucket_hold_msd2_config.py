#!/usr/bin/env python3
"""Emit the ``hold_msd2`` variant of the live slot config for the bucket sweep.

``hold_msd2`` = the deployed C3 config with the v31 **priceBucket** theta override
flipped to the binary buy-and-hold recipe (mid-hold + take-profit exits OFF, plus
an entry ``min_safety_d`` floor), keeping the mechanical tilt (vol_lookback 2700 /
dt 2). Only the three exit/entry knobs change; everything else (incl. v31_pm,
v1, the binary slot) is copied verbatim. We do an exact-string textual patch (not
a YAML round-trip) so the file stays comment-for-comment identical to the live
config apart from the three lines — and assert each substitution fires exactly
once so a config drift fails loud.

Usage:
    python scripts/perf/make_bucket_hold_msd2_config.py \
        --in config/strategy.yaml --out /tmp/bucket_sweep/strategy_hold_msd2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (old, new) — each `old` is unique to the v31 priceBucket override block
# (8-space indent distinguishes it from the base theta / v31_pm override).
_PATCHES = [
    (
        "        exit_safety_d: 1.0             # C3: restored mid-hold exit (was 0.0; inert but defensive)",
        "        exit_safety_d: 0.0             # hold_msd2: buy-and-hold — mid-hold exit OFF",
    ),
    (
        "        exit_edge_threshold: 0.0\n",
        "        exit_edge_threshold: 1.0       # hold_msd2: take-profit exit OFF (edge_held never exceeds 1.0)\n",
    ),
    (
        "        min_safety_d: 0.0\n",
        "        min_safety_d: 2.0              # hold_msd2: entry safety_d floor (buy-and-hold selective entry)\n",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="config/strategy.yaml")
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    text = Path(args.src).read_text()
    for old, new in _PATCHES:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"expected exactly 1 occurrence of:\n{old!r}\nfound {n} — config drifted, aborting")
        text = text.replace(old, new)
    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(_PATCHES)} bucket-override fields flipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
