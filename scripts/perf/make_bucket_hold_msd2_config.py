#!/usr/bin/env python3
"""Emit a `priceBucket` theta-override variant of the live slot config for sweeps.

Variants (v31 priceBucket override only; everything else — v31_pm, v1, the binary
slot — copied verbatim):

- ``hold_msd2`` = C3 + the binary buy-and-hold recipe: mid-hold + take-profit
  exits OFF (exit_safety_d 1.0→0.0, exit_edge_threshold 0.0→1.0) plus an entry
  ``min_safety_d`` floor (0.0→2.0). Keeps the mechanical tilt (vol_lookback 2700 /
  dt 2).
- ``msd2`` = C3 + ONLY the entry ``min_safety_d`` floor (0.0→2.0); the protective
  mid-hold / take-profit exits are KEPT. Isolates the entry-gate effect from the
  exit-disabling, so hold_msd2 vs msd2 vs baseline decomposes the buy-and-hold
  change into "entry gate" and "exits off".

Exact-string textual patch (not a YAML round-trip) so the file stays
comment-for-comment identical to the live config apart from the changed lines;
each substitution must fire exactly once or it aborts (config-drift guard).

Usage:
    python scripts/perf/make_bucket_hold_msd2_config.py --variant hold_msd2 --out /tmp/.../hold_msd2.yaml
    python scripts/perf/make_bucket_hold_msd2_config.py --variant msd2      --out /tmp/.../msd2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Each `old` is unique to the v31 priceBucket override block (8-space indent
# distinguishes it from the base theta / v31_pm override).
_EXIT_SAFETY_OFF = (
    "        exit_safety_d: 1.0             # C3: restored mid-hold exit (was 0.0; inert but defensive)",
    "        exit_safety_d: 0.0             # hold_msd2: buy-and-hold — mid-hold exit OFF",
)
_EXIT_EDGE_OFF = (
    "        exit_edge_threshold: 0.0\n",
    "        exit_edge_threshold: 1.0       # hold_msd2: take-profit exit OFF (edge_held never exceeds 1.0)\n",
)
_MIN_SAFETY_ON = (
    "        min_safety_d: 0.0\n",
    "        min_safety_d: 2.0              # entry safety_d floor (selective entry)\n",
)

_VARIANTS = {
    # buy-and-hold: exits off + entry floor
    "hold_msd2": [_EXIT_SAFETY_OFF, _EXIT_EDGE_OFF, _MIN_SAFETY_ON],
    # entry floor only; keep the protective exits
    "msd2": [_MIN_SAFETY_ON],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="config/strategy.yaml")
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--variant", choices=sorted(_VARIANTS), default="hold_msd2")
    args = ap.parse_args()

    patches = _VARIANTS[args.variant]
    text = Path(args.src).read_text()
    for old, new in patches:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"expected exactly 1 occurrence of:\n{old!r}\nfound {n} — config drifted, aborting")
        text = text.replace(old, new)
    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} (variant={args.variant}, {len(patches)} bucket-override field(s) changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
