"""SHR-105 — calibrate per-IOC clip-size cap from the recorded HL market clip distribution.

ANALYSIS-ONLY.  Places no orders, mutates nothing.  Reads the per-leg/per-kind
ex-own market clip summary CSV produced by ``tools/hl_own_fills_displayed_vs_filled.py``
(SHR-104) and derives the regime-keyed per-IOC clip-cap model that SHR-107 will
implement in ``hftbt_runner``.

Model
-----
Treat each on-venue print as one "clip" — the amount a single marketable IOC
fills in one pass through the book.  The displayed top-of-book depth can be
larger than any individual clip because the book refills between re-fires; the
sim's single-dump error is crossing the *whole displayed ladder at once* while
real takers (and ourselves live) catch only tens-of-shares per clip.

The cap is regime-keyed on **kind** × **book-width** because:
- Binary legs: tight spreads (0.001–0.009) → clips routinely 100–540 sh; cap
  at p90 (≈ 166 sh across the corpus) still lets one IOC fill a large fraction
  of a typical binary touch.
- Bucket legs: wide spreads (0.13–0.34) → clips collapse to tens of shares;
  cap at p90 (≈ 142 sh) already covers the upper bulk; p99 (765 sh) would be
  needed to cover the outlier wide-bucket clips like #1670 which on exit trade
  up to 947 sh, but those are tail events.

The cap must NOT contradict the own-fills displayed-vs-filled data from SHR-104:
- Median ratio_top = 1.00 (we fill ≈ one top-level's worth per IOC) means a
  cap at or near the typical top-of-book size is consistent.
- p90 ratio_top = 3.41 means we sometimes catch several touches' worth — that
  is the book refilling across multiple sub-fills within one clustered IOC, NOT
  a single gigantic clip.  The cap limits *per-clip*, not *per-order* (the
  re-fire mechanism handles multi-clip orders).

Proposed parameter spec (``RunConfig.ioc_clip_cap_shares``):
- Default: ``None`` → disabled (legacy uncapped behaviour; A/B baseline).
- Recommended HL value: ``166`` (market p90 binary) with a separate
  ``ioc_clip_cap_bucket_shares`` of ``142`` for bucket legs.
- A single ``ioc_clip_cap_shares`` scalar applied to both kinds is the
  minimal first cut; regime-split can be added in SHR-107.

Run::

    python tools/hl_clip_cap_calibration.py \\
        --summary docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv

    # With own-fills CSV for cross-reconciliation:
    python tools/hl_clip_cap_calibration.py \\
        --summary docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv \\
        --own-clips-csv docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RegimeSummary:
    """Parsed row from the ex-own market clip summary CSV."""

    group: str  # e.g. "#1591", "ALL_binary", "ALL"
    kind: str  # "binary", "bucket", or "" for ALL
    n: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p99: float
    max: float
    mean: float


@dataclass
class ClipCapSpec:
    """Derived per-IOC clip-cap recommendation for one kind/regime."""

    kind: str  # "binary" or "bucket"
    width_regime: str  # "all" / "tight" / "wide" (from per-leg breakdown)
    n: int
    p50: float
    p90: float
    p99: float
    max: float
    recommended_cap: float  # proposed ``ioc_clip_cap_shares`` value
    rationale: str


# ---------------------------------------------------------------------------
# Pure parsing
# ---------------------------------------------------------------------------


def parse_summary_csv(text: str) -> list[RegimeSummary]:
    """Parse the summary CSV produced by ``hl_own_fills_displayed_vs_filled.py``.

    Returns one :class:`RegimeSummary` per row; rows with ``group == 'ALL'``
    and an empty kind are the pooled total.
    """
    rows: list[RegimeSummary] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        rows.append(
            RegimeSummary(
                group=row["group"],
                kind=row["kind"],
                n=int(row["n"]),
                p10=float(row["p10"]),
                p25=float(row["p25"]),
                p50=float(row["p50"]),
                p75=float(row["p75"]),
                p90=float(row["p90"]),
                p99=float(row["p99"]),
                max=float(row["max"]),
                mean=float(row["mean"]),
            )
        )
    return rows


def aggregate_by_kind(rows: list[RegimeSummary]) -> dict[str, RegimeSummary]:
    """Extract the ALL_binary and ALL_bucket aggregate rows (kind-level summary)."""
    out: dict[str, RegimeSummary] = {}
    for r in rows:
        if r.group in ("ALL_binary", "ALL_bucket"):
            out[r.kind] = r
        elif r.group == "ALL" and r.kind == "":
            out["ALL"] = r
    return out


def per_leg_rows(rows: list[RegimeSummary]) -> list[RegimeSummary]:
    """Return only the per-leg rows (e.g. #1591, #1670)."""
    return [r for r in rows if r.group.startswith("#")]


# ---------------------------------------------------------------------------
# Cap derivation
# ---------------------------------------------------------------------------


# Inferred book-width regime from SHR-104 (median spread per leg).
# Binary legs are consistently tight (0.001–0.009); bucket legs wide (0.13–0.34).
# The per-leg assignment below is hand-coded from the SHR-104 tape analysis.
_LEG_WIDTH_REGIME: dict[str, str] = {
    "#1591": "tight",   # binary, typical spread ~0.001–0.002
    "#1640": "tight",   # binary, typical spread ~0.001
    "#2200": "tight",   # binary, spread 0.001–0.003
    "#2250": "tight",   # binary, spread ~0.001–0.004
    "#1610": "wide",    # bucket, spread ~0.13–0.20
    "#1670": "wide",    # bucket, spread ~0.20–0.40 (doom-loop)
    "#2230": "wide",    # bucket, spread ~0.10–0.20
    "#2280": "wide",    # bucket, spread ~0.13–0.25
}


def derive_caps(
    agg: dict[str, RegimeSummary],
    leg_rows: list[RegimeSummary],
    *,
    cap_percentile: str = "p90",
) -> list[ClipCapSpec]:
    """Derive the recommended clip cap per kind using the chosen percentile.

    ``cap_percentile`` is one of ``"p90"`` or ``"p99"``.  The SHR-105 default
    is p90 (conservative first cut); p99 gives a more permissive cap that still
    covers bulk flow while pruning the rare mega-clip.

    Returns one :class:`ClipCapSpec` per kind present in ``agg``.
    """
    specs: list[ClipCapSpec] = []
    for kind in ("binary", "bucket"):
        if kind not in agg:
            continue
        r = agg[kind]
        cap_val = getattr(r, cap_percentile)
        kind_legs = [lr for lr in leg_rows if lr.kind == kind]
        # Derive a width-regime note from the per-leg data.
        # All binary legs in the corpus are tight; all bucket legs are wide.
        unique_regimes = {_LEG_WIDTH_REGIME.get(lr.group, "unknown") for lr in kind_legs}
        width = next(iter(unique_regimes)) if len(unique_regimes) == 1 else "mixed"
        rationale = (
            f"Market {cap_percentile} = {cap_val:.0f} sh across {r.n:,} clips "
            f"({len(kind_legs)} legs, {width} books). "
            f"Median = {r.p50:.0f} sh; p99 = {r.p99:.0f} sh; max = {r.max:.0f} sh. "
            f"Cap at {cap_percentile} covers the bulk of real clips and still allows "
            f"a single IOC to fill the typical top-of-book resting depth on "
            f"{'tight binary' if kind == 'binary' else 'wide bucket'} books "
            f"while capping the sim's single gigantic dump that real takers never achieve."
        )
        specs.append(
            ClipCapSpec(
                kind=kind,
                width_regime=width,
                n=r.n,
                p50=r.p50,
                p90=r.p90,
                p99=r.p99,
                max=r.max,
                recommended_cap=cap_val,
                rationale=rationale,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Reconciliation with own-fills data from SHR-104
# ---------------------------------------------------------------------------


@dataclass
class OwnClipRow:
    """One row from the per-order displayed-vs-filled CSV (SHR-104 output)."""

    slot: str
    leg: str
    kind: str
    filled: float
    ratio_top: float
    width_bucket: str


def parse_own_clips_csv(text: str) -> list[OwnClipRow]:
    """Parse the per-order SHR-104 CSV.

    The per-order CSV written by ``hl_own_fills_displayed_vs_filled.py`` has
    columns: slot, leg, kind, side, ts_ns, n_prints, filled, limit_px,
    decision_ts_ns, best_bid, best_ask, width, width_bucket, disp_top,
    disp_at_limit, ratio_top, ratio_at_limit.
    We only care about the filled, kind, and width_bucket columns for
    reconciliation.  Also accepts older "coin"/"filled_qty" column names.
    """
    rows: list[OwnClipRow] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        try:
            # The SHR-104 per-order CSV uses column names "leg" and "filled"
            # (not "coin" / "filled_qty" which were the original draft names).
            filled_str = row.get("filled") or row.get("filled_qty") or "0"
            rows.append(
                OwnClipRow(
                    slot=row.get("slot", ""),
                    leg=row.get("leg") or row.get("coin", ""),
                    kind=row.get("kind", ""),
                    filled=float(filled_str or 0),
                    ratio_top=float(row.get("ratio_top", 0) or 0),
                    width_bucket=row.get("width_bucket", ""),
                )
            )
        except (KeyError, ValueError):
            pass
    return rows


def reconcile_caps_with_own_fills(
    specs: list[ClipCapSpec],
    own_rows: list[OwnClipRow],
) -> list[str]:
    """Return a list of reconciliation notes (strings).

    Checks that the proposed cap is consistent with the own-fills data from
    SHR-104 (median ratio_top = 1.00, p10 = 0.40, p90 = 3.41).

    IMPORTANT: ``own_rows`` comes from the per-ORDER CSV (one row per
    clustered IOC), where ``filled`` is the *total* filled across all
    sub-clips in that order — NOT a per-clip size.  The cap applies
    *per-clip*, not per-order.  The per-clip sizes reported in the SHR-104
    research note are:
        binary: median 141 sh, p90 501 sh, max 542 sh
        bucket: median  46 sh, p90 355 sh, max 515 sh

    Consistency check: the cap (per-clip) should be >= the per-clip median
    from SHR-104 above (not the per-order median).  Since the per-order CSV
    only has total-per-order ``filled``, we compute a rough per-clip estimate
    by dividing by ``n_prints`` when available, or flag the semantic
    mismatch explicitly.

    Hard constraints:
    1. cap > 0 (obviously).
    2. cap < sim's single-dump size on the doom-loop bucket (#1670: 516 sh).
       If the cap equals or exceeds that, it does not reduce the overshoot.
    """
    SIM_DOOM_LOOP_DUMP = 516.0  # sh — the sim's single dump SHR-103 documented

    notes: list[str] = []
    for spec in specs:
        kind_rows = [r for r in own_rows if r.kind == spec.kind]
        if not kind_rows:
            notes.append(
                f"{spec.kind}: no own-fill rows to reconcile against; "
                "accept on market-clip evidence alone."
            )
            continue

        # per-order totals (not per-clip; note the semantic mismatch)
        filled_vals = sorted(r.filled for r in kind_rows)
        n = len(filled_vals)
        median_per_order = filled_vals[n // 2]
        max_per_order = filled_vals[-1]

        # Hard constraint: cap must be below the sim's doom-loop single dump
        if spec.recommended_cap >= SIM_DOOM_LOOP_DUMP:
            cap_note = (
                f"WARNING — cap {spec.recommended_cap:.0f} >= sim doom-loop dump "
                f"{SIM_DOOM_LOOP_DUMP:.0f} sh: would NOT reduce the overshoot. "
                "Lower the cap."
            )
        else:
            cap_note = (
                f"OK — cap {spec.recommended_cap:.0f} < sim doom-loop dump "
                f"{SIM_DOOM_LOOP_DUMP:.0f} sh: will reshape large clips."
            )

        notes.append(
            f"{spec.kind}: proposed cap {spec.recommended_cap:.0f} sh. "
            f"Own-fill per-ORDER totals (n={n}): median {median_per_order:.0f} sh, "
            f"max {max_per_order:.0f} sh — these are order totals, not per-clip sizes; "
            f"per-clip sizes (SHR-104 research note) are lower "
            f"({'median 141 sh, max 542 sh' if spec.kind == 'binary' else 'median 46 sh, max 515 sh'}). "
            + cap_note
        )
    return notes


# ---------------------------------------------------------------------------
# Per-width-regime breakdown
# ---------------------------------------------------------------------------


@dataclass
class PerLegStats:
    """Per-leg clip stats for the regime table."""

    group: str
    kind: str
    width_regime: str
    n: int
    p10: float
    p50: float
    p90: float
    p99: float
    max: float


def per_leg_regime_table(leg_rows: list[RegimeSummary]) -> list[PerLegStats]:
    """Enrich each per-leg summary with its inferred book-width regime."""
    out: list[PerLegStats] = []
    for r in leg_rows:
        out.append(
            PerLegStats(
                group=r.group,
                kind=r.kind,
                width_regime=_LEG_WIDTH_REGIME.get(r.group, "unknown"),
                n=r.n,
                p10=r.p10,
                p50=r.p50,
                p90=r.p90,
                p99=r.p99,
                max=r.max,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_row(cells: Sequence[str | float | int], widths: Sequence[int]) -> str:
    parts = []
    for cell, w in zip(cells, widths):
        s = f"{cell:.0f}" if isinstance(cell, float) else str(cell)
        parts.append(s.ljust(w))
    return "| " + " | ".join(parts) + " |"


def _fmt_sep(widths: Sequence[int]) -> str:
    return "| " + " | ".join("-" * w for w in widths) + " |"


def build_report(
    all_rows: list[RegimeSummary],
    own_csv_text: Optional[str] = None,
    *,
    cap_percentile: str = "p90",
) -> str:
    """Build the full calibration report as a markdown string."""
    agg = aggregate_by_kind(all_rows)
    leg_rows = per_leg_rows(all_rows)
    specs = derive_caps(agg, leg_rows, cap_percentile=cap_percentile)
    regime_table = per_leg_regime_table(leg_rows)

    own_rows: list[OwnClipRow] = []
    if own_csv_text:
        own_rows = parse_own_clips_csv(own_csv_text)
    reconcile_notes = reconcile_caps_with_own_fills(specs, own_rows)

    lines: list[str] = []

    # ---- per-leg regime table ----
    lines += [
        "## Per-leg clip-size distribution (market ex-own)",
        "",
        "| leg | kind | width regime | n | p10 | p50 | p90 | p99 | max |",
        "| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for r in sorted(regime_table, key=lambda x: (x.kind, x.group)):
        lines.append(
            f"| {r.group} | {r.kind} | {r.width_regime} | {r.n:,} |"
            f" {r.p10:.0f} | {r.p50:.0f} | {r.p90:.0f} | {r.p99:.0f} | {r.max:.0f} |"
        )
    lines.append("")

    # ---- kind-aggregate table ----
    lines += [
        "## Kind-aggregate summary",
        "",
        "| kind | n | p10 | p25 | p50 | p75 | p90 | p99 | max | mean |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for kind in ("binary", "bucket", "ALL"):
        if kind in agg:
            r = agg[kind]
            lines.append(
                f"| **{r.kind or 'ALL'}** | {r.n:,} | {r.p10:.0f} |"
                f" {r.p25:.0f} | {r.p50:.0f} | {r.p75:.0f} | {r.p90:.0f} |"
                f" {r.p99:.0f} | {r.max:.0f} | {r.mean:.0f} |"
            )
    lines.append("")

    # ---- cap model ----
    lines += ["## Derived clip-cap model", ""]
    for spec in specs:
        lines += [
            f"**{spec.kind}** (`{spec.width_regime}` books)",
            "",
            f"- Recommended cap: **{spec.recommended_cap:.0f} sh** ({cap_percentile})",
            f"- Covers {_coverage_pct(spec, cap_percentile):.1f}% of real clips by count",
            f"- Rationale: {spec.rationale}",
            "",
        ]

    # ---- reconciliation ----
    lines += ["## Reconciliation with SHR-104 own-fills", ""]
    for note in reconcile_notes:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def _coverage_pct(spec: ClipCapSpec, percentile: str) -> float:
    """Return the percentage of clips covered by the cap percentile."""
    pct_map = {"p10": 10.0, "p25": 25.0, "p50": 50.0, "p75": 75.0,
               "p90": 90.0, "p99": 99.0}
    return pct_map.get(percentile, 90.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        required=True,
        metavar="CSV",
        help="Path to the ex-own market clip summary CSV "
        "(docs/research/2026-06-11-hl-market-clips-ex-own-summary.csv).",
    )
    parser.add_argument(
        "--own-clips-csv",
        default=None,
        metavar="CSV",
        help="Optional: path to the per-order SHR-104 displayed-vs-filled CSV "
        "(docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv). "
        "Used for cross-reconciliation only.",
    )
    parser.add_argument(
        "--cap-percentile",
        default="p90",
        choices=["p50", "p75", "p90", "p99"],
        help="Percentile of the market clip distribution to use as the per-IOC "
        "cap.  Default p90 (conservative; covers 90%% of real clips).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="MD",
        help="Write the calibration report as markdown to this path (in addition "
        "to printing to stdout).",
    )
    args = parser.parse_args(argv)

    summary_text = Path(args.summary).read_text()
    all_rows = parse_summary_csv(summary_text)

    own_text: str | None = None
    if args.own_clips_csv:
        own_text = Path(args.own_clips_csv).read_text()

    report = build_report(all_rows, own_text, cap_percentile=args.cap_percentile)
    print(report)

    if args.out:
        Path(args.out).write_text(report)
        print(f"\n(Report written to {args.out})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
