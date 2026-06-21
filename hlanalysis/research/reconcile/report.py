"""Render reconciliation results as markdown."""

from __future__ import annotations

from datetime import UTC, datetime

from hlanalysis.research.reconcile.reconcile import ReconcileResult, attributable_gaps


def _fmt_ns(ts_ns: int) -> str:
    """Format a nanosecond timestamp as a human-readable UTC string."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_float(val: float | None, decimals: int = 4) -> str:
    """Format a float or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        import math

        if math.isnan(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{val:.{decimals}f}"


def render_markdown(result: ReconcileResult) -> str:
    """Render a ReconcileResult as a markdown report card.

    Parameters
    ----------
    result:
        The full reconciliation result from ``run_reconcile``.

    Returns
    -------
    Markdown-formatted string with all 4 layer sections.
    """
    lines: list[str] = []

    # Header
    expiry_str = _fmt_ns(result.expiry_ns) if result.expiry_ns else "unknown"
    verdict_badge = result.verdict
    lines.append(f"# Sim-vs-Live Reconciliation: question #{result.question_idx}")
    lines.append(f"**Verdict: {verdict_badge}** | expiry: {expiry_str}")
    lines.append("")

    if result.fail_reasons:
        lines.append("**Failure reasons:**")
        for reason in result.fail_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # Layer 0
    lines.append("## Layer 0: Preconditions")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|-------|--------|")
    lines.append(f"| Config hash | {result.layer0.config_hash_match} |")
    lines.append(f"| Question identity | {result.layer0.question_identity_match} |")
    lines.append(f"| Window overlap | {result.layer0.window_match} |")
    lines.append(f"| **Overall** | **{result.layer0.overall}** |")
    lines.append("")

    # Layer 1
    l1 = result.layer1
    lines.append("## Layer 1: Decisions")
    lines.append("")
    lines.append(
        f"Match rate: {l1.match_rate:.1%} | Classification: {l1.classification} | "
        f"Aligned buckets: {l1.n_aligned} / live={l1.n_live_buckets} sim={l1.n_sim_buckets}"
    )
    lines.append("")
    lines.append(
        f"Decision events (non-hold): live={l1.n_live_events} sim={l1.n_sim_events} "
        "— the enter/exit decisions the match rate must verify (holds dominate the "
        "trace and no longer mask a divergent event within a bucket)."
    )
    lines.append("")

    if l1.first_divergence is not None:
        fd = l1.first_divergence
        ts_str = _fmt_ns(fd.ts_ns)
        rel_str = f"{fd.rel_diff:.4f}" if fd.rel_diff is not None else "N/A"
        lines.append(
            f"First divergence: `{ts_str}` field=`{fd.field}` live={fd.live_val} sim={fd.sim_val} rel_diff={rel_str}"
        )
        lines.append("")

    if not l1.diff_table.empty:
        show = l1.diff_table.head(10)
        cols = [c for c in show.columns if c != "bucket_ns"]
        header_cols = ["bucket_ns"] + cols
        lines.append("| " + " | ".join(str(c) for c in header_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
        for _, row in show.iterrows():
            bucket_str = _fmt_ns(int(row["bucket_ns"])) if "bucket_ns" in row else ""
            vals = [bucket_str]
            for c in cols:
                v = row.get(c)
                if isinstance(v, float):
                    vals.append(_fmt_float(v, 4))
                else:
                    vals.append(str(v) if v is not None else "N/A")
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # Layer 2
    l2 = result.layer2
    lines.append("## Layer 2: Fills")
    lines.append("")
    bk_str = f"{l2.book_parity_pct:.1%}" if l2.book_parity_pct is not None else "N/A"
    lines.append(
        f"Book parity: {bk_str} | Gap classification: {l2.gap_classification} | "
        f"Live fills: {l2.n_live_fills} | Sim fills: {l2.n_sim_fills}"
    )
    lines.append("")

    if not l2.episode_table.empty:
        lines.append(
            "| Episode | Status | Side | Live size | Sim size | Size diff | "
            "Live VWAP | Sim VWAP | VWAP diff | Latency (s) |"
        )
        lines.append(
            "|---------|--------|------|-----------|----------|-----------|-----------|----------|-----------|-------------|"
        )
        for i, row in enumerate(l2.episode_table.to_dict("records")):
            lat = row.get("latency_ns")
            lat_str = _fmt_float(lat / 1e9, 1) if isinstance(lat, (int, float)) and lat == lat else "N/A"
            lines.append(
                f"| {i + 1} "
                f"| {row.get('match_status', 'N/A')} "
                f"| {row.get('live_side', 'N/A')} "
                f"| {_fmt_float(row.get('live_size'), 4)} "
                f"| {_fmt_float(row.get('sim_size'), 4)} "
                f"| {_fmt_float(row.get('size_diff'), 4)} "
                f"| {_fmt_float(row.get('live_vwap'), 4)} "
                f"| {_fmt_float(row.get('sim_vwap'), 4)} "
                f"| {_fmt_float(row.get('vwap_diff'), 4)} "
                f"| {lat_str} |"
            )
        lines.append("")

    # Layer 2.5: reference-feed coverage
    if result.reference_gaps:
        gaps = result.reference_gaps
        # SHR-149: only gaps coinciding with GLOBAL ingest silence are real outages
        # attributable to data. A gap with other feeds ticking is a calm/illiquid
        # market on one symbol, not a recording outage.
        attributable = attributable_gaps(gaps)
        benign = [g for g in gaps if not g.global_silence]
        attr_total_s = sum(g.gap_seconds for g in attributable)
        # The gap that actually matters is the one straddling the first decision
        # divergence — that is where a stale reference desynced the entry gate.
        div_ns = result.layer1.first_divergence.ts_ns if result.layer1.first_divergence else None
        culprit = next((g for g in attributable if div_ns is not None and g.start_ns <= div_ns <= g.end_ns), None)

        lines.append("## Reference-feed coverage")
        lines.append("")
        if attributable:
            lines.append(
                f"⚠️ {len(attributable)} outage gap(s) (Σ {attr_total_s / 60:.0f} min) coinciding with "
                "**global ingest silence** in the recorded reference feed. During such a gap the sim "
                "holds a stale reference; where the price also moves, the entry gate desyncs from live "
                "— so divergence is **likely data-caused, not a strategy/harness fault**."
            )
        else:
            lines.append(
                "✅ No outage gaps: every reference gap over the window had other feeds still ticking "
                "(global ingest alive) — a calm/illiquid market on this symbol, **not** a recording "
                "outage, so divergence is **not** attributable to data here."
            )
        if benign:
            lines.append("")
            lines.append(f"_({len(benign)} benign gap(s) where other feeds kept ticking — not attributed.)_")
        if culprit is not None and div_ns is not None:
            lines.append("")
            lines.append(
                f"➡️ The first decision divergence ({_fmt_ns(div_ns)}) falls inside the "
                f"{culprit.gap_seconds:.0f}s outage {_fmt_ns(culprit.start_ns)} → {_fmt_ns(culprit.end_ns)} "
                "— the likely root cause of the fill divergence on this market."
            )
        lines.append("")
        widest = sorted(gaps, key=lambda g: g.gap_seconds, reverse=True)[:10]
        lines.append(f"Widest gaps (top {len(widest)} of {len(gaps)}):")
        lines.append("")
        lines.append("| Gap start | Gap end | Width (s) | Attribution |")
        lines.append("|-----------|---------|-----------|-------------|")
        for g in widest:
            mark = " ⬅️" if g is culprit else ""
            attr = "outage" if g.global_silence else "benign (feeds ticking)"
            lines.append(f"| {_fmt_ns(g.start_ns)} | {_fmt_ns(g.end_ns)} | {g.gap_seconds:.0f}{mark} | {attr} |")
        lines.append("")

    # Layer 3
    l3 = result.layer3
    lines.append("## Layer 3: PnL")
    lines.append("")
    lines.append("| Item | Live | Sim | Diff |")
    lines.append("|------|------|-----|------|")
    lines.append(f"| Realized PnL | ${l3.live_realized:.4f} | ${l3.sim_realized:.4f} | ${l3.pnl_diff:+.4f} |")
    lines.append(f"| Settlement winner | {l3.settlement_winner_match} | — | — |")
    lines.append(f"| PnL match | {l3.pnl_match} | — | — |")
    lines.append("")
    lines.append("**Waterfall attribution** (Perold implementation shortfall):")
    lines.append("")
    lines.append("| Component | $Amount |")
    lines.append("|-----------|---------|")
    for component, amount in l3.waterfall.items():
        lines.append(f"| {component} | ${amount:+.4f} |")
    lines.append("")
    lines.append(
        "_matched_entry/exit are split into **delay** (sim entered/exited at a "
        "different time into a moving market — benchmarked on the leg's own book "
        "mid, i.e. its delta-scaled fair value) and **impact** (sim filled a "
        "different-quality price at the same instant). opportunity_cost values the "
        "unmatched (un-replicated) legs marked at the resolved settlement price._"
    )
    lines.append("")

    return "\n".join(lines)
