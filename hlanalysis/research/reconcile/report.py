"""Render reconciliation results as markdown."""

from __future__ import annotations

from datetime import UTC, datetime

from hlanalysis.research.reconcile.reconcile import ReconcileResult


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
        lines.append("| Episode | Side | Live size | Sim size | Size diff | Live VWAP | Sim VWAP | VWAP diff |")
        lines.append("|---------|------|-----------|----------|-----------|-----------|----------|-----------|")
        for i, row in l2.episode_table.iterrows():
            lines.append(
                f"| {i + 1} "
                f"| {row.get('live_side', 'N/A')} "
                f"| {_fmt_float(row.get('live_size'), 4)} "
                f"| {_fmt_float(row.get('sim_size'), 4)} "
                f"| {_fmt_float(row.get('size_diff'), 4)} "
                f"| {_fmt_float(row.get('live_vwap'), 4)} "
                f"| {_fmt_float(row.get('sim_vwap'), 4)} "
                f"| {_fmt_float(row.get('vwap_diff'), 4)} |"
            )
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
    lines.append("**Waterfall attribution:**")
    lines.append("")
    lines.append("| Component | $Amount |")
    lines.append("|-----------|---------|")
    for component, amount in l3.waterfall.items():
        lines.append(f"| {component} | ${amount:+.4f} |")
    lines.append("")

    return "\n".join(lines)
