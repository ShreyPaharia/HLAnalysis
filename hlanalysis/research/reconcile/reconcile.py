"""Core 4-layer sim-vs-live reconciliation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hlanalysis.research.reconcile.book import (
    BookReader,
    RefPriceReader,
    book_parity_pct,
    recorded_ref_price_at,
)

# ── Layer 0: Preconditions ──────────────────────────────────────────────────


@dataclass
class PreconditionResult:
    """Result of pre-flight checks comparing live and sim traces.

    Parameters
    ----------
    config_hash_match:
        ``"PASS"`` | ``"FAIL:<detail>"`` | ``"SKIP:no_live_hash"``
    question_identity_match:
        ``"PASS"`` | ``"FAIL:<detail>"``
    window_match:
        ``"PASS"`` | ``"FAIL:<detail>"``
    overall:
        ``"PASS"`` | ``"FAIL"``
    """

    config_hash_match: str
    question_identity_match: str
    window_match: str
    overall: str


def check_preconditions(
    live_trace: pd.DataFrame,
    sim_trace: pd.DataFrame,
    live_config_hash: str | None = None,
    sim_config_hash: str | None = None,
    live_expiry_ns: int | None = None,
    sim_expiry_ns: int | None = None,
) -> PreconditionResult:
    """Layer 0: verify the two traces are for the same question with the same config.

    Parameters
    ----------
    live_trace:
        Decision trace from the live engine.
    sim_trace:
        Decision trace from the simulator.
    live_config_hash:
        Config hash from live (may be None if unavailable).
    sim_config_hash:
        Config hash from sim (may be None if unavailable).
    live_expiry_ns:
        Expiry timestamp for the live trace (ns); derived from trace if None.
    sim_expiry_ns:
        Expiry timestamp for the sim trace (ns); derived from trace if None.

    Returns
    -------
    PreconditionResult with per-check status and an overall verdict.
    """
    # -- Config hash check --
    # The live config hash MUST come from the per-question LIVE TRACE, never the
    # engine's *current* hash (which may have changed since the question ran and
    # would spuriously FAIL). When the live trace is empty for the question (no
    # trace captured), there is nothing to compare against -> SKIP, not FAIL.
    live_hash = _trace_config_hash(live_trace)
    if live_hash is None and not live_trace.empty:
        # Trace present but lacks a config_hash field: fall back to the caller's hint.
        live_hash = live_config_hash

    sim_hash = sim_config_hash if sim_config_hash is not None else _trace_config_hash(sim_trace)

    if live_hash is None or sim_hash is None:
        config_hash_match = "SKIP:no_live_hash"
    elif str(live_hash) != str(sim_hash):
        config_hash_match = f"FAIL:live={live_hash} sim={sim_hash}"
    else:
        config_hash_match = "PASS"

    # -- Question identity check --
    def _get_field(df: pd.DataFrame, col: str) -> Any:
        if col in df.columns and not df.empty:
            return df[col].iloc[0]
        return None

    live_qidx = _get_field(live_trace, "question_idx")
    sim_qidx = _get_field(sim_trace, "question_idx")
    live_klass = _get_field(live_trace, "klass")
    sim_klass = _get_field(sim_trace, "klass")

    identity_fails = []
    if live_qidx is not None and sim_qidx is not None and live_qidx != sim_qidx:
        identity_fails.append(f"question_idx:live={live_qidx} sim={sim_qidx}")
    if live_klass is not None and sim_klass is not None and live_klass != sim_klass:
        identity_fails.append(f"klass:live={live_klass} sim={sim_klass}")

    question_identity_match = "FAIL:" + "|".join(identity_fails) if identity_fails else "PASS"

    # -- Window overlap check --
    def _trace_window(df: pd.DataFrame, expiry_ns: int | None) -> tuple[int, int] | None:
        if "ts_ns" not in df.columns or df.empty:
            return None
        ts = df["ts_ns"].dropna()
        if ts.empty:
            return None
        return int(ts.min()), int(ts.max())

    live_window = _trace_window(live_trace, live_expiry_ns)
    sim_window = _trace_window(sim_trace, sim_expiry_ns)

    if live_window is None or sim_window is None:
        window_match = "FAIL:empty_trace"
    else:
        overlap_start = max(live_window[0], sim_window[0])
        overlap_end = min(live_window[1], sim_window[1])
        overlap = max(0, overlap_end - overlap_start)
        live_span = max(1, live_window[1] - live_window[0])
        sim_span = max(1, sim_window[1] - sim_window[0])
        # Divide by the LARGER span: a tiny sim window fully inside a long live
        # window would score 1.0 against min_span and pass, hiding that sim only
        # covers a sliver of the question (audit M3). Both must broadly overlap.
        max_span = max(live_span, sim_span)
        if overlap / max_span >= 0.5:
            window_match = "PASS"
        else:
            window_match = f"FAIL:overlap_pct={overlap / max_span:.2f}"

    # -- Overall --
    fails = [s for s in [config_hash_match, question_identity_match, window_match] if s.startswith("FAIL")]
    overall = "FAIL" if fails else "PASS"

    return PreconditionResult(
        config_hash_match=config_hash_match,
        question_identity_match=question_identity_match,
        window_match=window_match,
        overall=overall,
    )


# ── Layer 1: Decisions ──────────────────────────────────────────────────────


@dataclass
class DecisionDivergence:
    """A single field divergence between live and sim at a given time.

    Parameters
    ----------
    ts_ns:
        Bucket timestamp in nanoseconds.
    field:
        Name of the diverging field (e.g. ``"sigma"``).
    live_val:
        Live value.
    sim_val:
        Sim value.
    rel_diff:
        Relative difference for numeric fields; None for categorical.
    """

    ts_ns: int
    field: str
    live_val: Any
    sim_val: Any
    rel_diff: float | None


@dataclass
class DecisionResult:
    """Alignment comparison of live vs sim decision traces.

    Parameters
    ----------
    match_rate:
        Fraction of aligned buckets where ``action`` agrees.
    first_divergence:
        First bucket with a numeric or action divergence, or None.
    diff_table:
        Per-bucket aligned diff DataFrame.
    classification:
        One of ``"match"``, ``"sigma_diff"``, ``"gate_diff"``, ``"cadence"``.
    n_live_buckets:
        Number of unique minute buckets in the live trace.
    n_sim_buckets:
        Number of unique minute buckets in the sim trace.
    n_aligned:
        Number of buckets present in both traces.
    """

    match_rate: float
    first_divergence: DecisionDivergence | None
    diff_table: pd.DataFrame
    classification: str
    n_live_buckets: int
    n_sim_buckets: int
    n_aligned: int
    n_live_events: int = 0
    n_sim_events: int = 0


def reconcile_decisions(
    live_trace: pd.DataFrame,
    sim_trace: pd.DataFrame,
    bucket_seconds: int = 60,
    align_tol_seconds: int = 2,
    sigma_rel_tol: float = 0.05,
    edge_abs_tol: float = 0.005,
) -> DecisionResult:
    """Layer 1: align traces on a coarse time grid and compare decision fields.

    Parameters
    ----------
    live_trace:
        Decision trace from the live engine (canonical schema).
    sim_trace:
        Decision trace from the simulator (canonical schema).
    bucket_seconds:
        Grid size in seconds for coarse alignment.
    align_tol_seconds:
        Tolerance for bucket alignment (unused internally but kept for API).
    sigma_rel_tol:
        Relative sigma difference threshold for "sigma_diff" classification.
    edge_abs_tol:
        Absolute edge difference threshold for reporting divergence.

    Returns
    -------
    DecisionResult with match_rate, first_divergence, diff_table, and classification.
    """
    bucket_ns = bucket_seconds * 1_000_000_000

    numeric_fields = ["sigma", "p_model", "edge"]

    def _bucket_select(df: pd.DataFrame) -> pd.DataFrame:
        # Pick the most SIGNIFICANT row per bucket, not merely the first. A trace
        # is ~99.97% ``hold``; taking the first row would let a single ``hold``
        # mask the lone ``enter``/``exit`` in that minute and report a spurious
        # 100% match (the 2026-06-20 #1000465 blind spot). Prefer the first
        # non-hold action when the bucket contains one.
        #
        # BUT the action-bearing row (enter/exit) usually carries NULL
        # sigma/p_model/edge — the engine populates those on the gate-evaluation
        # (hold/scan) rows. So overlay the numeric fields with the bucket's
        # mean of non-null values; otherwise selecting the enter row would null
        # out the very fields the numeric comparison needs (audit H1).
        if df.empty or "ts_ns" not in df.columns:
            return df
        df = df.copy()
        df["_bucket"] = (df["ts_ns"] // bucket_ns) * bucket_ns
        if "action" in df.columns:
            df["_is_hold"] = (df["action"].astype(str) == "hold").astype(int)
            df = df.sort_values(["_bucket", "_is_hold", "ts_ns"])
        out = df.groupby("_bucket", sort=True).first().reset_index()
        for fld in numeric_fields:
            if fld in df.columns:
                per_bucket = df.assign(_v=pd.to_numeric(df[fld], errors="coerce")).groupby("_bucket")["_v"].mean()
                out[fld] = out["_bucket"].map(per_bucket)
        return out.drop(columns=["_is_hold"], errors="ignore")

    def _n_events(df: pd.DataFrame) -> int:
        if df.empty or "action" not in df.columns:
            return 0
        return int((df["action"].astype(str) != "hold").sum())

    n_live_events = _n_events(live_trace)
    n_sim_events = _n_events(sim_trace)

    live_b = _bucket_select(live_trace)
    sim_b = _bucket_select(sim_trace)

    n_live_buckets = len(live_b)
    n_sim_buckets = len(sim_b)

    if live_b.empty or sim_b.empty:
        empty_diff = pd.DataFrame()
        return DecisionResult(
            match_rate=0.0,
            first_divergence=None,
            diff_table=empty_diff,
            classification="cadence",
            n_live_buckets=n_live_buckets,
            n_sim_buckets=n_sim_buckets,
            n_aligned=0,
            n_live_events=n_live_events,
            n_sim_events=n_sim_events,
        )

    live_b = live_b.set_index("_bucket")
    sim_b = sim_b.set_index("_bucket")

    common_buckets = live_b.index.intersection(sim_b.index)
    n_aligned = len(common_buckets)

    if n_aligned == 0:
        return DecisionResult(
            match_rate=0.0,
            first_divergence=None,
            diff_table=pd.DataFrame(),
            classification="cadence",
            n_live_buckets=n_live_buckets,
            n_sim_buckets=n_sim_buckets,
            n_aligned=0,
            n_live_events=n_live_events,
            n_sim_events=n_sim_events,
        )

    live_aligned = live_b.loc[common_buckets]
    sim_aligned = sim_b.loc[common_buckets]

    rows = []
    action_matches = 0
    first_div: DecisionDivergence | None = None
    sigma_diffs: list[float] = []

    for bucket in common_buckets:
        l_row = live_aligned.loc[bucket]
        s_row = sim_aligned.loc[bucket]
        row: dict[str, Any] = {"bucket_ns": bucket}

        for fld in numeric_fields:
            l_val = float(l_row[fld]) if fld in live_aligned.columns and not _is_nan(l_row.get(fld)) else None
            s_val = float(s_row[fld]) if fld in sim_aligned.columns and not _is_nan(s_row.get(fld)) else None
            row[f"live_{fld}"] = l_val
            row[f"sim_{fld}"] = s_val

            if l_val is not None and s_val is not None:
                denom = abs(l_val) if abs(l_val) > 1e-12 else 1e-12
                rel_diff = abs(l_val - s_val) / denom
                row[f"rel_diff_{fld}"] = rel_diff

                if fld == "sigma":
                    sigma_diffs.append(rel_diff)

                if first_div is None:
                    # sigma uses a RELATIVE tolerance; edge and p_model use an
                    # ABSOLUTE one. p_model was previously never tested here
                    # (audit H2) and the old `and/or` chain was precedence-fragile.
                    if fld == "sigma":
                        diverged = rel_diff > sigma_rel_tol
                    else:  # edge, p_model
                        diverged = abs(l_val - s_val) > edge_abs_tol
                    if diverged:
                        first_div = DecisionDivergence(
                            ts_ns=int(bucket),
                            field=fld,
                            live_val=l_val,
                            sim_val=s_val,
                            rel_diff=rel_diff,
                        )
            else:
                row[f"rel_diff_{fld}"] = None

        l_action = l_row["action"] if "action" in live_aligned.columns else None
        s_action = s_row["action"] if "action" in sim_aligned.columns else None
        row["live_action"] = l_action
        row["sim_action"] = s_action
        action_match = l_action == s_action
        row["action_match"] = action_match
        if action_match:
            action_matches += 1
        elif first_div is None:
            first_div = DecisionDivergence(
                ts_ns=int(bucket),
                field="action",
                live_val=l_action,
                sim_val=s_action,
                rel_diff=None,
            )

        rows.append(row)

    match_rate = action_matches / n_aligned if n_aligned > 0 else 0.0
    diff_table = pd.DataFrame(rows)

    # Classification
    max_possible_buckets = max(n_live_buckets, n_sim_buckets)
    cadence_gap = (max_possible_buckets - n_aligned) / max(max_possible_buckets, 1)

    mean_sigma_diff = float(np.mean(sigma_diffs)) if sigma_diffs else 0.0

    if match_rate >= 0.95:
        classification = "match"
    elif cadence_gap > 0.10:
        classification = "cadence"
    elif mean_sigma_diff > sigma_rel_tol:
        classification = "sigma_diff"
    else:
        classification = "gate_diff"

    return DecisionResult(
        match_rate=match_rate,
        first_divergence=first_div,
        diff_table=diff_table,
        classification=classification,
        n_live_buckets=n_live_buckets,
        n_sim_buckets=n_sim_buckets,
        n_aligned=n_aligned,
        n_live_events=n_live_events,
        n_sim_events=n_sim_events,
    )


def _is_nan(val: Any) -> bool:
    """Return True if val is None or a float NaN."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _trace_config_hash(df: pd.DataFrame) -> str | None:
    """Return the (last) config_hash from a per-question trace, or None if absent."""
    if df.empty or "config_hash" not in df.columns:
        return None
    val = df["config_hash"].iloc[-1]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val)


# ── Layer 2: Fills ──────────────────────────────────────────────────────────


@dataclass
class FillEpisode:
    """A contiguous cluster of same-direction fills forming one position leg.

    Parameters
    ----------
    side:
        ``"BUY"`` or ``"SELL"``.
    start_ns:
        Timestamp of the first fill in nanoseconds.
    end_ns:
        Timestamp of the last fill in nanoseconds.
    total_size:
        Sum of fill sizes.
    vwap:
        Volume-weighted average price.
    n_fills:
        Number of fills in this episode.
    """

    side: str
    start_ns: int
    end_ns: int
    total_size: float
    vwap: float
    n_fills: int


@dataclass
class FillsResult:
    """Comparison of live vs sim fill episodes.

    Parameters
    ----------
    live_episodes:
        Episodes detected in the live fills.
    sim_episodes:
        Episodes detected in the sim fills.
    episode_table:
        Matched episode pairs with size_diff and vwap_diff columns.
    book_parity_pct:
        Fraction of live fills whose price was present in the recorded book.
    gap_classification:
        One of ``"match"``, ``"latency"``, ``"depth_slippage"``,
        ``"missed_episode"``, ``"extra_episode"``.
    n_live_fills:
        Total number of live fill rows.
    n_sim_fills:
        Total number of sim fill rows.
    """

    live_episodes: list[FillEpisode]
    sim_episodes: list[FillEpisode]
    episode_table: pd.DataFrame
    book_parity_pct: float | None
    gap_classification: str
    n_live_fills: int
    n_sim_fills: int


def _group_episodes(fills: pd.DataFrame) -> list[FillEpisode]:
    """Group fills into episodes (contiguous same-side clusters).

    Parameters
    ----------
    fills:
        DataFrame with cols: ts_ns, side, price, size.

    Returns
    -------
    List of FillEpisode objects in chronological order.
    """
    if fills.empty:
        return []

    fills = fills.sort_values("ts_ns").reset_index(drop=True)
    episodes: list[FillEpisode] = []
    current_side: str | None = None
    sizes: list[float] = []
    prices: list[float] = []
    start_ns = 0
    end_ns = 0

    for _, row in fills.iterrows():
        # Canonicalise side case. Real venue/sim fills use lowercase
        # ``buy``/``sell``; downstream PnL pairing compares against ``BUY``/
        # ``SELL``. Normalising here keeps both consistent (the 2026-06-20
        # #1000465 false FAIL was sim_realized=$0 from this case mismatch).
        side = str(row["side"]).strip().upper()
        price = float(row["price"])
        size = float(row["size"])
        ts = int(row["ts_ns"])

        if current_side is None:
            current_side = side
            start_ns = ts
            end_ns = ts
            sizes = [size]
            prices = [price]
        elif side == current_side:
            sizes.append(size)
            prices.append(price)
            end_ns = ts
        else:
            # Side flip: close current episode
            total = sum(sizes)
            vwap = sum(p * s for p, s in zip(prices, sizes)) / total if total > 0 else 0.0
            episodes.append(
                FillEpisode(
                    side=current_side,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    total_size=total,
                    vwap=vwap,
                    n_fills=len(sizes),
                )
            )
            current_side = side
            start_ns = ts
            end_ns = ts
            sizes = [size]
            prices = [price]

    if current_side is not None and sizes:
        total = sum(sizes)
        vwap = sum(p * s for p, s in zip(prices, sizes)) / total if total > 0 else 0.0
        episodes.append(
            FillEpisode(
                side=current_side,
                start_ns=start_ns,
                end_ns=end_ns,
                total_size=total,
                vwap=vwap,
                n_fills=len(sizes),
            )
        )

    return episodes


@dataclass
class EpisodeMatch:
    """Result of pairing live and sim fill episodes by side.

    Parameters
    ----------
    matched:
        ``(live_episode, sim_episode)`` pairs that correspond (same side, closest
        start within the cap).
    unmatched_live:
        Live episodes with no sim counterpart — a leg live traded and sim did not.
    unmatched_sim:
        Sim episodes with no live counterpart — a leg sim traded and live did not.
    """

    matched: list[tuple[FillEpisode, FillEpisode]]
    unmatched_live: list[FillEpisode]
    unmatched_sim: list[FillEpisode]


def _match_episodes(
    live_eps: list[FillEpisode],
    sim_eps: list[FillEpisode],
    max_match_gap_ns: int,
) -> EpisodeMatch:
    """Pair live/sim episodes by SIDE, greedily by closest start time.

    Repeatedly take the globally-closest unmatched same-side ``(live, sim)`` pair
    whose start-time gap is within ``max_match_gap_ns``. Episodes left over on
    either side are real structural divergence (a round-trip one venue made and
    the other did not) and are returned separately rather than force-paired across
    an arbitrary clock distance. The shared spine for Layer 2 (the episode table)
    and Layer 3 (the matched-leg PnL waterfall).
    """
    used_live: set[int] = set()
    used_sim: set[int] = set()
    matched: list[tuple[FillEpisode, FillEpisode]] = []

    for side in ("BUY", "SELL"):
        li_idx = [i for i, e in enumerate(live_eps) if e.side == side]
        sj_idx = [j for j, e in enumerate(sim_eps) if e.side == side]
        # Greedy: rank all candidate pairs by start-time gap, take closest first.
        candidates = sorted(
            ((abs(live_eps[i].start_ns - sim_eps[j].start_ns), i, j) for i in li_idx for j in sj_idx),
            key=lambda t: t[0],
        )
        for gap, i, j in candidates:
            if i in used_live or j in used_sim or gap > max_match_gap_ns:
                continue
            used_live.add(i)
            used_sim.add(j)
            matched.append((live_eps[i], sim_eps[j]))

    # Keep matched pairs chronological by live start time for stable reporting.
    matched.sort(key=lambda pair: pair[0].start_ns)
    unmatched_live = [e for i, e in enumerate(live_eps) if i not in used_live]
    unmatched_sim = [e for j, e in enumerate(sim_eps) if j not in used_sim]
    return EpisodeMatch(matched=matched, unmatched_live=unmatched_live, unmatched_sim=unmatched_sim)


def _episode_match_rows(match: EpisodeMatch) -> list[dict[str, Any]]:
    """Flatten an :class:`EpisodeMatch` into chronological report rows."""
    rows: list[dict[str, Any]] = []
    for lep, sep in match.matched:
        rows.append(
            {
                "match_status": "matched",
                "live_side": lep.side,
                "live_start_ns": lep.start_ns,
                "sim_start_ns": sep.start_ns,
                "live_size": lep.total_size,
                "sim_size": sep.total_size,
                "size_diff": sep.total_size - lep.total_size,
                "live_vwap": lep.vwap,
                "sim_vwap": sep.vwap,
                "vwap_diff": sep.vwap - lep.vwap,
                "latency_ns": lep.start_ns - sep.start_ns,
            }
        )
    for lep in match.unmatched_live:
        rows.append(
            {
                "match_status": "live_only",
                "live_side": lep.side,
                "live_start_ns": lep.start_ns,
                "sim_start_ns": None,
                "live_size": lep.total_size,
                "sim_size": None,
                "size_diff": None,
                "live_vwap": lep.vwap,
                "sim_vwap": None,
                "vwap_diff": None,
                "latency_ns": None,
            }
        )
    for sep in match.unmatched_sim:
        rows.append(
            {
                "match_status": "sim_only",
                "live_side": sep.side,
                "live_start_ns": None,
                "sim_start_ns": sep.start_ns,
                "live_size": None,
                "sim_size": sep.total_size,
                "size_diff": None,
                "live_vwap": None,
                "sim_vwap": sep.vwap,
                "vwap_diff": None,
                "latency_ns": None,
            }
        )

    def _sort_key(r: dict[str, Any]) -> int:
        return int(r.get("live_start_ns") or r.get("sim_start_ns") or 0)

    rows.sort(key=_sort_key)
    return rows


def _round_trip_pnl(eps: list[FillEpisode]) -> float:
    """Realized PnL of a set of episodes, pairing buys with sells in order.

    Used to value the *unmatched* round-trips (legs one venue traded and the other
    did not) so they enter the PnL waterfall as their own attributable buckets.
    """
    buys = [e for e in eps if e.side == "BUY"]
    sells = [e for e in eps if e.side == "SELL"]
    return sum((se.vwap - be.vwap) * min(be.total_size, se.total_size) for be, se in zip(buys, sells))


def _realized_from_fills(fills: pd.DataFrame) -> float:
    """Realized PnL from a fill stream via running average-cost accounting.

    A sell realizes ``(price − avg_cost) × size`` against the running long basis;
    a buy updates the basis. Correct for scale-ins, scale-outs and partial closes
    (a plain buy/sell episode pairing drops the remainder of any unbalanced leg).
    Open inventory at the end is left unrealized (settlement is handled elsewhere).
    """
    if fills.empty or not {"side", "price", "size", "ts_ns"}.issubset(fills.columns):
        return 0.0
    ordered = fills.sort_values("ts_ns")
    pos = 0.0
    avg_cost = 0.0
    realized = 0.0
    for _, row in ordered.iterrows():
        side = str(row["side"]).strip().upper()
        price = float(row["price"])
        size = float(row["size"])
        if side == "BUY":
            if pos + size > 0:
                avg_cost = (avg_cost * pos + price * size) / (pos + size)
            pos += size
        else:  # SELL
            closed = min(size, pos) if pos > 0 else 0.0
            realized += (price - avg_cost) * closed
            pos -= size
            if pos <= 0:
                pos = max(pos, 0.0)
                avg_cost = 0.0 if pos == 0 else avg_cost
    return realized


def reconcile_fills(
    live_fills: pd.DataFrame,
    sim_fills: pd.DataFrame,
    data_root: Path | None = None,
    book_reader: BookReader | None = None,
    max_match_gap_seconds: float = 1800.0,
) -> FillsResult:
    """Layer 2: group fills into episodes, compare VWAP/size, check book parity.

    Parameters
    ----------
    live_fills:
        Live venue fills with cols: ts_ns, side, price, size, symbol, [fee, closed_pnl].
    sim_fills:
        Sim fills with same schema (fee/closed_pnl optional).
    data_root:
        Root of the HL recorded data tree (for book parity check).
    book_reader:
        Injectable book reader for testing.
    max_match_gap_seconds:
        Maximum live↔sim episode start-time gap to still call a match. Sim and live
        execute the same legs at timestamps that legitimately drift (sim fills a
        modeled book; live the real venue), so this is generous (default 30 min).
        The old 300 s cap dropped genuinely-corresponding legs.

    Returns
    -------
    FillsResult with episode lists, matched table, book_parity_pct, and classification.
    """
    live_eps = _group_episodes(live_fills) if not live_fills.empty else []
    sim_eps = _group_episodes(sim_fills) if not sim_fills.empty else []

    # Compute book parity on live fills
    bk_pct: float | None = None
    if not live_fills.empty and "symbol" in live_fills.columns:
        if data_root is not None or book_reader is not None:
            bk_pct = book_parity_pct(live_fills, data_root or Path("."), reader=book_reader)

    max_match_gap_ns = int(max_match_gap_seconds * 1_000_000_000)
    match = _match_episodes(live_eps, sim_eps, max_match_gap_ns)
    n_unmatched_live = len(match.unmatched_live)
    n_unmatched_sim = len(match.unmatched_sim)
    episode_table = pd.DataFrame(_episode_match_rows(match))

    # Classification
    n_live = len(live_eps)
    n_sim = len(sim_eps)

    if n_live == 0 and n_sim == 0:
        classification = "match"
    elif n_unmatched_live and n_unmatched_sim:
        # Both venues have legs the other lacks → genuine structural divergence
        # (e.g. each made a round-trip the other did not). Most often a reference-
        # feed gap desynced the entry gate — see check_reference_coverage.
        classification = "structure_diff"
    elif n_unmatched_live:
        classification = "missed_episode"  # sim is missing legs live had
    elif n_unmatched_sim:
        classification = "extra_episode"  # sim invented legs live did not have
    else:
        # Fully matched: grade on latency / VWAP.
        latencies = episode_table["latency_ns"].dropna().abs()
        max_latency_s = latencies.max() / 1e9 if not latencies.empty else 0.0
        vwap_diffs = episode_table["vwap_diff"].dropna().abs()
        max_vwap_diff = vwap_diffs.max() if not vwap_diffs.empty else 0.0

        if max_vwap_diff > 0.005:
            classification = "depth_slippage"
        elif max_latency_s > 30:
            classification = "latency"
        else:
            classification = "match"

    return FillsResult(
        live_episodes=live_eps,
        sim_episodes=sim_eps,
        episode_table=episode_table,
        book_parity_pct=bk_pct,
        gap_classification=classification,
        n_live_fills=len(live_fills),
        n_sim_fills=len(sim_fills),
    )


# ── Reference-feed coverage (Layer 2.5) ─────────────────────────────────────


@dataclass
class ReferenceGap:
    """A hole in the recorded reference (perp mark) feed.

    During a gap the SIM holds the last reference value while the live engine
    (on its own uninterrupted feed) tracked the true price — desyncing the entry
    gate and producing genuine sim≠live fill divergence that is NOT a harness bug.

    Parameters
    ----------
    start_ns:
        Timestamp of the last tick before the gap.
    end_ns:
        Timestamp of the first tick after the gap.
    gap_seconds:
        Width of the hole in seconds.
    """

    start_ns: int
    end_ns: int
    gap_seconds: float


# A reader returns the sorted reference tick timestamps (ns) in [start_ns, end_ns].
# Signature: (ref_symbol, start_ns, end_ns, data_root) -> list[int]
ReferenceTsReader = Callable[[str, int, int, "Path | None"], "list[int]"]


def _default_reference_ts_reader(
    ref_symbol: str,
    start_ns: int,
    end_ns: int,
    data_root: Path | None,
) -> list[int]:
    """Read recorded HL perp ``mark`` exchange timestamps for ``ref_symbol``."""
    if data_root is None:
        return []
    try:
        import duckdb  # noqa: PLC0415
    except ImportError:
        return []
    pattern = str(
        data_root / f"venue=hyperliquid/product_type=perp/mechanism=clob/event=mark/symbol={ref_symbol}/**/*.parquet"
    )
    try:
        con = duckdb.connect()
        df = con.execute(
            f"""
            SELECT exchange_ts FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE exchange_ts BETWEEN {start_ns} AND {end_ns}
            ORDER BY exchange_ts
            """
        ).df()
        con.close()
        return [int(x) for x in df["exchange_ts"].tolist()]
    except Exception:
        return []


def check_reference_coverage(
    start_ns: int,
    end_ns: int,
    ref_symbol: str = "BTC",
    data_root: Path | None = None,
    gap_threshold_seconds: float = 60.0,
    ts_reader: ReferenceTsReader | None = None,
) -> list[ReferenceGap]:
    """Detect gaps in the recorded reference feed over ``[start_ns, end_ns]``.

    Parameters
    ----------
    start_ns, end_ns:
        Window to inspect (the question's trading window).
    ref_symbol:
        Reference perp symbol (e.g. ``"BTC"``).
    data_root:
        Root of the HL recorded data tree; may be None when ``ts_reader`` is given.
    gap_threshold_seconds:
        Report inter-tick gaps wider than this. The recorder normally ticks every
        ~1-3 s, so 60 s comfortably separates a real outage from jitter.
    ts_reader:
        Injectable timestamp reader for testing.

    Returns
    -------
    Chronological list of ReferenceGap. Empty if coverage is dense (or unreadable).
    """
    reader = ts_reader or _default_reference_ts_reader
    ts = sorted(reader(ref_symbol, start_ns, end_ns, data_root))
    threshold_ns = gap_threshold_seconds * 1_000_000_000
    gaps: list[ReferenceGap] = []
    for prev, cur in zip(ts, ts[1:]):
        delta = cur - prev
        if delta > threshold_ns:
            gaps.append(ReferenceGap(start_ns=prev, end_ns=cur, gap_seconds=delta / 1e9))
    return gaps


# ── Settlement-winner helpers ───────────────────────────────────────────────


def _side_idx_from_symbol(symbol: str) -> int | None:
    """Decode the side index from an HL outcome symbol (``#NNN`` -> NNN % 10).

    Per the HL symbol convention ``#NNN`` -> ``side_idx = NNN % 10`` where
    0 = Yes, 1 = No. Returns None if no ``#NNN`` suffix is present.
    """
    if "#" not in symbol:
        return None
    digits = "".join(ch for ch in symbol.rsplit("#", 1)[1] if ch.isdigit())
    if not digits:
        return None
    return int(digits) % 10


def _winner_from_settlement_fill(fills: pd.DataFrame, expiry_ns: int | None = None) -> str | None:
    """Derive the live winning side from the settlement-as-fill.

    HL books settlement as a venue fill at price ~1.0 (the held leg won) / ~0.0
    (it lost). The held leg's Yes/No identity comes from the symbol's side index.
    Returns ``"yes"``/``"no"`` or None when no settlement-priced fill is present.

    ``expiry_ns`` guards against misreading a *legitimate deep-ITM trading fill*
    (e.g. buying a 0.99 favorite mid-session) as the settlement leg: when given,
    only fills at/after expiry are considered (audit M2). Settlement fills land at
    expiry; trading fills do not.
    """
    if fills is None or fills.empty:
        return None
    if "price" not in fills.columns or "symbol" not in fills.columns or "ts_ns" not in fills.columns:
        return None
    ordered = fills.sort_values("ts_ns")
    if expiry_ns is not None:
        ordered = ordered[ordered["ts_ns"] >= expiry_ns]
        if ordered.empty:
            return None
    settle = ordered[(ordered["price"] >= 0.99) | (ordered["price"] <= 0.01)]
    if settle.empty:
        return None
    row = settle.iloc[-1]
    side_idx = _side_idx_from_symbol(str(row["symbol"]))
    if side_idx is None:
        return None
    leg_is_yes = side_idx == 0
    won = float(row["price"]) >= 0.5
    return ("yes" if leg_is_yes else "no") if won else ("no" if leg_is_yes else "yes")


def _norm_winner(val: Any) -> str | None:
    """Normalise a winner token for comparison; None / 'unknown' -> None."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "unknown", "none", "nan"):
        return None
    return s


# ── Layer 3: PnL ────────────────────────────────────────────────────────────


@dataclass
class PnLResult:
    """PnL comparison between live and sim.

    Parameters
    ----------
    live_realized:
        Sum of venue fill closed_pnl (venue-authoritative).
    sim_realized:
        Sim realized PnL computed from fill episodes.
    pnl_diff:
        live_realized - sim_realized.
    settlement_winner_match:
        ``"PASS"`` | ``"FAIL:<detail>"`` | ``"SKIP:no_settlement"``.
    waterfall:
        Attribution components: entry_vwap_diff, exit_vwap_diff, size_diff,
        fee_diff, residual.
    pnl_match:
        ``"PASS"`` if |pnl_diff| <= threshold, else ``"FAIL:<amount>"``.
    """

    live_realized: float
    sim_realized: float
    pnl_diff: float
    settlement_winner_match: str
    waterfall: dict[str, float]
    pnl_match: str


def reconcile_pnl(
    live_fills: pd.DataFrame,
    sim_fills: pd.DataFrame,
    live_settlement: dict[str, Any],
    sim_resolved: dict[str, Any],
    pnl_abs_threshold: float = 5.0,
    max_match_gap_seconds: float = 1800.0,
    expiry_ns: int | None = None,
    ref_symbol: str = "BTC",
    data_root: Path | None = None,
    ref_price_reader: RefPriceReader | None = None,
) -> PnLResult:
    """Layer 3: compare live vs sim realized PnL with attribution waterfall.

    Parameters
    ----------
    live_fills:
        Live fills with cols: ts_ns, side, price, size, [fee], [closed_pnl].
    sim_fills:
        Sim fills with same schema.
    live_settlement:
        Dict with keys: question_idx, realized_pnl, ts_ns.  May be empty.
    sim_resolved:
        Dict with keys: winner_side (str), resolved_outcome (float).  May be empty.
    pnl_abs_threshold:
        Maximum acceptable |pnl_diff| for PASS verdict.
    ref_symbol:
        Reference perp symbol for the delay/impact split (e.g. ``"BTC"``).
    data_root:
        Root of the HL recorded data tree; used by the default reference reader
        to sample the perp ``mark`` price at both episode timestamps.
    ref_price_reader:
        Injectable reference-price reader (for tests). When neither this nor
        ``data_root`` is available the matched VWAP gap cannot be split and is
        attributed entirely to impact (delay = 0).

    Returns
    -------
    PnLResult with realized PnL values, diff, and waterfall attribution.
    """
    # Live realized: sum of closed_pnl from venue fills
    live_realized = 0.0
    if not live_fills.empty and "closed_pnl" in live_fills.columns:
        live_realized = float(live_fills["closed_pnl"].fillna(0).sum())
    elif live_settlement:
        live_realized = float(live_settlement.get("realized_pnl", 0.0) or 0.0)

    # Sim realized: compute from fill episodes (entry + exit)
    sim_eps = _group_episodes(sim_fills) if not sim_fills.empty else []
    sim_fees_total = 0.0
    if not sim_fills.empty and "fee" in sim_fills.columns:
        sim_fees_total = float(sim_fills["fee"].fillna(0).sum())

    live_fees_total = 0.0
    if not live_fills.empty and "fee" in live_fills.columns:
        live_fees_total = float(live_fills["fee"].fillna(0).sum())

    # Sim realized: running average-cost over fills. Robust to scale-ins,
    # scale-outs and partial closes (the old index-paired ``min(size)`` dropped
    # the remainder of any unbalanced leg).
    sim_realized = _realized_from_fills(sim_fills) - sim_fees_total

    # ── Waterfall attribution (matched-structure) ──────────────────────────
    # Decompose pnl_diff (= live_realized − sim_realized) over the legs the two
    # actually share, plus the round-trips unique to each side. The old code
    # paired episodes positionally, comparing live's leg-1 to sim's leg-1 even
    # when those were different trades — yielding huge components that only
    # cancelled by luck. We now attribute only across *matched* legs and bucket
    # the unmatched round-trips explicitly (the 2026-06-20 #1000465 fix).
    live_eps_all = _group_episodes(live_fills) if not live_fills.empty else []
    max_match_gap_ns = int(max_match_gap_seconds * 1_000_000_000)
    match = _match_episodes(live_eps_all, sim_eps, max_match_gap_ns)

    matched_buys = [(le, se) for le, se in match.matched if le.side == "BUY"]
    matched_sells = [(le, se) for le, se in match.matched if le.side == "SELL"]

    # Split the matched VWAP gap (Perold implementation shortfall, SHR-146) into:
    #   delay  — sim entered/exited at a DIFFERENT TIME into a moving market;
    #   impact — sim filled a DIFFERENT-QUALITY price at the SAME instant.
    # The common arrival benchmark is the reference (perp mark) price sampled at
    # BOTH the live and sim episode timestamps. delay = ref-move × size; the
    # remainder of the VWAP gap is impact. delay + impact == the old matched_*_vwap.
    def _ref_at(ts_ns: int) -> float | None:
        if ref_price_reader is None and data_root is None:
            return None
        return recorded_ref_price_at(ref_symbol, ts_ns, data_root, reader=ref_price_reader)

    entry_delay = 0.0  # $ from sim entering matched legs at a different TIME
    entry_impact = 0.0  # $ from sim entering matched legs at a different-quality price
    exit_delay = 0.0  # $ from sim exiting matched legs at a different TIME
    exit_impact = 0.0  # $ from sim exiting matched legs at a different-quality price
    size_diff_pnl = 0.0  # $ from sim trading a different size on matched legs
    # Pair the i-th matched buy with the i-th matched sell into a round-trip.
    for (lbe, sbe), (lse, sse) in zip(matched_buys, matched_sells):
        live_size = (lbe.total_size + lse.total_size) / 2
        sim_size = (sbe.total_size + sse.total_size) / 2

        # Entry: contribution to (live_realized − sim_realized) is "sim − live".
        entry_gap = (sbe.vwap - lbe.vwap) * live_size  # sim paid more → +live
        r_live_entry = _ref_at(lbe.start_ns)
        r_sim_entry = _ref_at(sbe.start_ns)
        if r_live_entry is not None and r_sim_entry is not None:
            e_delay = (r_sim_entry - r_live_entry) * live_size
        else:
            e_delay = 0.0
        entry_delay += e_delay
        entry_impact += entry_gap - e_delay

        # Exit: contribution is "live − sim" (live sold higher → +live), so the
        # delay term carries the same orientation: ref-move from sim to live time.
        exit_gap = (lse.vwap - sse.vwap) * live_size
        r_live_exit = _ref_at(lse.start_ns)
        r_sim_exit = _ref_at(sse.start_ns)
        if r_live_exit is not None and r_sim_exit is not None:
            x_delay = (r_live_exit - r_sim_exit) * live_size
        else:
            x_delay = 0.0
        exit_delay += x_delay
        exit_impact += exit_gap - x_delay

        size_diff_pnl += (sse.vwap - sbe.vwap) * (live_size - sim_size)

    # Round-trips one side made and the other did not, valued on their own fills.
    live_only_pnl = _round_trip_pnl(match.unmatched_live)  # PnL live booked, sim lacked
    sim_only_pnl = _round_trip_pnl(match.unmatched_sim)  # PnL sim booked, live lacked

    fee_diff = sim_fees_total - live_fees_total
    pnl_diff = live_realized - sim_realized
    accounted = (
        entry_delay + entry_impact + exit_delay + exit_impact + size_diff_pnl + live_only_pnl - sim_only_pnl + fee_diff
    )
    residual = pnl_diff - accounted

    waterfall = {
        "matched_entry_delay": round(entry_delay, 6),
        "matched_entry_impact": round(entry_impact, 6),
        "matched_exit_delay": round(exit_delay, 6),
        "matched_exit_impact": round(exit_impact, 6),
        "matched_size": round(size_diff_pnl, 6),
        "live_only_roundtrips": round(live_only_pnl, 6),
        "sim_only_roundtrips": round(-sim_only_pnl, 6),
        "fee_diff": round(fee_diff, 6),
        "residual": round(residual, 6),
    }

    # Settlement winner check. Prefer an explicit winner_side from settlement;
    # otherwise derive the live winner from the settlement-as-fill (HL books
    # settlement as a venue fill at px~1.0 => the filled leg won). Compare to the
    # sim resolved outcome; SKIP if neither side is available.
    live_winner = live_settlement.get("winner_side") if live_settlement else None
    if live_winner is None:
        live_winner = _winner_from_settlement_fill(live_fills, expiry_ns=expiry_ns)
    sim_winner = sim_resolved.get("winner_side") if sim_resolved else None
    if sim_winner is None and sim_resolved:
        sim_winner = sim_resolved.get("resolved_outcome")

    norm_live = _norm_winner(live_winner)
    norm_sim = _norm_winner(sim_winner)

    if norm_live is None or norm_sim is None:
        settlement_winner_match = "SKIP:no_settlement"
    elif norm_live == norm_sim:
        settlement_winner_match = "PASS"
    else:
        settlement_winner_match = f"FAIL:live={live_winner} sim={sim_winner}"

    # PnL match verdict
    if abs(pnl_diff) <= pnl_abs_threshold:
        pnl_match = "PASS"
    else:
        pnl_match = f"FAIL:{pnl_diff:+.2f}"

    return PnLResult(
        live_realized=live_realized,
        sim_realized=sim_realized,
        pnl_diff=pnl_diff,
        settlement_winner_match=settlement_winner_match,
        waterfall=waterfall,
        pnl_match=pnl_match,
    )


# ── Verdict ─────────────────────────────────────────────────────────────────


@dataclass
class ReconcileResult:
    """Complete reconciliation result across all 4 layers.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    layer0:
        Precondition check result.
    layer1:
        Decision alignment result.
    layer2:
        Fill episode comparison result.
    layer3:
        PnL comparison result.
    verdict:
        ``"PASS"`` or ``"FAIL"``.
    fail_reasons:
        List of human-readable failure reasons.
    reference_gaps:
        Holes in the recorded reference feed over the window. When non-empty, any
        fill/decision divergence is likely data-caused (sim held a stale reference
        while live tracked the true price), not a strategy or harness fault.
    """

    question_idx: int
    expiry_ns: int
    layer0: PreconditionResult
    layer1: DecisionResult
    layer2: FillsResult
    layer3: PnLResult
    verdict: str
    fail_reasons: list[str] = field(default_factory=list)
    reference_gaps: list[ReferenceGap] = field(default_factory=list)


def verdict(
    layer0: PreconditionResult,
    layer1: DecisionResult,
    layer2: FillsResult,
    layer3: PnLResult,
    pnl_abs_threshold: float = 5.0,
    decision_match_min: float = 0.95,
    decision_coverage_min: float = 0.5,
) -> tuple[str, list[str]]:
    """Apply thresholds and return overall PASS/FAIL + list of failure reasons.

    Parameters
    ----------
    layer0:
        Precondition result.
    layer1:
        Decision result.
    layer2:
        Fill result (included in report but not a hard gate).
    layer3:
        PnL result.
    pnl_abs_threshold:
        Maximum acceptable |pnl_diff|.
    decision_match_min:
        Minimum acceptable decision match_rate.
    decision_coverage_min:
        Minimum fraction of buckets that must be ALIGNED across both traces. The
        match rate is computed only over aligned buckets, so a sim trace that is
        truncated/misaligned can match 100% of its few overlapping buckets and
        sail through (audit H3). Require the aligned set to cover a real share of
        the larger trace before trusting the match rate.

    Returns
    -------
    Tuple of (verdict_str, fail_reasons_list).  PASS requires all gates to clear.
    """
    reasons: list[str] = []

    if layer0.overall == "FAIL":
        if layer0.config_hash_match.startswith("FAIL"):
            reasons.append(f"config_hash: {layer0.config_hash_match}")
        if layer0.question_identity_match.startswith("FAIL"):
            reasons.append(f"question_identity: {layer0.question_identity_match}")
        if layer0.window_match.startswith("FAIL"):
            reasons.append(f"window_overlap: {layer0.window_match}")

    coverage = layer1.n_aligned / max(layer1.n_live_buckets, layer1.n_sim_buckets, 1)
    if coverage < decision_coverage_min:
        reasons.append(
            f"decision_coverage={coverage:.2%} < {decision_coverage_min:.0%} "
            "(sim trace truncated or misaligned — match rate is over too few buckets)"
        )

    if layer1.match_rate < decision_match_min:
        reasons.append(f"decision_match_rate={layer1.match_rate:.2%} < {decision_match_min:.0%}")

    if layer3.pnl_match.startswith("FAIL"):
        reasons.append(f"pnl_diff: {layer3.pnl_match}")

    overall = "FAIL" if reasons else "PASS"
    return overall, reasons


def run_reconcile(
    question_idx: int,
    expiry_ns: int,
    live_fills: pd.DataFrame,
    live_trace: pd.DataFrame,
    live_settlement: dict[str, Any],
    live_config_hash: str | None,
    sim_fills: pd.DataFrame,
    sim_trace: pd.DataFrame,
    sim_resolved: dict[str, Any],
    data_root: Path | None = None,
    book_reader: BookReader | None = None,
    pnl_abs_threshold: float = 5.0,
    decision_match_min: float = 0.95,
    ref_symbol: str = "BTC",
    reference_ts_reader: ReferenceTsReader | None = None,
    ref_price_reader: RefPriceReader | None = None,
) -> ReconcileResult:
    """Run all 4 layers and return a ReconcileResult.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    live_fills:
        Venue-confirmed live fills.
    live_trace:
        Live decision trace.
    live_settlement:
        Live settlement dict.
    live_config_hash:
        Config hash from the live engine.
    sim_fills:
        Sim fills.
    sim_trace:
        Sim decision trace.
    sim_resolved:
        Sim resolution dict with winner_side and resolved_outcome.
    data_root:
        Root of HL recorded data (for book parity checks).
    book_reader:
        Injectable book reader for testing.
    pnl_abs_threshold:
        Maximum acceptable |pnl_diff| for PASS.
    decision_match_min:
        Minimum acceptable decision match_rate for PASS.

    Returns
    -------
    ReconcileResult with all 4 layers populated and a final verdict.
    """
    sim_config_hash: str | None = None
    if "config_hash" in sim_trace.columns and not sim_trace.empty:
        sim_config_hash = str(sim_trace["config_hash"].iloc[-1])

    layer0 = check_preconditions(
        live_trace=live_trace,
        sim_trace=sim_trace,
        live_config_hash=live_config_hash,
        sim_config_hash=sim_config_hash,
    )
    layer1 = reconcile_decisions(live_trace=live_trace, sim_trace=sim_trace)
    layer2 = reconcile_fills(
        live_fills=live_fills,
        sim_fills=sim_fills,
        data_root=data_root,
        book_reader=book_reader,
    )
    layer3 = reconcile_pnl(
        live_fills=live_fills,
        sim_fills=sim_fills,
        live_settlement=live_settlement,
        sim_resolved=sim_resolved,
        pnl_abs_threshold=pnl_abs_threshold,
        expiry_ns=expiry_ns,
        ref_symbol=ref_symbol,
        data_root=data_root,
        ref_price_reader=ref_price_reader,
    )

    v, fail_reasons = verdict(
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        pnl_abs_threshold=pnl_abs_threshold,
        decision_match_min=decision_match_min,
    )

    # Layer 2.5: reference-feed coverage over the live trace window. A gap here
    # means the sim replayed a stale (hold-last) reference while live tracked the
    # true price — the usual cause of fill/decision divergence that is data-driven,
    # not a strategy or harness fault. Surface it so a divergent verdict can be
    # attributed rather than mistaken for a regression.
    reference_gaps: list[ReferenceGap] = []
    if (
        (data_root is not None or reference_ts_reader is not None)
        and not live_trace.empty
        and "ts_ns" in live_trace.columns
    ):
        win_lo = int(live_trace["ts_ns"].min())
        win_hi = int(live_trace["ts_ns"].max())
        reference_gaps = check_reference_coverage(
            start_ns=win_lo,
            end_ns=win_hi,
            ref_symbol=ref_symbol,
            data_root=data_root,
            ts_reader=reference_ts_reader,
        )

    return ReconcileResult(
        question_idx=question_idx,
        expiry_ns=expiry_ns,
        layer0=layer0,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        verdict=v,
        fail_reasons=fail_reasons,
        reference_gaps=reference_gaps,
    )
