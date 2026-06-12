"""Reconstruct the sim's held-position timeline for engine-replay parity (R3 v2).

The engine ``ReplayRunner`` accepts a static ``position_lookup: dict[int, Position]``
that it consults on every ``strategy.evaluate()`` call.  Feeding it the sim's
ACTUAL position at each engine-clock instant makes the two paths structurally
equivalent — they model the same held-position context — and converts the gate
from a σ-only check into a true decision-equivalence gate.

``build_position_timeline`` reconstructs, from the sim's fills-parquet (timestamped
buy/sell records written by ``run_one_question`` when ``fills_dir`` is provided),
the full position state for each question at every point in time.  The result is a
``PositionTimeline``: a sorted list of ``(ts_ns, Position | None)`` change-points.
At any instant ``T`` the sim's held position is the entry whose ``ts_ns`` is the
largest that is ``<= T``.

``current_position_at`` is the cheap O(log n) lookup helper for engine replays.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from hlanalysis.marketdata.position_math import (
    STOP_DISABLED_SENTINEL,
    PositionState,
    apply_fill,
)
from hlanalysis.strategy.types import Position


# ---------------------------------------------------------------------------
# Change-point timeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PositionChange:
    """One position change-point on the sim's timeline for a single question."""
    ts_ns: int
    position: Position | None   # None ⟹ no position held after this ts


@dataclass
class PositionTimeline:
    """Sorted sequence of position change-points for one ``question_idx``.

    Sorted ascending by ``ts_ns`` (guaranteed by the constructor).
    Empty list ⟹ the sim never held a position for this question.
    """
    question_idx: int
    changes: list[PositionChange]

    def __post_init__(self) -> None:
        self.changes.sort(key=lambda c: c.ts_ns)
        self._ts_list: list[int] = [c.ts_ns for c in self.changes]

    def current_at(self, ts_ns: int) -> Position | None:
        """Return the position the sim held at ``ts_ns`` (or None if flat).

        Uses binary search: O(log n) per call.
        """
        if not self.changes:
            return None
        # bisect_right gives the insertion point AFTER all entries with ts <= ts_ns.
        # The preceding entry (i - 1) is the most recent change at or before ts_ns.
        i = bisect.bisect_right(self._ts_list, ts_ns)
        if i == 0:
            return None   # ts_ns is before the first change-point ⟹ flat
        return self.changes[i - 1].position


# ---------------------------------------------------------------------------
# Reconstruction from fills-parquet rows
# ---------------------------------------------------------------------------

def build_position_timeline(
    fill_rows: list[dict[str, Any]],
    *,
    question_idx: int,
    stop_loss_pct: float | None = None,
) -> PositionTimeline:
    """Reconstruct the position timeline for ``question_idx`` from fill rows.

    ``fill_rows`` should be a list of dicts with at minimum the keys:
        ``ts_ns`` (int), ``side`` ("buy"/"sell"), ``price`` (float),
        ``size`` (float), ``symbol`` (str), ``cloid`` (str).

    Rows should be sorted by ``ts_ns`` ascending.  The "settle" fill (cloid
    "settle") is INCLUDED — it closes the final position at settlement time.

    A ``PositionChange`` is appended at each fill's ``ts_ns`` reflecting the
    new position STATE after that fill.  A fill that closes the position fully
    (new_pos is None from ``apply_fill``) emits a ``PositionChange`` with
    ``position=None`` (flat).
    """
    pos_state: PositionState | None = None
    pos_symbol: str = ""
    changes: list[PositionChange] = []

    for row in fill_rows:
        ts_ns = int(row["ts_ns"])
        side = str(row["side"])
        price = float(row["price"])
        size = float(row["size"])
        symbol = str(row["symbol"])

        # Settlement fill: apply and mark flat — same as any close.
        new_state, _ = apply_fill(pos_state, side, size, price)

        if new_state is None or new_state.qty == 0.0:
            # Position fully closed.
            pos_state = None
            pos_symbol = ""
            changes.append(PositionChange(ts_ns=ts_ns, position=None))
        else:
            # Position open or reduced-but-not-closed.
            pos_state = new_state
            pos_symbol = symbol
            stop_px = (
                STOP_DISABLED_SENTINEL if stop_loss_pct is None
                else max(0.0, new_state.avg_entry * (1.0 - stop_loss_pct / 100.0))
            )
            changes.append(PositionChange(
                ts_ns=ts_ns,
                position=Position(
                    question_idx=question_idx,
                    symbol=pos_symbol,
                    qty=new_state.qty,
                    avg_entry=new_state.avg_entry,
                    stop_loss_price=stop_px,
                    last_update_ts_ns=ts_ns,
                ),
            ))

    return PositionTimeline(question_idx=question_idx, changes=changes)


def build_timelines_from_fills_parquet(
    parquet_path: "str | Any",
    *,
    stop_loss_pct: float | None = None,
) -> dict[int, PositionTimeline]:
    """Load fill rows from a fills parquet file and reconstruct all position
    timelines grouped by ``question_idx``.

    Returns a mapping ``{question_idx: PositionTimeline}`` covering every
    question that had at least one fill in the parquet.

    Requires ``duckdb`` (already a project dependency).
    """
    import duckdb

    path = str(parquet_path)
    rows = duckdb.connect().execute(
        f"SELECT ts_ns, side, price, size, symbol, cloid, question_idx "
        f"FROM read_parquet('{path}') ORDER BY ts_ns"
    ).fetchall()
    cols = ["ts_ns", "side", "price", "size", "symbol", "cloid", "question_idx"]

    # Group by question_idx; hedge fills (is_hedge=True) are skipped because
    # the binary position math doesn't apply them.
    by_qi: dict[int, list[dict]] = {}
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        qi = int(d["question_idx"])
        by_qi.setdefault(qi, []).append(d)

    timelines: dict[int, PositionTimeline] = {}
    for qi, qi_rows in by_qi.items():
        timelines[qi] = build_position_timeline(
            qi_rows, question_idx=qi, stop_loss_pct=stop_loss_pct
        )
    return timelines


__all__ = [
    "PositionChange",
    "PositionTimeline",
    "build_position_timeline",
    "build_timelines_from_fills_parquet",
]
