"""Per-scan decision trace writer for ``--decision-trace-out``.

Writes one JSONL row per ``strategy.evaluate()`` call with a canonical schema
that captures all diagnostic signal emitted by both theta_harvester and
late_resolution strategies.  Zero overhead when no writer is active.

Schema (all fields nullable except ts_ns/question_idx/action):
    ts_ns              int   — nanosecond epoch of the scan tick
    question_idx       int   — stable HL question index
    klass              str   — question class (priceBinary / priceBucket)
    strategy_id        str   — registered strategy id
    action             str   — "enter" / "exit" / "hold"
    reason             str   — first diagnostic message (e.g. "edge", "no_favorite")
    chosen_symbol      str   — symbol of the chosen entry/exit leg (or "")
    chosen_side        str   — "buy" or "sell" for the chosen leg (or "")
    reference_price    float — reference price fed to strategy.evaluate
    sigma              float — annualised σ from the "edge" diagnostic
    p_model            float — model win probability from the "edge" diagnostic
    edge               float — chosen-side edge (edge_yes or chosen_edge)
    safety_d_entry     float — entry safety_d from the "safety_d_entry" diagnostic
    safety_d_exit      float — exit safety_d from the "safety_d_exit" diagnostic
    tte_s              float — time-to-expiry in seconds from "tte_out_of_window" or entry diag
    favorite_side      str   — "yes" or "no" for the chosen leg (or "")
    intended_size      float — size from the first OrderIntent (or null)
    intended_price     float — limit_price from the first OrderIntent (or null)
    bid_px             float — best bid of the chosen leg (or null)
    bid_sz             float — best bid size of the chosen leg (or null)
    ask_px             float — best ask of the chosen leg (or null)
    ask_sz             float — best ask size of the chosen leg (or null)
    position_qty       float — current position qty (or null when flat)
    position_avg_entry float — current position avg entry price (or null when flat)
    config_hash        str   — strategy_config_sig hash (or "")
    diag_fields        dict  — full dict of all diagnostic kv pairs across all diagnostics
"""

from __future__ import annotations

import json
from io import TextIOWrapper
from pathlib import Path

from hlanalysis.strategy.types import BookState, Decision, Position, QuestionView

# ---------------------------------------------------------------------------
# Row extraction helpers
# ---------------------------------------------------------------------------


def _collect_diag_fields(decision: Decision) -> dict[str, str]:
    """Merge all diagnostic kv pairs from all diagnostics into one flat dict.

    Later diagnostics with the same key win (preserving order priority).
    """
    merged: dict[str, str] = {}
    for diag in decision.diagnostics:
        for k, v in diag.fields:
            merged[k] = v
    return merged


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def build_trace_row(
    *,
    ts_ns: int,
    question: QuestionView,
    strategy_id: str,
    decision: Decision,
    reference_price: float,
    books: dict[str, BookState],
    position: Position | None,
    config_hash: str,
) -> dict:
    """Extract one JSONL row from a strategy.evaluate() call."""
    diag_fields = _collect_diag_fields(decision)
    reason = decision.diagnostics[0].message if decision.diagnostics else ""

    # Chosen symbol + side from the first OrderIntent (if any).
    chosen_symbol = ""
    chosen_side = ""
    intended_size: float | None = None
    intended_price: float | None = None
    if decision.intents:
        intent = decision.intents[0]
        chosen_symbol = intent.symbol
        chosen_side = intent.side
        intended_size = intent.size
        intended_price = intent.limit_price

    # Resolve favorite_side from chosen_symbol vs yes_symbol.
    if chosen_symbol and question.yes_symbol:
        favorite_side = "yes" if chosen_symbol == question.yes_symbol else "no"
    else:
        favorite_side = ""

    # Book state for the chosen leg.
    chosen_book = books.get(chosen_symbol) if chosen_symbol else None
    bid_px: float | None = chosen_book.bid_px if chosen_book is not None else None
    bid_sz: float | None = chosen_book.bid_sz if chosen_book is not None else None
    ask_px: float | None = chosen_book.ask_px if chosen_book is not None else None
    ask_sz: float | None = chosen_book.ask_sz if chosen_book is not None else None

    # Position state.
    position_qty: float | None = position.qty if position is not None else None
    position_avg_entry: float | None = position.avg_entry if position is not None else None

    # Parse numeric fields from diag_fields.
    sigma = _float_or_none(diag_fields.get("sigma"))
    p_model = _float_or_none(diag_fields.get("p_model"))

    # Edge: prefer chosen_edge (bucket diagnostics), fall back to edge_yes.
    edge_raw = diag_fields.get("chosen_edge") or diag_fields.get("edge_yes")
    edge = _float_or_none(edge_raw)

    # safety_d fields (from named diagnostic messages).
    safety_d_entry: float | None = None
    safety_d_exit: float | None = None
    tte_s: float | None = None
    for diag in decision.diagnostics:
        msg = diag.message
        fdict = dict(diag.fields)
        if msg == "safety_d_entry":
            safety_d_entry = _float_or_none(fdict.get("d"))
        elif msg in ("safety_d_exit", "exit_safety_d_below_min", "exit_safety_d_5m_below_min"):
            safety_d_exit = _float_or_none(fdict.get("d"))
        elif msg == "tte_out_of_window":
            tte_s = _float_or_none(fdict.get("tte_s"))

    # tte_s fallback: derive from question expiry.
    if tte_s is None and question.expiry_ns > 0 and ts_ns > 0:
        tte_s = max(0.0, (question.expiry_ns - ts_ns) / 1e9)

    return {
        "ts_ns": ts_ns,
        "question_idx": question.question_idx,
        "klass": question.klass,
        "strategy_id": strategy_id,
        "action": decision.action.value if hasattr(decision.action, "value") else str(decision.action),
        "reason": reason,
        "chosen_symbol": chosen_symbol,
        "chosen_side": chosen_side,
        "reference_price": reference_price,
        "sigma": sigma,
        "p_model": p_model,
        "edge": edge,
        "safety_d_entry": safety_d_entry,
        "safety_d_exit": safety_d_exit,
        "tte_s": tte_s,
        "favorite_side": favorite_side,
        "intended_size": intended_size,
        "intended_price": intended_price,
        "bid_px": bid_px,
        "bid_sz": bid_sz,
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "position_qty": position_qty,
        "position_avg_entry": position_avg_entry,
        "config_hash": config_hash,
        "diag_fields": diag_fields,
    }


# ---------------------------------------------------------------------------
# Writer class
# ---------------------------------------------------------------------------


class DecisionTraceWriter:
    """Append-mode JSONL writer for per-scan decision trace rows.

    Open once per run (across all questions) and close when done.  Each call
    to :meth:`write` serialises one row and flushes immediately so partial
    output is always readable.

    Usage::

        with DecisionTraceWriter("/tmp/trace.jsonl") as writer:
            for q in questions:
                run_one_question(..., decision_trace_writer=writer)
    """

    __slots__ = ("_path", "_fh")

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._fh: TextIOWrapper | None = None

    def open(self) -> DecisionTraceWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> DecisionTraceWriter:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def write(self, row: dict) -> None:
        """Serialise one row as JSONL and flush."""
        if self._fh is not None:
            self._fh.write(json.dumps(row, default=str))
            self._fh.write("\n")
            self._fh.flush()


__all__ = [
    "DecisionTraceWriter",
    "build_trace_row",
]
