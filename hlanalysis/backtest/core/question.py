"""QuestionView re-export + a shared builder used by both PM and HL sources.

`QuestionView` already abstracts over PM and HL markets — it lives in the
strategy package because strategies consume it. The builder here is the seam
each `DataSource` uses to construct a view from a `QuestionDescriptor` plus
runtime context (now_ns, settlement state, klass-specific kv).
"""
from __future__ import annotations

from typing import Literal

from hlanalysis.strategy.types import QuestionView

from .data_source import QuestionDescriptor


def build_question_view(
    q: QuestionDescriptor,
    *,
    now_ns: int,
    strike: float,
    period: str = "24h",
    settled: bool | None = None,
    settled_side: Literal["yes", "no", "unknown"] | None = None,
    name: str = "",
    kv: tuple[tuple[str, str], ...] = (),
) -> QuestionView:
    """Construct a QuestionView consistent with the strategy's contract.

    ``settled`` defaults to ``now_ns > end_ts_ns`` when not supplied. The
    first two ``leg_symbols`` are mapped to ``yes_symbol`` / ``no_symbol``
    for binary klass; for ``priceBucket`` these stay empty because they have
    no semantic meaning across more than two legs.
    """
    if settled is None:
        settled = now_ns > q.end_ts_ns

    if q.klass == "priceBinary":
        yes_symbol = q.leg_symbols[0] if len(q.leg_symbols) >= 1 else ""
        no_symbol = q.leg_symbols[1] if len(q.leg_symbols) >= 2 else ""
    else:
        yes_symbol = ""
        no_symbol = ""

    return QuestionView(
        question_idx=q.question_idx,
        yes_symbol=yes_symbol,
        no_symbol=no_symbol,
        strike=strike,
        expiry_ns=q.end_ts_ns,
        underlying=q.underlying,
        klass=q.klass,
        period=period,
        settled=settled,
        settled_side=settled_side,
        leg_symbols=q.leg_symbols,
        name=name,
        kv=kv,
    )


__all__ = ["QuestionView", "build_question_view"]
