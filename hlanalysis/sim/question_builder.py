from __future__ import annotations

from hlanalysis.strategy.types import QuestionView

from .data.schemas import PMMarket


def build_question_view(market: PMMarket, *, day_open_btc: float, now_ns: int) -> QuestionView:
    settled = now_ns > market.end_ts_ns
    settled_side = market.resolved_outcome if settled else None
    return QuestionView(
        question_idx=hash(market.condition_id) & 0x7FFF_FFFF,
        yes_symbol=market.yes_token_id,
        no_symbol=market.no_token_id,
        strike=day_open_btc,
        expiry_ns=market.end_ts_ns,
        underlying="BTC",
        klass="priceBinary",
        period="24h",
        settled=settled,
        settled_side=settled_side,
    )
