from __future__ import annotations

import pytest
from pydantic import ValidationError

from hlanalysis.sim.data.schemas import PMMarket, PMTrade


def test_pmmarket_validates_fields():
    m = PMMarket(
        condition_id="0xabc",
        yes_token_id="1",
        no_token_id="2",
        start_ts_ns=0,
        end_ts_ns=86_400_000_000_000,
        resolved_outcome="yes",
        total_volume_usd=12345.0,
        n_trades=42,
    )
    assert m.condition_id == "0xabc"
    assert m.resolved_outcome == "yes"


def test_pmmarket_rejects_bad_outcome():
    with pytest.raises(ValidationError):
        PMMarket(
            condition_id="x",
            yes_token_id="1",
            no_token_id="2",
            start_ts_ns=0,
            end_ts_ns=1,
            resolved_outcome="maybe",  # type: ignore[arg-type]
            total_volume_usd=0,
            n_trades=0,
        )


def test_pmtrade_validates_price_range():
    t = PMTrade(ts_ns=1, token_id="1", side="buy", price=0.5, size=10.0)
    assert 0 <= t.price <= 1
