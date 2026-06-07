import asyncio

from hlanalysis.engine.exec_types import ClearinghouseState, VenuePosition
from hlanalysis.engine.reconcile_report import SlotRecon, compare_slot


def _vp(sym, qty, avg, upnl):
    return VenuePosition(symbol=sym, qty=qty, avg_entry=avg, unrealized_pnl=upnl)


def test_compare_slot_clean_no_drift():
    # DB and venue agree on one BTC position.
    r = compare_slot(
        alias="v1",
        db_positions=[("BTC", 100.0)],          # (symbol, qty)
        db_realized_pnl=120.0,
        venue=ClearinghouseState(
            positions=(_vp("BTC", 100.0, 0.9, 5.0),),
            account_value_usd=1000.0,
        ),
        qty_tolerance=1e-6,
    )
    assert isinstance(r, SlotRecon)
    assert r.realized_pnl == 120.0
    assert r.open_mtm == 5.0
    assert r.total_true_pnl == 125.0
    assert r.account_value_usd == 1000.0
    assert r.drift == []          # no drift
    assert r.has_drift is False


def test_compare_slot_qty_mismatch_is_drift():
    r = compare_slot(
        alias="v1",
        db_positions=[("BTC", 100.0)],
        db_realized_pnl=0.0,
        venue=ClearinghouseState(positions=(_vp("BTC", 60.0, 0.9, 0.0),),
                                 account_value_usd=1.0),
        qty_tolerance=1e-6,
    )
    assert r.has_drift is True
    assert any(d.kind == "qty_mismatch" and d.symbol == "BTC" for d in r.drift)


def test_compare_slot_vanished_and_orphan():
    # DB has ETH the venue doesn't (vanished); venue has SOL the DB doesn't (orphan).
    r = compare_slot(
        alias="v31",
        db_positions=[("ETH", 50.0)],
        db_realized_pnl=0.0,
        venue=ClearinghouseState(positions=(_vp("SOL", 10.0, 1.0, 0.0),),
                                 account_value_usd=1.0),
        qty_tolerance=1e-6,
    )
    kinds = {(d.kind, d.symbol) for d in r.drift}
    assert ("vanished", "ETH") in kinds
    assert ("orphan", "SOL") in kinds


def test_compare_slot_skips_when_positions_unknown():
    # PM data-api flap: positions_known=False → DO NOT treat empty as truth.
    r = compare_slot(
        alias="v31_pm",
        db_positions=[("UPDOWN", 25.0)],
        db_realized_pnl=3.0,
        venue=ClearinghouseState(positions=(), account_value_usd=0.0,
                                 positions_known=False),
        qty_tolerance=1e-6,
    )
    assert r.positions_known is False
    assert r.drift == []          # position recon skipped, no false 'vanished'
    assert r.realized_pnl == 3.0  # PnL still reported


from hlanalysis.engine.reconcile_report import format_report, Drift


def test_format_report_clean():
    recon = [
        SlotRecon(alias="v1", realized_pnl=120.0, open_mtm=5.0,
                  account_value_usd=1000.0, positions_known=True, drift=[]),
    ]
    text = format_report(recon)
    assert "v1" in text
    assert "120" in text          # realized
    assert "125" in text          # total true pnl
    assert "OK" in text


def test_format_report_flags_drift():
    recon = [
        SlotRecon(alias="v31", realized_pnl=0.0, open_mtm=0.0,
                  account_value_usd=1.0, positions_known=True,
                  drift=[Drift("vanished", "ETH", 50.0, 0.0)]),
    ]
    text = format_report(recon)
    assert "DRIFT" in text
    assert "ETH" in text
    assert "vanished" in text


def test_format_report_marks_unknown_positions():
    recon = [
        SlotRecon(alias="v31_pm", realized_pnl=3.0, open_mtm=0.0,
                  account_value_usd=0.0, positions_known=False, drift=[]),
    ]
    text = format_report(recon)
    assert "positions unknown" in text.lower() or "skipped" in text.lower()


def test_gather_slot_uses_client_and_dal():
    from hlanalysis.engine.reconcile_report import gather_slot

    class FakeDAL:
        def realized_pnl_since(self, since_ts_ns):
            return 42.0
        def all_positions(self):
            class P:  # minimal Position stand-in
                symbol = "BTC"; qty = 100.0
            return [P()]

    class FakeClient:
        def clearinghouse_state(self):
            return ClearinghouseState(positions=(), account_value_usd=7.0)

    r = gather_slot(alias="v1", dal=FakeDAL(), exec_client=FakeClient(),
                    qty_tolerance=1e-6)
    assert r.alias == "v1"
    assert r.realized_pnl == 42.0
    assert r.account_value_usd == 7.0
    # DB has BTC, venue has none → vanished drift
    assert any(d.kind == "vanished" and d.symbol == "BTC" for d in r.drift)


def test_alert_sends_only_on_drift(monkeypatch):
    import hlanalysis.engine.reconcile_report as rr

    sent: list[str] = []

    class FakeTG:
        def __init__(self, **kw): ...
        async def send(self, text, *, markdown=True):
            sent.append(text); return True

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")

    # Use injection seam directly — avoids fragile module-attribute patching.
    asyncio.run(rr._maybe_alert(
        "report", has_drift=False,
        tg_factory=FakeTG, session_factory=lambda: FakeSession(),
    ))
    assert sent == []                       # no drift → no alert

    asyncio.run(rr._maybe_alert(
        "report", has_drift=True,
        tg_factory=FakeTG, session_factory=lambda: FakeSession(),
    ))
    assert len(sent) == 1 and "DRIFT" in sent[0]


def test_compare_slot_flags_pnl_mismatch_and_prefers_venue():
    # Local ledger says -421 (corrupted by bad settlement rows); venue says +198.
    r = compare_slot(
        alias="v1",
        db_positions=[],
        db_realized_pnl=-421.49,
        venue=ClearinghouseState(positions=(), account_value_usd=0.0),
        qty_tolerance=1e-6,
        venue_realized_pnl=198.41,
        pnl_tolerance=1.0,
    )
    assert r.pnl_mismatch is True          # local vs venue diverge > $1
    assert r.has_drift is True             # pnl mismatch counts as drift
    assert r.venue_realized_pnl == 198.41
    # total_true_pnl prefers the authoritative venue figure, not the local one
    assert r.total_true_pnl == 198.41      # + open_mtm 0
    text = format_report([r])
    assert "pnl_mismatch" in text and "DRIFT" in text


def test_compare_slot_no_pnl_mismatch_within_tolerance():
    r = compare_slot(
        alias="v31",
        db_positions=[],
        db_realized_pnl=100.0,
        venue=ClearinghouseState(positions=(), account_value_usd=50.0),
        qty_tolerance=1e-6,
        venue_realized_pnl=100.4,          # within $1 tolerance
        pnl_tolerance=1.0,
    )
    assert r.pnl_mismatch is False
    assert r.has_drift is False
    assert r.total_true_pnl == 100.4       # still prefers venue
