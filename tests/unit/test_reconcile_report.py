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

    # Creds passed explicitly (engine's TG_BOT_TOKEN/TG_CHAT_ID, not env TELEGRAM_*).
    asyncio.run(rr._maybe_alert(
        "report", has_drift=False, bot_token="t", chat_id="c",
        tg_factory=FakeTG, session_factory=lambda: FakeSession(),
    ))
    assert sent == []                       # no drift → no alert

    asyncio.run(rr._maybe_alert(
        "report", has_drift=True, bot_token="t", chat_id="c",
        tg_factory=FakeTG, session_factory=lambda: FakeSession(),
    ))
    assert len(sent) == 1 and "DRIFT" in sent[0]


def test_post_tg_noop_without_creds():
    import hlanalysis.engine.reconcile_report as rr
    sent = []

    class FakeTG:
        def __init__(self, **kw): ...
        async def send(self, text, *, markdown=True):
            sent.append(text); return True

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    ok = asyncio.run(rr._post_tg("x", bot_token=None, chat_id=None,
                                 tg_factory=FakeTG, session_factory=lambda: FakeSession()))
    assert ok is False and sent == []       # missing creds → no send


def test_format_daily_summary_single_message():
    from hlanalysis.engine.reconcile_report import format_daily_summary
    recon = [
        SlotRecon(alias="v1_pm", realized_pnl=2.17, open_mtm=0.0,
                  account_value_usd=121.0, positions_known=True, fills_count=8),
        SlotRecon(alias="v31", realized_pnl=0.0, open_mtm=0.0,
                  account_value_usd=1299.0, positions_known=True,
                  venue_realized_pnl=138.12, fills_count=527),
    ]
    msg = format_daily_summary(recon, date_str="2026-06-08")
    # ONE message containing every strategy + a desk total
    assert msg.count("Desk daily report") == 1
    assert "v1_pm" in msg and "v31" in msg
    assert "fills 8" in msg and "fills 527" in msg
    assert "Total strategy PnL: +140.29" in msg   # 2.17 + 138.12
    assert "all reconciled" in msg


def test_format_daily_summary_splits_hl_by_klass():
    # SHR-77: HL slots show their outcome PnL + fills split into binary vs
    # bucket; PM slots (no breakdown) stay a single line; desk total unchanged.
    from hlanalysis.engine.reconcile_report import KlassStat, format_daily_summary
    recon = [
        SlotRecon(alias="v1_pm", realized_pnl=2.17, open_mtm=0.0,
                  account_value_usd=121.0, positions_known=True, fills_count=8),
        SlotRecon(alias="v31", realized_pnl=0.0, open_mtm=18.12,
                  account_value_usd=1299.0, positions_known=True,
                  venue_realized_pnl=120.0, fills_count=527,
                  klass_breakdown={
                      "priceBinary": KlassStat(realized_pnl=100.0, open_mtm=0.0, fills=500),
                      "priceBucket": KlassStat(realized_pnl=20.0, open_mtm=18.12, fills=27),
                  }),
    ]
    msg = format_daily_summary(recon, date_str="2026-06-08")
    # PM slot is unchanged (single line, no split sub-lines under it).
    assert "v1_pm: PnL +2.17" in msg
    # HL slot keeps its headline AND gains a binary/bucket split.
    assert "v31: PnL +138.12" in msg          # 120.0 + 18.12 total_true_pnl
    assert "binary" in msg and "bucket" in msg
    assert "+100.00" in msg                    # binary total_pnl
    assert "+38.12" in msg                     # bucket total_pnl 20.0 + 18.12
    assert "fills 500" in msg and "fills 27" in msg
    # Desk total unchanged: 2.17 + 138.12
    assert "Total strategy PnL: +140.29" in msg


def test_format_daily_summary_shows_unknown_klass():
    # Unmapped coins land in an explicit "unknown" bucket — never silently
    # folded into binary/bucket on a money report.
    from hlanalysis.engine.reconcile_report import KlassStat, format_daily_summary
    recon = [
        SlotRecon(alias="v31", realized_pnl=0.0, open_mtm=0.0,
                  account_value_usd=10.0, positions_known=True,
                  venue_realized_pnl=5.0, fills_count=2,
                  klass_breakdown={
                      "priceBinary": KlassStat(realized_pnl=3.0, open_mtm=0.0, fills=1),
                      "unknown": KlassStat(realized_pnl=2.0, open_mtm=0.0, fills=1),
                  }),
    ]
    msg = format_daily_summary(recon)
    assert "unknown" in msg
    assert "+2.00" in msg


def test_gather_slot_splits_outcome_fills_by_klass():
    from hlanalysis.engine.reconcile_report import gather_slot

    class P:
        def __init__(self, symbol, qty):
            self.symbol = symbol; self.qty = qty

    class FakeDAL:
        def realized_pnl_since(self, since_ts_ns):
            return 0.0
        def all_positions(self):
            return [P("#160", 5.0)]
        def coin_klass_map(self):
            return {"#150": "priceBinary", "#151": "priceBinary",
                    "#160": "priceBucket"}

    class FakeFill:
        def __init__(self, symbol, closed_pnl, fee):
            self.symbol = symbol; self.closed_pnl = closed_pnl; self.fee = fee

    class FakeClient:
        def clearinghouse_state(self):
            return ClearinghouseState(
                positions=(_vp("#160", 5.0, 0.4, 1.5),),
                account_value_usd=100.0,
            )
        def user_fills(self, *, since_ts_ns):
            return [
                FakeFill("#150", 10.0, 0.5),   # binary
                FakeFill("#151", 4.0, 0.0),    # binary
                FakeFill("#160", 6.0, 0.0),    # bucket
                FakeFill("@7", 99.0, 0.0),     # non-outcome spot — excluded
            ]

    r = gather_slot(alias="v31", dal=FakeDAL(), exec_client=FakeClient(),
                    qty_tolerance=1e-6, fetch_venue_realized=True)
    bd = r.klass_breakdown
    assert bd is not None
    # binary: realized = (10-0.5) + 4 = 13.5, 2 fills, no open mtm
    assert bd["priceBinary"].realized_pnl == 13.5
    assert bd["priceBinary"].fills == 2
    assert bd["priceBinary"].open_mtm == 0.0
    # bucket: realized = 6.0, 1 fill, open mtm 1.5 from the "#160" venue position
    assert bd["priceBucket"].realized_pnl == 6.0
    assert bd["priceBucket"].fills == 1
    assert bd["priceBucket"].open_mtm == 1.5
    # Outcome-only fill count excludes the "@7" spot fill.
    assert r.fills_count == 3


def test_gather_slot_unmapped_fill_is_unknown():
    from hlanalysis.engine.reconcile_report import gather_slot

    class FakeDAL:
        def realized_pnl_since(self, since_ts_ns):
            return 0.0
        def all_positions(self):
            return []
        def coin_klass_map(self):
            return {}   # nothing mapped

    class FakeFill:
        def __init__(self, symbol, closed_pnl, fee):
            self.symbol = symbol; self.closed_pnl = closed_pnl; self.fee = fee

    class FakeClient:
        def clearinghouse_state(self):
            return ClearinghouseState(positions=(), account_value_usd=1.0)
        def user_fills(self, *, since_ts_ns):
            return [FakeFill("#999", 7.0, 0.0)]

    r = gather_slot(alias="v31", dal=FakeDAL(), exec_client=FakeClient(),
                    qty_tolerance=1e-6, fetch_venue_realized=True)
    assert r.klass_breakdown is not None
    assert "unknown" in r.klass_breakdown
    assert r.klass_breakdown["unknown"].realized_pnl == 7.0
    assert r.klass_breakdown["unknown"].fills == 1


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


def test_strategy_pnl_is_outcome_only_not_full_account():
    # true_pnl must be the OUTCOME-only venue realized ($161.04), NOT the
    # full-account allTime equity PnL ($362.68) which nets in non-strategy
    # perp/spot trades. allTime is shown only as labelled context.
    r = compare_slot(
        alias="v1", db_positions=[], db_realized_pnl=161.04,
        venue=ClearinghouseState(positions=(), account_value_usd=894.07),
        qty_tolerance=1e-6, venue_realized_pnl=161.04, pnl_tolerance=1.0,
        account_pnl_all_time=362.68,
    )
    assert r.total_true_pnl == 161.04          # outcome-only, NOT 362.68
    assert r.pnl_mismatch is False             # local==venue outcome realized
    text = format_report([r])
    assert "161.04" in text
    assert "strategy_pnl(outcome-only)" in text
    assert "362.68" in text and "non-strategy" in text  # full-account = context
