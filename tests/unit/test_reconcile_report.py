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


def test_compare_slot_pm_subshare_rounding_not_drift():
    # PM data-api settled size vs our booked size routinely differ by ~8e-3
    # shares (2dp sell-floor + indexer rounding). The live engine reconcile
    # tolerates this (abs_tol=2e-2); the report must match so PM slots don't
    # false-flag DRIFT every cycle (incident 2026-06-08 v31_pm 56.1685/56.1764).
    r = compare_slot(
        alias="v31_pm",
        db_positions=[("0xTOK", 56.168538)],
        db_realized_pnl=17.90,
        venue=ClearinghouseState(
            positions=(_vp("0xTOK", 56.1764, 0.5, -0.28),),
            account_value_usd=188.0,
        ),
        qty_tolerance=2e-2,   # engine abs_tol
    )
    assert r.drift == []
    assert r.has_drift is False


def test_compare_slot_real_qty_diff_flags_at_engine_tol():
    # A real missed fill (~1 share) still flags at the engine tolerance.
    r = compare_slot(
        alias="v1",
        db_positions=[("#150", 10.0)],
        db_realized_pnl=0.0,
        venue=ClearinghouseState(
            positions=(_vp("#150", 9.0, 0.9, 0.0),),
            account_value_usd=1.0,
        ),
        qty_tolerance=2e-2,
    )
    assert any(d.kind == "qty_mismatch" and d.symbol == "#150" for d in r.drift)


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
    # New format: fills shown as window/total — with no windowed data, shows ?/N
    assert "?/8" in msg and "?/527" in msg
    # Total shown in new dual format
    assert "total +140.29" in msg   # 2.17 + 138.12
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
    assert "v1_pm" in msg
    # HL slot keeps its headline AND gains a binary/bucket split.
    assert "v31" in msg
    assert "total +138.12" in msg          # 120.0 + 18.12 total_true_pnl shown as total
    assert "binary" in msg and "bucket" in msg
    assert "+100.00" in msg                    # binary total_pnl
    assert "+38.12" in msg                     # bucket total_pnl 20.0 + 18.12
    assert "500" in msg and "27" in msg
    # Desk total unchanged: 2.17 + 138.12
    assert "total +140.29" in msg


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


# ---------------------------------------------------------------------------
# New tests for trailing-24h windowed PnL fields
# ---------------------------------------------------------------------------

def test_slot_recon_window_total_pnl_uses_venue_window():
    """window_total_pnl = venue_realized_pnl_window + open_mtm when venue window is set."""
    r = SlotRecon(
        alias="v1", realized_pnl=500.0, open_mtm=12.5,
        account_value_usd=1000.0, positions_known=True,
        venue_realized_pnl=500.0,
        venue_realized_pnl_window=80.0,   # only 80 realized in last 24h
        realized_pnl_window=79.0,         # local (not used when venue window set)
        fills_count=50, fills_count_window=5,
    )
    # window_total_pnl = venue_realized_pnl_window (80) + open_mtm (12.5)
    assert r.window_total_pnl == 80.0 + 12.5
    # total_true_pnl still uses all-time venue realized
    assert r.total_true_pnl == 500.0 + 12.5


def test_slot_recon_window_total_pnl_falls_back_to_local_for_pm():
    """For PM (venue_realized_pnl_window=None), window_total_pnl uses local realized_pnl_window."""
    r = SlotRecon(
        alias="v31_pm", realized_pnl=30.0, open_mtm=3.0,
        account_value_usd=200.0, positions_known=True,
        venue_realized_pnl=None,          # PM: no venue realized
        venue_realized_pnl_window=None,   # PM: no venue window
        realized_pnl_window=15.0,         # local 24h realized
        fills_count=20, fills_count_window=4,
    )
    # window_total_pnl falls back to local realized_pnl_window (15) + open_mtm (3)
    assert r.window_total_pnl == 15.0 + 3.0
    # total_true_pnl uses local all-time realized (no venue)
    assert r.total_true_pnl == 30.0 + 3.0


def test_format_daily_summary_shows_both_24h_and_total():
    """format_daily_summary renders 24h and total PnL on each slot line and footer."""
    from hlanalysis.engine.reconcile_report import format_daily_summary
    recon = [
        SlotRecon(
            alias="v1", realized_pnl=500.0, open_mtm=10.0,
            account_value_usd=900.0, positions_known=True,
            venue_realized_pnl=500.0,
            venue_realized_pnl_window=80.0,
            fills_count=50, fills_count_window=5,
        ),
        SlotRecon(
            alias="v31_pm", realized_pnl=30.0, open_mtm=3.0,
            account_value_usd=200.0, positions_known=True,
            venue_realized_pnl=None,
            realized_pnl_window=15.0,
            fills_count=20, fills_count_window=4,
        ),
    ]
    msg = format_daily_summary(recon, date_str="2026-06-08")
    # Per-slot: both 24h and total visible
    # v1: window_total_pnl = 80 + 10 = 90; total_true_pnl = 500 + 10 = 510
    assert "v1: 24h +90.00 | total +510.00" in msg
    assert "fills 5/50" in msg
    # v31_pm: window_total_pnl = 15 + 3 = 18; total_true_pnl = 30 + 3 = 33
    assert "v31_pm: 24h +18.00 | total +33.00" in msg
    assert "fills 4/20" in msg
    # Footer: 24h total = 90 + 18 = 108; all-time total = 510 + 33 = 543
    assert "Total strategy PnL: 24h +108.00 | total +543.00" in msg
    assert "all reconciled" in msg


def test_format_daily_summary_fills_question_mark_when_window_unknown():
    """When fills_count_window is None the fills field shows ?/N."""
    from hlanalysis.engine.reconcile_report import format_daily_summary
    recon = [
        SlotRecon(
            alias="v31", realized_pnl=0.0, open_mtm=0.0,
            account_value_usd=500.0, positions_known=True,
            venue_realized_pnl=100.0,
            fills_count=200,
            fills_count_window=None,   # window fill count not available
        ),
    ]
    msg = format_daily_summary(recon)
    assert "fills ?/200" in msg


def test_format_daily_summary_desk_footer_dual_totals():
    """Desk footer shows BOTH 24h and total; recon status is unchanged."""
    from hlanalysis.engine.reconcile_report import format_daily_summary
    recon = [
        SlotRecon(alias="v1", realized_pnl=100.0, open_mtm=5.0,
                  account_value_usd=500.0, positions_known=True,
                  venue_realized_pnl=100.0,
                  venue_realized_pnl_window=20.0,
                  fills_count=10, fills_count_window=2),
        SlotRecon(alias="v31", realized_pnl=0.0, open_mtm=2.0,
                  account_value_usd=800.0, positions_known=True,
                  venue_realized_pnl=50.0,
                  venue_realized_pnl_window=10.0,
                  fills_count=30, fills_count_window=3),
    ]
    msg = format_daily_summary(recon)
    # v1 window: 20+5=25; v31 window: 10+2=12; total window = 37
    # v1 all-time: 100+5=105; v31 all-time: 50+2=52; total all-time = 157
    assert "Total strategy PnL: 24h +37.00 | total +157.00" in msg
    assert "all reconciled ✅" in msg


def test_format_daily_summary_klass_split_window_and_total():
    """Per-class lines show both 24h and total PnL + fills when windowed breakdown is set."""
    from hlanalysis.engine.reconcile_report import KlassStat, format_daily_summary
    recon = [
        SlotRecon(
            alias="v31",
            realized_pnl=0.0, open_mtm=5.0,
            account_value_usd=1000.0, positions_known=True,
            venue_realized_pnl=120.0, fills_count=50,
            klass_breakdown={
                "priceBinary": KlassStat(realized_pnl=80.0, open_mtm=0.0, fills=40),
                "priceBucket": KlassStat(realized_pnl=40.0, open_mtm=5.0, fills=10),
            },
            venue_realized_pnl_window=30.0, fills_count_window=8,
            klass_breakdown_window={
                "priceBinary": KlassStat(realized_pnl=20.0, open_mtm=0.0, fills=5),
                "priceBucket": KlassStat(realized_pnl=10.0, open_mtm=5.0, fills=3),
            },
        ),
    ]
    msg = format_daily_summary(recon, date_str="2026-06-08")
    # Per-class lines must show both window and total PnL and fills.
    # binary: win total_pnl=20+0=20; tot total_pnl=80+0=80; wins fills=5 tot=40
    assert "binary: 24h +20.00 | total +80.00 | fills 5/40" in msg
    # bucket: win total_pnl=10+5=15; tot total_pnl=40+5=45; wins fills=3 tot=10
    assert "bucket: 24h +15.00 | total +45.00 | fills 3/10" in msg


def test_format_daily_summary_klass_zero_window_fills_shows_zero_not_question():
    """A class with all-time activity but ZERO fills in the trailing window must
    render '24h +0.00 ... fills 0/N', not 'n/a ... ?/N'. The windowed breakdown
    only contains classes that had window fills/open-MTM, so an absent class means
    a known zero — not unknown data. (This was the v31 'bucket: 24h n/a | fills
    ?/307' in the live report: 307 bucket fills all-time, 0 in the last 24h.)"""
    from hlanalysis.engine.reconcile_report import KlassStat, format_daily_summary
    recon = [
        SlotRecon(
            alias="v31",
            realized_pnl=0.0, open_mtm=0.0,
            account_value_usd=1344.0, positions_known=True,
            venue_realized_pnl=183.38, fills_count=550,
            klass_breakdown={
                "priceBinary": KlassStat(realized_pnl=458.30, open_mtm=0.0, fills=243),
                "priceBucket": KlassStat(realized_pnl=-274.93, open_mtm=0.0, fills=307),
            },
            venue_realized_pnl_window=45.26, fills_count_window=23,
            # window breakdown was COMPUTED (not None) but bucket had no fills
            # in the trailing window, so it is simply absent from the dict.
            klass_breakdown_window={
                "priceBinary": KlassStat(realized_pnl=45.26, open_mtm=0.0, fills=23),
            },
        ),
    ]
    msg = format_daily_summary(recon, date_str="2026-06-08")
    assert "binary: 24h +45.26 | total +458.30 | fills 23/243" in msg
    # The fix: zero window activity is a KNOWN zero, not "?".
    assert "bucket: 24h +0.00 | total -274.93 | fills 0/307" in msg
    assert "?/307" not in msg
    assert "bucket: 24h n/a" not in msg


def test_format_daily_summary_klass_window_unknown_still_question():
    """When the windowed breakdown could NOT be computed at all
    (klass_breakdown_window is None — e.g. the venue fill fetch failed), the
    per-class window column genuinely is unknown and must still render 'n/a'/'?'."""
    from hlanalysis.engine.reconcile_report import KlassStat, format_daily_summary
    recon = [
        SlotRecon(
            alias="v31",
            realized_pnl=0.0, open_mtm=0.0,
            account_value_usd=500.0, positions_known=True,
            venue_realized_pnl=100.0, fills_count=300,
            klass_breakdown={
                "priceBucket": KlassStat(realized_pnl=100.0, open_mtm=0.0, fills=300),
            },
            venue_realized_pnl_window=None, fills_count_window=None,
            klass_breakdown_window=None,   # window data unavailable
        ),
    ]
    msg = format_daily_summary(recon)
    assert "bucket: 24h n/a | total +100.00 | fills ?/300" in msg


def test_gather_slot_windowed_fills_filtered_by_ts_ns():
    """gather_slot filters fills to the trailing window using ts_ns; fills without
    ts_ns are excluded from the window but included in the all-time count."""
    from hlanalysis.engine.reconcile_report import gather_slot

    class P:
        def __init__(self, symbol, qty):
            self.symbol = symbol; self.qty = qty

    class FakeDAL:
        def realized_pnl_since(self, since_ts_ns):
            return 0.0
        def all_positions(self):
            return []
        def coin_klass_map(self):
            return {"#100": "priceBinary", "#101": "priceBinary"}

    NOW_NS = 1_000_000_000_000_000_000   # arbitrary "now"
    WINDOW_NS = 24 * 3600 * 1_000_000_000

    class FakeFill:
        def __init__(self, symbol, closed_pnl, fee, ts_ns):
            self.symbol = symbol
            self.closed_pnl = closed_pnl
            self.fee = fee
            self.ts_ns = ts_ns

    class FakeClient:
        def clearinghouse_state(self):
            return ClearinghouseState(positions=(), account_value_usd=50.0)
        def user_fills(self, *, since_ts_ns):
            return [
                # In-window fill (ts_ns >= NOW_NS - WINDOW_NS)
                FakeFill("#100", 10.0, 0.5, NOW_NS - WINDOW_NS + 1),
                # Out-of-window fill (ts_ns < NOW_NS - WINDOW_NS)
                FakeFill("#101", 5.0, 0.0, NOW_NS - WINDOW_NS - 1),
            ]

    r = gather_slot(
        alias="v31", dal=FakeDAL(), exec_client=FakeClient(),
        qty_tolerance=1e-6, fetch_venue_realized=True,
        now_ns=NOW_NS, window_hours=24.0,
    )
    # All-time: both fills included
    assert r.fills_count == 2
    assert r.venue_realized_pnl == pytest.approx((10.0 - 0.5) + 5.0)
    # Window: only the in-window fill
    assert r.fills_count_window == 1
    assert r.venue_realized_pnl_window == pytest.approx(10.0 - 0.5)


def test_gather_slot_window_pm_uses_dal_since():
    """For PM (fetch_venue_realized=False) the window realized comes from dal.realized_pnl_since."""
    from hlanalysis.engine.reconcile_report import gather_slot

    NOW_NS = 2_000_000_000_000_000_000
    WINDOW_START = NOW_NS - 24 * 3600 * 1_000_000_000

    class FakeDAL:
        def realized_pnl_since(self, since_ts_ns):
            # Return different values for all-time vs window queries
            if since_ts_ns == 0:
                return 100.0
            elif since_ts_ns == WINDOW_START:
                return 25.0
            return 0.0

        def all_positions(self):
            return []

        def fills_count(self):
            return 50

    class FakeClient:
        def clearinghouse_state(self):
            return ClearinghouseState(positions=(), account_value_usd=200.0,
                                      positions_known=False)

    r = gather_slot(
        alias="v31_pm", dal=FakeDAL(), exec_client=FakeClient(),
        qty_tolerance=1e-6, fetch_venue_realized=False,
        now_ns=NOW_NS, window_hours=24.0,
    )
    assert r.realized_pnl == 100.0           # all-time local
    assert r.realized_pnl_window == 25.0     # trailing-24h local
    assert r.venue_realized_pnl is None      # no venue for PM
    assert r.venue_realized_pnl_window is None
    # window_total_pnl = 25 (local window) + 0 (open_mtm) = 25
    assert r.window_total_pnl == 25.0
    # total_true_pnl = 100 (all-time local) + 0 = 100
    assert r.total_true_pnl == 100.0


def test_compare_slot_windowed_fields_passthrough():
    """compare_slot passes through windowed fields unchanged into SlotRecon."""
    from hlanalysis.engine.reconcile_report import KlassStat

    win_klass = {"priceBinary": KlassStat(10.0, 0.0, 3)}
    r = compare_slot(
        alias="v31",
        db_positions=[],
        db_realized_pnl=100.0,
        venue=ClearinghouseState(positions=(), account_value_usd=50.0),
        qty_tolerance=1e-6,
        venue_realized_pnl=100.0,
        venue_realized_pnl_window=15.0,
        realized_pnl_window=14.0,
        fills_count_window=3,
        klass_breakdown_window=win_klass,
    )
    assert r.venue_realized_pnl_window == 15.0
    assert r.realized_pnl_window == 14.0
    assert r.fills_count_window == 3
    assert r.klass_breakdown_window is win_klass
    assert r.window_total_pnl == 15.0   # venue window (15) + open_mtm (0)


# Import pytest for approx assertions used above
import pytest
