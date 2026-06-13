# tests/unit/test_engine_runtime_cadence.py
"""v3.7 cadence port: the live engine must couple the shared MarketState's
per-symbol mark-bucket period to each slot's vol_sampling_dt_seconds, with no
train/serve skew, the 60s default preserved, and HL/PM independent.
"""

from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    HyperliquidAccount,
    PolymarketAccount,
    StrategyConfig,
    ThetaParams,
)
from hlanalysis.engine.config import AlertsConfig, TelegramConfig
from hlanalysis.engine.runtime import (
    EngineRuntime,
    reference_sampling_dt_seconds,
    reference_vol_lookback_seconds,
)

_NS = 1_000_000_000


class _FakeExec:
    """Minimal ExecutionClient stand-in — _build_slot only needs an object with
    realized_pnl_since to wire the Scanner's pnl_provider; it is never called
    during cadence registration."""

    def realized_pnl_since(self, _ns: int) -> float:  # pragma: no cover - unused
        return 0.0


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=500,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200,
        max_strike_distance_pct=50,
        min_recent_volume_usd=100,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )


def _theta(dt: int) -> ThetaParams:
    return ThetaParams(vol_lookback_seconds=3600, vol_sampling_dt_seconds=dt)


def _theta_cfg(*, alias: str, reference_symbol: str, dt: int) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=43200,
        price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    return StrategyConfig(
        name="theta_harvester",
        account_alias=alias,
        paper_mode=True,
        strategy_type="theta_harvester",
        reference_symbol=reference_symbol,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        theta=_theta(dt),
        **{"global": _global()},
    )


def _late_cfg(
    *,
    alias: str,
    reference_symbol: str,
    dt: int = 60,
    reference_sigma_source: str = "mark",
) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.85,
        distance_from_strike_usd_min=0,
        vol_max=100,
        vol_lookback_seconds=3600,
        vol_sampling_dt_seconds=dt,
    )
    return StrategyConfig(
        name="late_resolution",
        account_alias=alias,
        paper_mode=True,
        strategy_type="late_resolution",
        reference_symbol=reference_symbol,
        reference_sigma_source=reference_sigma_source,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{"global": _global()},
    )


def _runtime(strategies, tmp_path) -> EngineRuntime:
    accounts = {}
    for s in strategies:
        if s.reference_symbol == "BTCUSDT":
            accounts[s.account_alias] = PolymarketAccount(
                private_key="0x0",
                clob_api_key="k",
                clob_api_secret="s",
                clob_api_passphrase="p",
            )
        else:
            accounts[s.account_alias] = HyperliquidAccount(
                account_address="0x0",
                api_secret_key="0x0",
                base_url="https://api.hyperliquid.xyz",
            )
    deploy = DeployConfig(
        env="dev",
        accounts=accounts,
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    rt = EngineRuntime(
        strategies=strategies,
        deploy_cfg=deploy,
        adapter_factory=lambda: None,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, _p: _FakeExec(),
    )
    rt.slots = [rt._build_slot(s) for s in strategies]
    return rt


# ---- helpers ---------------------------------------------------------------


def test_reference_sampling_dt_defaults_to_60_for_late_resolution():
    cfg = _late_cfg(alias="v1", reference_symbol="BTC")
    assert reference_sampling_dt_seconds(cfg) == 60


def test_reference_sampling_dt_reads_theta_block():
    assert reference_sampling_dt_seconds(_theta_cfg(alias="v31", reference_symbol="BTC", dt=5)) == 5
    assert reference_vol_lookback_seconds(_theta_cfg(alias="v31", reference_symbol="BTC", dt=5)) == 3600


def test_reference_sampling_dt_reads_late_resolution_slot():
    """v1 (late_resolution) cadence comes from its allowlist/defaults
    vol_sampling_dt_seconds, mirroring how theta reads the theta block."""
    assert reference_sampling_dt_seconds(_late_cfg(alias="v1", reference_symbol="BTC", dt=5)) == 5


def test_v1_late_and_v31_theta_lockstep_dt5_no_conflict(tmp_path):
    """v1 (late) + v31 (theta) both reading BTC at dt=5 is satisfiable — they
    agree, so registration succeeds and BTC buckets at 5s. This is the lockstep
    path validated in summeries/v1_cadence_validation_2026_05_30.md."""
    rt = _runtime(
        [
            _late_cfg(alias="v1", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC") == 5 * _NS


# ---- live wiring: no skew, default preserved, HL/PM independent ------------


def test_registration_default_60s_preserves_behavior(tmp_path):
    """Production-shaped slots all at dt=60 → every reference symbol still
    buckets at 60s (existing deployments unchanged)."""
    rt = _runtime(
        [
            _late_cfg(alias="v1", reference_symbol="BTC"),
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=60),
            _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC") == 60 * _NS
    assert rt.market_state.mark_bucket_ns_for("BTCUSDT") == 60 * _NS


def test_no_train_serve_skew_period_equals_strategy_assumption(tmp_path):
    """Acceptance: the period MarketState buckets at == the period the
    strategy's σ formula assumes, for every slot's reference symbol."""
    strategies = [
        _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
        _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
    ]
    rt = _runtime(strategies, tmp_path)
    rt._register_reference_cadences(rt.slots)
    for cfg in strategies:
        assert rt.market_state.mark_bucket_ns_for(cfg.reference_symbol) == (reference_sampling_dt_seconds(cfg) * _NS)


def test_hl_pm_independent(tmp_path):
    """Flipping the HL slot to dt=5 does NOT affect the PM reference symbol."""
    rt = _runtime(
        [
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.mark_bucket_ns_for("BTC") == 5 * _NS
    assert rt.market_state.mark_bucket_ns_for("BTCUSDT") == 60 * _NS


def test_same_symbol_different_cadence_coexist(tmp_path):
    """Two slots reading the SAME reference symbol with different dt both
    register successfully — each cadence is bucketed independently from the
    shared feed (the old single-cadence conflict-guard was removed by the
    (symbol, dt) refactor)."""
    rt = _runtime(
        [
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31b", reference_symbol="BTC", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)  # no raise
    # Both cadences are actually registered on the shared symbol (not just
    # resolvable by explicit dt, which holds unconditionally). The first
    # registered cadence (dt=5) is the symbol's default for dt-less reads.
    assert rt.market_state._cadences_by_symbol["BTC"] == [5 * _NS, 60 * _NS]
    assert rt.market_state.mark_bucket_ns_for("BTC") == 5 * _NS


# ---- per-symbol σ source: mark | bbo (Part B) ------------------------------


def test_reference_source_defaults_to_mark(tmp_path):
    """No reference_sigma_source set → HL/PM symbols stay mark-sourced
    (legacy behaviour, bit-identical)."""
    rt = _runtime(
        [
            _late_cfg(alias="v1", reference_symbol="BTC"),
            _theta_cfg(alias="v31_pm", reference_symbol="BTCUSDT", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.reference_source_for("BTC") == "mark"
    assert rt.market_state.reference_source_for("BTCUSDT") == "mark"


def test_pm_slot_can_opt_into_bbo_source_hl_stays_mark(tmp_path):
    """A PM slot (BTCUSDT) opts σ into the dense BBO feed; the HL slot (BTC)
    is independent and stays mark-sourced."""
    rt = _runtime(
        [
            _late_cfg(alias="v1", reference_symbol="BTC"),
            _late_cfg(
                alias="v1_pm",
                reference_symbol="BTCUSDT",
                reference_sigma_source="bbo",
            ),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    assert rt.market_state.reference_source_for("BTC") == "mark"
    assert rt.market_state.reference_source_for("BTCUSDT") == "bbo"


def test_conflicting_reference_source_same_symbol_raises(tmp_path):
    """Two slots reading the SAME reference symbol with different σ sources is
    unsatisfiable (one shared OHLC history) — fail fast at startup."""
    rt = _runtime(
        [
            _late_cfg(
                alias="v1_pm",
                reference_symbol="BTCUSDT",
                reference_sigma_source="bbo",
            ),
            _late_cfg(
                alias="v31_pm",
                reference_symbol="BTCUSDT",
                reference_sigma_source="mark",
            ),
        ],
        tmp_path,
    )
    with pytest.raises(ValueError, match="conflicting reference source"):
        rt._register_reference_cadences(rt.slots)


def test_per_class_override_registers_extra_cadence(tmp_path) -> None:
    """A v31 theta slot with a priceBucket dt=2 override registers BOTH dt=5
    (default) and dt=2 on the shared MarketState for its reference symbol, so
    both bar series accumulate from the one BTC feed."""
    from hlanalysis.engine.config import ThetaParams

    cfg = _theta_cfg(alias="v31", reference_symbol="BTC", dt=5)
    cfg = cfg.model_copy(
        update={
            "theta_overrides": {"priceBucket": ThetaParams(vol_sampling_dt_seconds=2)},
        }
    )
    rt = _runtime([cfg], tmp_path)
    rt._register_reference_cadences(rt.slots)
    # Assert BOTH cadences are actually REGISTERED on the shared symbol. Do NOT
    # assert via mark_bucket_ns_for(sym, dt=2) — that returns dt*1e9 for any
    # explicit dt regardless of registration, so it would pass vacuously.
    assert rt.market_state._cadences_by_symbol["BTC"] == [5 * _NS, 2 * _NS]


# ---- R9: two sibling slots with independent default cadences on one symbol ---


def test_two_sibling_slots_independent_default_cadences(tmp_path) -> None:
    """R9 capability: two sibling slots sharing BTC with different default dts
    (e.g. v31 at dt=5 and a hypothetical v31_slow at dt=60) can coexist and
    each registers its own cadence on the shared symbol.

    The runtime registers BOTH cadences, and the first-registered is the
    symbol's default for dt-less reads. This test pins that both are present
    so each slot's scanner can address its own buffer by explicit dt rather
    than relying on cadences[0].
    """
    rt = _runtime(
        [
            _theta_cfg(alias="v31", reference_symbol="BTC", dt=5),
            _theta_cfg(alias="v31_slow", reference_symbol="BTC", dt=60),
        ],
        tmp_path,
    )
    rt._register_reference_cadences(rt.slots)
    # Both cadences registered.
    registered = rt.market_state._cadences_by_symbol["BTC"]
    assert 5 * _NS in registered, f"dt=5 cadence not found in {registered}"
    assert 60 * _NS in registered, f"dt=60 cadence not found in {registered}"
    # The slot registered first (v31, dt=5) is cadences[0] — the default for
    # dt-less reads. The second slot (dt=60) must read via explicit dt=60 to
    # avoid aliasing the dt=5 buffer.
    assert registered[0] == 5 * _NS, (
        f"v31 (dt=5) was registered first; expected cadences[0]=5s got {registered[0] // _NS}s"
    )
