"""reconcile_report must read the UNIFIED state DB and scope per strategy.

Bug (2026-06-15): post-cutover, reconcile_report opened the legacy per-slot DBs
via state_db_path_for(alias). Those are stale (engine writes the unified DB now)
AND still at schema 0005 (no fill.strategy_id), so the new ORM's select(Fill)
raised `OperationalError: no such column: fill.strategy_id` for every slot.

Fix: when a unified DB exists, build a StrategyScopedDAL over it (scoped by the
slot's strategy_id); else fall back to the per-slot path.
"""

from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.config import (
    AlertsConfig,
    DeployConfig,
    HyperliquidAccount,
    TelegramConfig,
)
from hlanalysis.engine.scoped_dal import StrategyScopedDAL
from hlanalysis.engine.state import CachedStateDAL, Fill


def _deploy(tmp_path: Path) -> DeployConfig:
    return DeployConfig(
        env="test",
        accounts={
            "v1": HyperliquidAccount(
                account_address="0xv1", api_secret_key="0xv1", base_url="https://api.hyperliquid.xyz"
            ),
            "v31": HyperliquidAccount(
                account_address="0xv31", api_secret_key="0xv31", base_url="https://api.hyperliquid.xyz"
            ),
        },
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="t", chat_id="c")),
        state_db_path=str(tmp_path / "engine" / "state.db"),
        kill_switch_path=str(tmp_path / "engine" / "halt"),
    )


def _fill(fid: str, qidx: int, sym: str, pnl: float) -> Fill:
    return Fill(
        fill_id=fid,
        cloid=fid,
        question_idx=qidx,
        symbol=sym,
        side="sell",
        price=1.0,
        size=1.0,
        fee=0.0,
        ts_ns=10,
        closed_pnl=pnl,
    )


def test_recon_dal_unified_scopes_by_strategy(tmp_path):
    cfg = _deploy(tmp_path)
    shared = cfg.state_db_path_shared()
    shared.parent.mkdir(parents=True, exist_ok=True)
    base = CachedStateDAL(shared)
    base.run_migrations()
    StrategyScopedDAL(base, strategy_id="v1", account="v1").append_fill(_fill("f1", 1, "#1", 5.0))
    StrategyScopedDAL(base, strategy_id="v31", account="v31").append_fill(_fill("f2", 2, "#2", -3.0))

    from hlanalysis.engine.reconcile_report import _build_recon_dal

    # Must NOT raise "no such column", and must scope per strategy.
    assert _build_recon_dal(cfg, "v1").realized_pnl_since(0) == 5.0
    assert _build_recon_dal(cfg, "v31").realized_pnl_since(0) == -3.0


def test_recon_dal_legacy_fallback_when_no_unified(tmp_path):
    cfg = _deploy(tmp_path)
    # No unified DB on disk → fall back to the per-slot path.
    from hlanalysis.engine.reconcile_report import _build_recon_dal

    dal = _build_recon_dal(cfg, "v1")
    assert Path(dal.db_path) == Path(cfg.state_db_path_for("v1"))
