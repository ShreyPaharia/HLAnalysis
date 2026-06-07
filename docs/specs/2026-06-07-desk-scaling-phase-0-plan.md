# Phase 0 — Trust PnL & Risk Visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-slot true PnL and open risk *trustworthy* by adding a venue-reconciliation report that runs on the box, and lock in the settlement-inclusive PnL accounting so it can't regress.

**Architecture:** Reuse the existing read-only venue surface (`exec_client.clearinghouse_state()` returns `ClearinghouseState(positions, account_value_usd, positions_known)`), the existing `StateDAL` PnL helpers, and the `engine-diag` SSM pattern. New code is one module `hlanalysis/engine/reconcile_report.py` split into a **pure core** (compare + format, fully unit-tested) and a **thin IO shell** (build read-only clients, gather, alert). A small DRY refactor extracts per-slot client construction so the report and the runtime share one builder.

**Tech Stack:** Python 3.12, sqlmodel/sqlite (`StateDAL`), pytest, loguru, existing `TelegramClient`, AWS SSM Make targets.

Spec: `docs/specs/2026-06-07-desk-scaling-phase-0-1-design.md` (Phase 0 §4).

---

## File structure

| File | Responsibility |
|------|----------------|
| `hlanalysis/engine/config_builders.py` (modify) | Add `build_exec_client(alias, acct, paper_mode)` — single source of per-slot client construction, lifted from `runtime._build_slot`. |
| `hlanalysis/engine/runtime.py` (modify) | Call `build_exec_client` instead of inline HLClient/PMClient construction (bit-identical). |
| `hlanalysis/engine/reconcile_report.py` (create) | Pure: `SlotRecon`, `compare_slot`, `format_report`. IO: `gather_slot`, `build_report`, `main` CLI. |
| `tests/unit/test_reconcile_report.py` (create) | Unit tests for the pure core (compare + format) and a drift-alert test with a fake client. |
| `tests/unit/test_pnl_accounting_regression.py` (create) | Pin settlement-inclusive PnL (SHR-49/53). |
| `Makefile` (modify) | Add `reconcile-report` target (SSM, env-sourced, like `engine-diag`). |
| `DEPLOYMENT.md` (modify) | Runbook entry for the report + Phase-0 gate checklist. |

---

## Task 1: Pin settlement-inclusive PnL accounting (W0.2 regression guard)

**Files:**
- Test: `tests/unit/test_pnl_accounting_regression.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pnl_accounting_regression.py
"""Regression guard: realized_pnl_since MUST include settlement payouts.
HIP-4 binaries close via settlement, not HL fills (SHR-49/53). If this
breaks, the daily-loss gate and the reconciliation report both go blind to
the dominant PnL component of the binary strategy."""
from pathlib import Path

from hlanalysis.engine.state import StateDAL


def test_realized_pnl_since_includes_settlement(tmp_path: Path) -> None:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    # A settlement payout with NO corresponding fill row.
    dal.record_settlement(
        question_idx=7, symbol="BTC", realized_pnl=120.0, ts_ns=1_000,
    )

    # realized_pnl_since(0) must surface the settlement even with zero fills.
    assert dal.realized_pnl_since(0) == 120.0
    # And it must respect the since window.
    assert dal.realized_pnl_since(2_000) == 0.0
    # settlement_pnl_since is the isolated component.
    assert dal.settlement_pnl_since(0) == 120.0
```

- [ ] **Step 2: Run test to verify it passes (this is a guard on existing behavior)**

Run: `uv run pytest tests/unit/test_pnl_accounting_regression.py -v`
Expected: PASS (the code in `state.py:realized_pnl_since` already sums `settlement_pnl_since`). If it FAILS, settlement accounting has regressed — stop and fix `state.py` before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_pnl_accounting_regression.py
git commit -m "test(engine): pin settlement-inclusive realized PnL (SHR-49/53 guard)"
```

---

## Task 2: Extract `build_exec_client` (DRY, enables the report to reuse client construction)

**Files:**
- Modify: `hlanalysis/engine/config_builders.py`
- Modify: `hlanalysis/engine/runtime.py:422-440` (the inline client construction)
- Test: `tests/unit/test_build_exec_client.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_build_exec_client.py
from hlanalysis.engine.config_builders import build_exec_client
from hlanalysis.engine.config import HyperliquidAccount, PolymarketAccount
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.pm_client import PMClient


def test_build_exec_client_hl():
    acct = HyperliquidAccount(
        account_address="0xabc", api_secret_key="0xkey",
        base_url="https://api.hyperliquid.xyz",
    )
    client = build_exec_client("v1", acct, paper_mode=True)
    assert isinstance(client, HLClient)


def test_build_exec_client_pm():
    acct = PolymarketAccount(
        clob_host="https://clob.polymarket.com", chain_id=137,
        private_key="0xpk", clob_api_key="k", clob_api_secret="s",
        clob_api_passphrase="p", funder_address="0xfund",
    )
    client = build_exec_client("v1_pm", acct, paper_mode=True)
    assert isinstance(client, PMClient)
```

> NOTE: confirm the exact `HyperliquidAccount` / `PolymarketAccount` field names in `hlanalysis/engine/config.py` before running; adjust the kwargs above to match. Do not invent fields.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_build_exec_client.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_exec_client'`.

- [ ] **Step 3: Add the builder** (in `config_builders.py`, lift the construction from `runtime._build_slot`)

```python
# hlanalysis/engine/config_builders.py  (append; add imports at top of file)
from .config import AccountConfig, HyperliquidAccount, PolymarketAccount
from .exec_client import ExecutionClient
from .hl_client import HLClient


def build_exec_client(
    alias: str, acct: AccountConfig, paper_mode: bool,
) -> ExecutionClient:
    """Single source of per-slot ExecutionClient construction.

    Shared by EngineRuntime._build_slot (live wiring) and reconcile_report
    (read-only out-of-band check), so both build the identical client for an
    alias from the same AccountConfig.
    """
    if isinstance(acct, HyperliquidAccount):
        return HLClient(
            account_address=acct.account_address,
            api_secret_key=acct.api_secret_key,
            base_url=acct.base_url,
            paper_mode=paper_mode,
        )
    if isinstance(acct, PolymarketAccount):
        from .pm_client import PMClient
        return PMClient(
            paper_mode=paper_mode,
            clob_host=acct.clob_host,
            chain_id=acct.chain_id,
            private_key=acct.private_key,
            clob_api_key=acct.clob_api_key,
            clob_api_secret=acct.clob_api_secret,
            clob_api_passphrase=acct.clob_api_passphrase,
            funder_address=acct.funder_address,
        )
    raise TypeError(f"unknown account type for alias {alias!r}: {type(acct).__name__}")
```

> Match the PMClient kwargs to the real `runtime._build_slot` call (lines ~433-440) exactly — copy them verbatim including any beyond `funder_address`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_build_exec_client.py -v`
Expected: PASS.

- [ ] **Step 5: Replace the inline construction in `runtime._build_slot`**

In `hlanalysis/engine/runtime.py`, replace the `elif isinstance(acct, HyperliquidAccount): ... elif isinstance(acct, PolymarketAccount): ...` block (the inline construction at ~424-440) with:

```python
        if self.exec_client_factory is not None:
            exec_client = self.exec_client_factory(alias, acct, s_cfg.paper_mode)
        else:
            from .config_builders import build_exec_client
            exec_client = build_exec_client(alias, acct, s_cfg.paper_mode)
```

- [ ] **Step 6: Run the engine runtime tests to verify bit-identical wiring**

Run: `uv run pytest tests/ -k "runtime or build_slot or exec_client" -q`
Expected: PASS (no behavior change).

- [ ] **Step 7: Commit**

```bash
git add hlanalysis/engine/config_builders.py hlanalysis/engine/runtime.py tests/unit/test_build_exec_client.py
git commit -m "refactor(engine): extract build_exec_client shared by runtime + recon report"
```

---

## Task 3: Pure reconciliation core — `SlotRecon` + `compare_slot`

**Files:**
- Create: `hlanalysis/engine/reconcile_report.py`
- Test: `tests/unit/test_reconcile_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_reconcile_report.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_reconcile_report.py -v`
Expected: FAIL with `ModuleNotFoundError: hlanalysis.engine.reconcile_report`.

- [ ] **Step 3: Write the pure core**

```python
# hlanalysis/engine/reconcile_report.py
"""Out-of-band venue reconciliation report (Phase 0, W0.3/W0.4).

Run on the box via SSM (env-sourced for credentials), same pattern as
engine-diag. Per slot: compare engine-DB realized PnL + open positions against
the venue's clearinghouse_state(), report realized + open-MTM = total true PnL,
flag position drift, and alert on Telegram when drift exceeds tolerance.

Split into a PURE core (SlotRecon / compare_slot / format_report — no IO, fully
unit-tested) and a thin IO shell (gather_slot / build_report / main).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exec_types import ClearinghouseState


@dataclass(frozen=True, slots=True)
class Drift:
    kind: str       # "qty_mismatch" | "vanished" | "orphan"
    symbol: str
    db_qty: float
    venue_qty: float


@dataclass(frozen=True, slots=True)
class SlotRecon:
    alias: str
    realized_pnl: float
    open_mtm: float
    account_value_usd: float
    positions_known: bool
    drift: list[Drift] = field(default_factory=list)

    @property
    def total_true_pnl(self) -> float:
        return self.realized_pnl + self.open_mtm

    @property
    def has_drift(self) -> bool:
        return len(self.drift) > 0


def compare_slot(
    *,
    alias: str,
    db_positions: list[tuple[str, float]],   # (symbol, qty)
    db_realized_pnl: float,
    venue: ClearinghouseState,
    qty_tolerance: float,
) -> SlotRecon:
    """Pure three-way-ish compare: DB positions vs venue positions, plus PnL.

    Open MTM = Σ venue unrealized_pnl. Position drift is skipped entirely when
    venue.positions_known is False (PM data-api flap) so an empty venue set is
    never mistaken for 'everything vanished'.
    """
    open_mtm = sum(vp.unrealized_pnl for vp in venue.positions)
    drift: list[Drift] = []

    if venue.positions_known:
        venue_by_sym = {vp.symbol: vp.qty for vp in venue.positions}
        db_by_sym = {sym: qty for sym, qty in db_positions}

        for sym, db_qty in db_by_sym.items():
            v_qty = venue_by_sym.get(sym)
            if v_qty is None:
                drift.append(Drift("vanished", sym, db_qty, 0.0))
            elif abs(v_qty - db_qty) > qty_tolerance:
                drift.append(Drift("qty_mismatch", sym, db_qty, v_qty))
        for sym, v_qty in venue_by_sym.items():
            if sym not in db_by_sym:
                drift.append(Drift("orphan", sym, 0.0, v_qty))

    return SlotRecon(
        alias=alias,
        realized_pnl=db_realized_pnl,
        open_mtm=open_mtm,
        account_value_usd=venue.account_value_usd,
        positions_known=venue.positions_known,
        drift=drift,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_reconcile_report.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/reconcile_report.py tests/unit/test_reconcile_report.py
git commit -m "feat(engine): pure reconciliation core (DB vs venue positions + PnL)"
```

---

## Task 4: Report formatter (`format_report`)

**Files:**
- Modify: `hlanalysis/engine/reconcile_report.py`
- Test: `tests/unit/test_reconcile_report.py` (add)

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
from hlanalysis.engine.reconcile_report import format_report


def test_format_report_clean():
    recon = [
        SlotRecon(alias="v1", realized_pnl=120.0, open_mtm=5.0,
                  account_value_usd=1000.0, positions_known=True, drift=[]),
    ]
    text = format_report(recon)
    assert "v1" in text
    assert "120" in text          # realized
    assert "125" in text          # total true pnl
    assert "OK" in text or "✅" in text


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_reconcile_report.py -k format -v`
Expected: FAIL with `ImportError: cannot import name 'format_report'`.

- [ ] **Step 3: Implement `format_report`** (append to `reconcile_report.py`)

```python
def format_report(recon: list[SlotRecon]) -> str:
    """Plain-text/HTML-safe report. One block per slot, drift called out.

    Kept dependency-free so it can be unit-tested and reused by the Telegram
    alert (TelegramClient.send escapes <, >, & itself).
    """
    lines: list[str] = ["Engine reconciliation report", ""]
    for r in recon:
        status = "DRIFT" if r.has_drift else "OK"
        lines.append(f"[{r.alias}] {status}")
        lines.append(
            f"  realized={r.realized_pnl:+.2f}  open_mtm={r.open_mtm:+.2f}  "
            f"true_pnl={r.total_true_pnl:+.2f}  acct_value={r.account_value_usd:.2f}"
        )
        if not r.positions_known:
            lines.append("  (positions unknown — recon skipped this cycle)")
        for d in r.drift:
            lines.append(
                f"  ! {d.kind} {d.symbol}: db_qty={d.db_qty} venue_qty={d.venue_qty}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_reconcile_report.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/reconcile_report.py tests/unit/test_reconcile_report.py
git commit -m "feat(engine): reconciliation report formatter"
```

---

## Task 5: IO shell — `gather_slot`, `build_report`, and the `main` CLI

**Files:**
- Modify: `hlanalysis/engine/reconcile_report.py`
- Test: `tests/unit/test_reconcile_report.py` (add a gather test with a fake client + fake DAL)

- [ ] **Step 1: Write the failing test** (append)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_reconcile_report.py -k gather -v`
Expected: FAIL with `ImportError: cannot import name 'gather_slot'`.

- [ ] **Step 3: Implement the IO shell** (append to `reconcile_report.py`)

```python
def gather_slot(*, alias, dal, exec_client, qty_tolerance: float) -> SlotRecon:
    """IO: pull realized PnL (incl settlement) + DB positions + venue state for
    one slot and run the pure compare. clearinghouse_state() is a blocking SDK
    call; the caller offloads it via asyncio.to_thread when needed."""
    db_realized = dal.realized_pnl_since(0)
    db_positions = [(p.symbol, p.qty) for p in dal.all_positions()]
    venue = exec_client.clearinghouse_state()
    return compare_slot(
        alias=alias, db_positions=db_positions, db_realized_pnl=db_realized,
        venue=venue, qty_tolerance=qty_tolerance,
    )


def build_report(deploy_cfg, strategies_cfg, *, qty_tolerance: float) -> list[SlotRecon]:
    """IO: build a read-only client + open the DAL per slot, gather each.

    A slot whose client/DAL errors yields a positions_known=False SlotRecon so
    one bad slot never aborts the whole report."""
    from .config_builders import build_exec_client
    from .state import StateDAL

    out: list[SlotRecon] = []
    for s_cfg in strategies_cfg.strategies:
        alias = s_cfg.account_alias
        try:
            acct = deploy_cfg.accounts[alias]
            client = build_exec_client(alias, acct, paper_mode=True)  # read-only path
            dal = StateDAL(Path(deploy_cfg.state_db_path_for(alias)))
            out.append(gather_slot(alias=alias, dal=dal, exec_client=client,
                                   qty_tolerance=qty_tolerance))
        except Exception as e:  # noqa: BLE001 — a bad slot must not abort the report
            out.append(SlotRecon(alias=alias, realized_pnl=0.0, open_mtm=0.0,
                                 account_value_usd=0.0, positions_known=False,
                                 drift=[]))
            from loguru import logger
            logger.warning("recon slot {} failed: {}", alias, e)
    return out


async def _maybe_alert(report_text: str, has_drift: bool) -> None:
    """Send the report to Telegram when there is drift. Credentials come from
    the same env the engine uses; no-op if unset."""
    import os
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (has_drift and token and chat):
        return
    import aiohttp
    from hlanalysis.alerts.telegram import TelegramClient
    async with aiohttp.ClientSession() as session:
        tg = TelegramClient(bot_token=token, chat_id=chat, session=session)
        await tg.send("⚠️ Reconciliation DRIFT\n\n" + report_text)


def main() -> None:
    p = argparse.ArgumentParser(description="Out-of-band venue reconciliation report.")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument("--qty-tolerance", type=float, default=1e-6)
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args()

    from .config import load_deploy_config, load_strategies_config
    deploy_cfg = load_deploy_config(args.deploy_config)
    strategies_cfg = load_strategies_config(args.strategy_config)

    recon = build_report(deploy_cfg, strategies_cfg, qty_tolerance=args.qty_tolerance)
    has_drift = any(r.has_drift for r in recon)
    text = format_report(recon)

    if args.json:
        print(json.dumps({
            "generated_at_ns": time.time_ns(),
            "has_drift": has_drift,
            "slots": [
                {"alias": r.alias, "realized_pnl": r.realized_pnl,
                 "open_mtm": r.open_mtm, "total_true_pnl": r.total_true_pnl,
                 "account_value_usd": r.account_value_usd,
                 "positions_known": r.positions_known,
                 "drift": [vars(d) for d in r.drift]}
                for r in recon
            ],
        }))
    else:
        print(text)

    asyncio.run(_maybe_alert(text, has_drift))
    # Nonzero exit on drift so an SSM/cron caller can detect it.
    raise SystemExit(1 if has_drift else 0)


if __name__ == "__main__":
    main()
```

> Confirm `load_deploy_config` / `load_strategies_config` import paths against `diag.py:440` (it imports them from `hlanalysis.engine.config`). Match exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_reconcile_report.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/reconcile_report.py tests/unit/test_reconcile_report.py
git commit -m "feat(engine): reconcile-report IO shell + CLI with drift alert"
```

---

## Task 6: Drift-alert behavior test (W0.3 alert verification)

**Files:**
- Test: `tests/unit/test_reconcile_report.py` (add)

- [ ] **Step 1: Write the failing test** (append) — assert Telegram is called on drift, not on clean

```python
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
    monkeypatch.setattr(rr, "aiohttp", type("M", (), {"ClientSession": lambda: FakeSession()}), raising=False)
    monkeypatch.setattr("hlanalysis.alerts.telegram.TelegramClient", FakeTG)

    asyncio.run(rr._maybe_alert("report", has_drift=False))
    assert sent == []                       # no drift → no alert

    asyncio.run(rr._maybe_alert("report", has_drift=True))
    assert len(sent) == 1 and "DRIFT" in sent[0]
```

> If `import aiohttp` inside `_maybe_alert` resists monkeypatching, refactor `_maybe_alert` to accept an injected `session_factory` and `tg_factory` (default to the real ones) — the cleaner, more testable shape. Pick whichever keeps the test honest; do not weaken the test to pass.

- [ ] **Step 2: Run test to verify it fails, then make it pass**

Run: `uv run pytest tests/unit/test_reconcile_report.py -k alert -v`
Expected: FAIL first; then PASS after the inject-factory refactor if needed.

- [ ] **Step 3: Run the full module suite**

Run: `uv run pytest tests/unit/test_reconcile_report.py -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add hlanalysis/engine/reconcile_report.py tests/unit/test_reconcile_report.py
git commit -m "test(engine): reconcile-report alerts only on drift"
```

---

## Task 7: Ops wiring — Make target + runbook + Phase-0 gate checklist

**Files:**
- Modify: `Makefile`
- Modify: `DEPLOYMENT.md`

- [ ] **Step 1: Add the `reconcile-report` Make target** (model on the `engine-diag` target; runs over SSM, sources `/etc/hl-engine/env` for credentials)

```makefile
# One-shot venue reconciliation report: per-slot realized + open-MTM = true PnL,
# plus position drift (qty mismatch / vanished / orphan) vs the venue. Sends a
# Telegram alert when drift is found. Pass JSON=1 for machine-readable output.
reconcile-report:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; exit 1; \
	fi && \
	JSON_ARG="" && [ -n "$$JSON" ] && JSON_ARG="--json" || true && \
	echo "Fetching reconciliation report from $$INSTANCE_ID..." && \
	CMD_ID=$$(aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters "commands=[\"cd /opt/hl-recorder && source /etc/hl-engine/env && uv run python -m hlanalysis.engine.reconcile_report $$JSON_ARG\"]" \
		--query "Command.CommandId" --output text) && \
	sleep 6 && \
	aws ssm get-command-invocation --command-id "$$CMD_ID" --instance-id "$$INSTANCE_ID" \
		--query "StandardOutputContent" --output text
```

Also add `reconcile-report` to the `.PHONY` line and the `help` target (mirror the `engine-diag` help line).

- [ ] **Step 2: Add the runbook + Phase-0 gate to `DEPLOYMENT.md`**

Add a section documenting: `make reconcile-report` (and `JSON=1`), what drift kinds mean, that it exits nonzero on drift, and the **Phase 0 exit gate checklist**:

```markdown
### Reconciliation report (Phase 0)

`make reconcile-report` — per-slot realized + open-MTM true PnL reconciled to the
venue, with position-drift detection. `JSON=1` for machine-readable output.
Exits nonzero on drift; sends a Telegram alert when `TELEGRAM_*` env is set.

**Phase 0 exit gate — all must hold for 7 consecutive days, all 4 slots:**
- [ ] `engine-diag` + `reconcile-report` accessible via one command (documented above).
- [ ] Per-slot true PnL (realized + settlement) reported by `engine-diag`; total
      true PnL (+ open MTM) reported by `reconcile-report`.
- [ ] `reconcile-report` shows no unexplained drift beyond tolerance.
- [ ] An injected drift raises a Telegram alert (verified once).
```

- [ ] **Step 3: Verify the Make target parses**

Run: `make -n reconcile-report`
Expected: prints the command body with no Make syntax error (does not need AWS to resolve for a dry parse; if `STACK_NAME` is required, expect it to expand).

- [ ] **Step 4: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: PASS (existing green count + the new tests).

- [ ] **Step 5: Commit**

```bash
git add Makefile DEPLOYMENT.md
git commit -m "ops(engine): add reconcile-report SSM target + Phase-0 runbook gate"
```

---

## Task 8: Deploy & operationalize observability (W0.1) — ops, not code

**Files:** none (deploy + verification). This task is the "deploy tier-1" workstream.

- [ ] **Step 1: Confirm the observability code is on `main`**

Run: `git log --oneline main -- hlanalysis/engine/diag.py scripts/engine_events.py | head`
Expected: the diag + engine_events commits are present on `main`. If `reconcile_report.py` (this plan) is on a feature branch, it merges to `main` before deploy.

- [ ] **Step 2: Deploy to EC2 via SSM** (per CLAUDE.md — deploy is SSM-only)

Run: `make deploy-engine` (and `make engine-status` after)
Expected: engine restarts cleanly; `engine-status` shows no `restart_blocked` / `halt` flags.

- [ ] **Step 3: Smoke the three tools against the live box**

Run: `make engine-diag PRETTY=1` ; `make reconcile-report` ; `make engine-events Q=<a-recent-qidx>`
Expected: each returns a populated snapshot/report; `reconcile-report` shows per-slot true PnL and (initially) no drift.

- [ ] **Step 4: Start the 7-day Phase-0 reconciliation soak**

Record the start date in the Linear epic (SHR-30) and the gate checklist in DEPLOYMENT.md. Re-run `make reconcile-report` daily (or wire a systemd timer — see "Optional" below); the gate is met when 7 consecutive days show no unexplained drift across all 4 slots.

- [ ] **Step 5: (Optional) systemd timer on the box** — only if daily manual runs are a burden. A `hl-reconcile.timer` running `python -m hlanalysis.engine.reconcile_report` daily, memory-capped (the box is a 1 GB t4g.micro — keep it a one-shot, not a resident loop). Document in DEPLOYMENT.md; do not add a resident process.

---

## Self-review notes

- **Spec coverage:** W0.1→Task 8; W0.2→Tasks 1+(diag relabel in Task 3/4 output); W0.3→Tasks 3–7; W0.4→Task 4 (per-slot realized/settlement/MTM breakdown in `format_report` / JSON). Gate→Task 7 Step 2.
- **Open calibration:** `qty_tolerance` (and a future PnL tolerance) start strict (1e-6 on qty) and are tuned after observing the first days of real drift — per spec §9. Position drift is qty-based; a PnL tolerance band can be added to `compare_slot` later without interface change.
- **Verify-before-coding flags:** account field names (Task 2), PMClient kwargs (Task 2), config loader import path (Task 5) — each step says to match the real source, not invent.
