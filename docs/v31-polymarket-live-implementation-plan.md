# v3.1 Polymarket Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take v3.1 (theta_harvester strategy with PM-ablation params) from backtest to live real-money on Polymarket BTC Up/Down daily markets, as a third (strategy, account) slot alongside the existing HL `v1` and `v31` slots.

**Architecture:** Three-stage rollout. **(A)** Behaviour-preserving refactor: extract `ExecutionClient` protocol from `HLClient`, swap `DeployConfig.hl_accounts: dict[str, HLConfig]` for a discriminated-union `accounts: dict[str, AccountConfig]`, hard-cutover the deploy.yaml. **(B)** Read-side: new `PolymarketAdapter(VenueAdapter)` wraps PM CLOB WS + Gamma REST, normalizes to `NormalizedEvent`; wired into the recorder first so PM data starts streaming to parquet before any order code lands. **(C)** Write-side: `PMClient` implementing `ExecutionClient` (paper-mode first, then `py-clob-client-v2`-backed live mode); depth-walk slippage gate in `RiskGate`; settlement poller maps Gamma `resolved=true` → `SettlementEvent` → existing `Router._close_settled` flow.

**Tech Stack:** Python 3.12, `uv`, `pytest`, `loguru`, `aiohttp`, `websockets`, `pydantic`, `py-clob-client-v2` (NEW), `httpx`. Test discipline: **TDD with fakes** for pure logic + **one recorded raw-PM-WS fixture** for adapter integration test.

**Source of truth for params:** `config/tuning.v3-1-final-pm.yaml` (post-ablation 2026-05-23). Source of truth for context: `docs/v31-polymarket-live-plan.md`.

---

## File Structure

### Created

```
hlanalysis/
  adapters/
    polymarket.py                  # P3: PM WS adapter; mirrors hyperliquid.py shape
    polymarket_normalize.py        # P1: pure WS-payload → NormalizedEvent helpers
    polymarket_gamma.py            # P2: thin sync Gamma REST client
  engine/
    exec_types.py                  # P0: PlaceRequest/OrderAck/etc. moved out of hl_client.py
    exec_client.py                 # P0: ExecutionClient Protocol
    pm_client.py                   # P6/P8: PMClient implementing ExecutionClient
    pm_settlement_poller.py        # P9: Gamma resolution poll → SettlementEvent
  alerts/
    pm_rules.py                    # P10: GasLow / OrderUnconfirmed / RedemptionTimeout

tests/
  unit/
    test_pm_normalize.py           # P1
    test_pm_gamma.py               # P2
    test_pm_adapter.py             # P3
    test_pm_client_paper.py        # P6
    test_pm_client_live.py         # P8 (with httpx mock)
    test_risk_depth_walk.py        # P7
    test_pm_settlement_poller.py   # P9
    test_exec_client_protocol.py   # P0
  integration/
    test_recorder_pm_smoke.py      # P4
    test_engine_pm_paper_loop.py   # P6
  fixtures/
    pm/
      ws_book_frames.jsonl         # P3: recorded raw PM WS frames (1 capture, ~60s)
      gamma_active_btc_updown.json # P2: live Gamma /events response snapshot
      clob_order_responses/        # P8: httpx-mocked CLOB responses
        derive_api_key.json
        order_accepted_fak.json
        order_rejected.json
        cancel_ok.json
        balance.json

scripts/
  capture_pm_ws_fixture.py         # P3: one-shot, connects to PM WS, writes JSONL
  pm_derive_api_key.py             # P8: one-time bootstrap to derive + persist L2 creds

config/
  symbols.yaml                      # P4: add PM subscription block (modified)
  strategy.yaml                     # P5: add v31_pm slot (modified)
  deploy.yaml                       # P5: add accounts.v31_pm + cutover hl_accounts→accounts
```

### Modified

| File | Phase | What changes |
|------|-------|--------------|
| `hlanalysis/engine/hl_client.py` | P0 | Move dataclasses to `exec_types.py`; import them back; class now `class HLClient(ExecutionClient)` |
| `hlanalysis/engine/config.py` | P0 | `hl_accounts` → `accounts: dict[str, AccountConfig]` discriminated union; `load_deploy_config` updated |
| `hlanalysis/engine/runtime.py` | P0/P5 | `AccountSlot.hl` → `AccountSlot.exec`; `_build_slot` dispatches on `AccountConfig` type |
| `hlanalysis/engine/router.py` | P0 | `hl: HLClient` → `exec_client: ExecutionClient`; `self.hl.place(...)` → `self.exec.place(...)` |
| `hlanalysis/engine/scanner.py` | P0 | `pnl_provider` wired from `slot.exec.realized_pnl_since` |
| `hlanalysis/engine/reconcile.py` | P0 | No code change — types now resolved from `exec_types` |
| `hlanalysis/recorder/runner.py` | P4 | Register `PolymarketAdapter` |
| `hlanalysis/engine/risk.py` | P7 | Add depth-walk slippage check |
| `hlanalysis/strategy/theta_harvester.py` | P7 | Add `fee_model: "flat" \| "pm_binary"` to `ThetaHarvesterConfig`; honour in entry edge |
| `pyproject.toml` | P8 | Add `py-clob-client-v2`, `httpx[http2]` |
| `config/deploy.yaml` | P5 | **Hard cutover** `hl_accounts:` → `accounts:` keyed by venue |

---

## Phase 0 — Refactor: ExecutionClient seam + venue-typed AccountConfig

**Goal:** Pure refactor. No behavior change. Every existing test stays green.

### Task 0.1: Extract `exec_types.py`

**Files:**
- Create: `hlanalysis/engine/exec_types.py`
- Modify: `hlanalysis/engine/hl_client.py` (re-export for back-compat)
- Test: `tests/unit/test_hl_client.py` (existing — must still pass)

- [ ] **Step 1: Move dataclasses out of `hl_client.py`**

Create `hlanalysis/engine/exec_types.py` with the existing PlaceRequest, OrderAck, VenuePosition, ClearinghouseState, OpenOrderRow, UserFillRow definitions copied verbatim from `hl_client.py:13-75`. Add this module docstring at the top:

```python
"""Venue-neutral execution-layer types. Shared by HLClient, PMClient, and
any future ExecutionClient implementations. Lifted out of hl_client.py
during the v3.1-PM refactor so the engine wiring can carry one Protocol
type instead of a concrete HL class."""
```

- [ ] **Step 2: Re-export from `hl_client.py`**

Replace the dataclass definitions in `hl_client.py:13-75` with:

```python
from .exec_types import (
    PlaceRequest, OrderAck, VenuePosition, ClearinghouseState,
    OpenOrderRow, UserFillRow,
)
```

Leave `RestError` in `hl_client.py` (it's HL-specific retry-wrapper plumbing).

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/unit/test_hl_client.py tests/unit/test_router.py tests/unit/test_reconcile.py -q`
Expected: PASS (no behavior changes).

- [ ] **Step 4: Commit**

```bash
git add hlanalysis/engine/exec_types.py hlanalysis/engine/hl_client.py
git commit -m "refactor(engine): lift PlaceRequest/OrderAck/etc. into exec_types.py"
```

### Task 0.2: Define `ExecutionClient` Protocol

**Files:**
- Create: `hlanalysis/engine/exec_client.py`
- Create: `tests/unit/test_exec_client_protocol.py`

- [ ] **Step 1: Write the Protocol-conformance test**

Create `tests/unit/test_exec_client_protocol.py`:

```python
from __future__ import annotations

from hlanalysis.engine.exec_client import ExecutionClient
from hlanalysis.engine.hl_client import HLClient


def test_hl_client_satisfies_execution_client_protocol():
    """HLClient must structurally conform to ExecutionClient.

    Protocols use runtime-checkable duck typing; this test pins the surface
    so any future HLClient/PMClient change that drops a method fails fast.
    """
    paper = HLClient(
        account_address="0xtest", api_secret_key="0xfake",
        base_url="https://api.hyperliquid.xyz", paper_mode=True,
    )
    assert isinstance(paper, ExecutionClient)
```

- [ ] **Step 2: Run it — expect FAIL (module missing)**

Run: `uv run pytest tests/unit/test_exec_client_protocol.py -q`
Expected: FAIL with `ModuleNotFoundError: hlanalysis.engine.exec_client`.

- [ ] **Step 3: Write `exec_client.py`**

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .exec_types import (
    ClearinghouseState, OpenOrderRow, OrderAck, PlaceRequest, UserFillRow,
)


@runtime_checkable
class ExecutionClient(Protocol):
    """Venue-neutral order-execution interface.

    Implementations: HLClient (Hyperliquid HIP-4 + perp + spot), PMClient
    (Polymarket CLOB). The Router and Reconciler depend on this Protocol,
    not on any concrete class.
    """

    paper_mode: bool

    def place(self, req: PlaceRequest) -> OrderAck: ...
    def cancel(self, *, cloid: str, symbol: str) -> bool: ...
    def open_orders(self) -> list[OpenOrderRow]: ...
    def clearinghouse_state(self) -> ClearinghouseState: ...
    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]: ...
    def realized_pnl_since(self, since_ts_ns: int) -> float: ...
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/unit/test_exec_client_protocol.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/exec_client.py tests/unit/test_exec_client_protocol.py
git commit -m "feat(engine): define ExecutionClient Protocol; HLClient conforms"
```

### Task 0.3: Venue-typed `AccountConfig` (hard cutover)

**Files:**
- Modify: `hlanalysis/engine/config.py:177-235`
- Modify: `config/deploy.yaml`
- Modify: `tests/unit/test_engine_config.py` (existing tests — update assertions)

- [ ] **Step 1: Write the new-shape test**

Append to `tests/unit/test_engine_config.py`:

```python
import yaml
from hlanalysis.engine.config import (
    AccountConfig, HyperliquidAccount, PolymarketAccount,
    DeployConfig, load_deploy_config,
)


def test_account_config_discriminates_on_venue(tmp_path):
    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text("""
deploy:
  env: test
  accounts:
    v1:
      venue: hyperliquid
      account_address: "0xabc"
      api_secret_key: "0xdef"
      base_url: https://api.hyperliquid.xyz
    v31_pm:
      venue: polymarket
      clob_host: https://clob.polymarket.com
      chain_id: 137
      private_key: "0xfeed"
      clob_api_key: "ak"
      clob_api_secret: "as"
      clob_api_passphrase: "ap"
  alerts:
    telegram:
      bot_token: T
      chat_id: C
  state_db_path: data/engine/state.db
  kill_switch_path: data/engine/halt
""")
    cfg = load_deploy_config(deploy_yaml)
    assert isinstance(cfg.accounts["v1"], HyperliquidAccount)
    assert isinstance(cfg.accounts["v31_pm"], PolymarketAccount)
    assert cfg.accounts["v31_pm"].chain_id == 137
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/unit/test_engine_config.py::test_account_config_discriminates_on_venue -q`
Expected: FAIL with `ImportError` for the new names.

- [ ] **Step 3: Update `engine/config.py`**

Replace `HLConfig` / `DeployConfig.hl_accounts` block (`config.py:177-235`) with:

```python
class _AccountBase(BaseModel):
    model_config = ConfigDict(frozen=True)
    venue: str  # discriminator


class HyperliquidAccount(_AccountBase):
    venue: Literal["hyperliquid"] = "hyperliquid"
    account_address: str
    api_secret_key: str
    base_url: str


class PolymarketAccount(_AccountBase):
    venue: Literal["polymarket"] = "polymarket"
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    private_key: str
    clob_api_key: str
    clob_api_secret: str
    clob_api_passphrase: str


AccountConfig = Annotated[
    HyperliquidAccount | PolymarketAccount,
    Field(discriminator="venue"),
]


# Back-compat alias — many tests still import HLConfig by name. Drop in a
# follow-up release once call sites switch.
HLConfig = HyperliquidAccount


class DeployConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    env: str
    accounts: dict[str, AccountConfig]
    alerts: AlertsConfig
    state_db_path: str
    kill_switch_path: str

    def state_db_path_for(self, alias: str) -> str:
        base = Path(self.state_db_path)
        if len(self.accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)

    def kill_switch_path_for(self, alias: str) -> str:
        base = Path(self.kill_switch_path)
        if len(self.accounts) <= 1 and alias == "default":
            return str(base)
        return str(base.parent / alias / base.name)


# Back-compat property: monitoring code reads `.hl_accounts` directly.
# Kept as a read-only view; remove after the call sites are migrated.
def _hl_accounts(self: "DeployConfig") -> dict[str, HyperliquidAccount]:
    return {a: c for a, c in self.accounts.items() if isinstance(c, HyperliquidAccount)}
DeployConfig.hl_accounts = property(_hl_accounts)  # type: ignore[assignment]
```

Add `Annotated, Literal` imports from `typing` at the top of the file.

Update `load_deploy_config` (`config.py:283-294`):

```python
def load_deploy_config(path: Path) -> DeployConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _substitute_env(raw)
    deploy = raw["deploy"]
    if "accounts" not in deploy:
        raise ValueError(
            f"{path}: deploy must define `accounts:` (venue-typed). "
            "Legacy `hl_accounts:` / `hl:` are no longer supported; see "
            "DEPLOYMENT.md for migration."
        )
    return DeployConfig(**deploy)
```

- [ ] **Step 4: Update `config/deploy.yaml` (hard cutover)**

Replace the file's `hl_accounts:` block with:

```yaml
deploy:
  env: dev
  accounts:
    v1:
      venue: hyperliquid
      account_address: ${HL_ACCOUNT_ADDRESS}
      api_secret_key: ${HL_API_SECRET_KEY}
      base_url: https://api.hyperliquid.xyz
    v31:
      venue: hyperliquid
      account_address: ${HL_ACCOUNT_ADDRESS_V31}
      api_secret_key: ${HL_API_SECRET_KEY_V31}
      base_url: https://api.hyperliquid.xyz
  alerts:
    telegram:
      bot_token: ${TG_BOT_TOKEN}
      chat_id: ${TG_CHAT_ID}
  state_db_path: data/engine/state.db
  kill_switch_path: data/engine/halt
```

The v31_pm block lands in Phase 5.

- [ ] **Step 5: Update call sites referencing `hl_accounts`**

Search & replace in engine code:

```bash
grep -rn "hl_accounts\|deploy_cfg.hl\b" hlanalysis/ tests/
```

Update each call site to use `cfg.accounts[alias]` (preferred) or the
back-compat `cfg.hl_accounts` property. Specifically:
- `runtime.py:328` (`if alias not in self.deploy_cfg.hl_accounts`) → keep using `.hl_accounts` for now (it returns HL-only entries). Phase 5 expands the dispatch.
- `runtime.py:333` (`hl_cfg = self.deploy_cfg.hl_accounts[alias]`) → same.

- [ ] **Step 6: Run full unit suite**

Run: `uv run pytest tests/unit -q -x`
Expected: PASS. If `test_engine_runtime_config.py` fails, fix the YAML literal there to use `accounts:` shape.

- [ ] **Step 7: Commit**

```bash
git add hlanalysis/engine/config.py config/deploy.yaml tests/unit/test_engine_config.py
git commit -m "refactor(engine): venue-typed AccountConfig; hard cutover deploy.yaml

hl_accounts: dict[str, HLConfig] → accounts: dict[str, AccountConfig]
with a Hyperliquid|Polymarket discriminated union. Keep HLConfig and
.hl_accounts as back-compat aliases so call sites can migrate
incrementally. No behavior change for HL slots."
```

### Task 0.4: Rename `AccountSlot.hl` → `AccountSlot.exec`

**Files:**
- Modify: `hlanalysis/engine/runtime.py:169-197, 326-381, 489-499, 543, 620`
- Modify: `hlanalysis/engine/router.py:34-54, 183-202, 526-527`
- Modify: `tests/integration/test_engine_multi_account.py` (existing)

- [ ] **Step 1: Run grep to find all sites**

```bash
grep -rn "slot\.hl\|self\.hl\b\|hl=slot\.hl\|hl=self\.hl" hlanalysis/ tests/
```

Expect ~25 hits across `runtime.py`, `router.py`, `scanner.py`, tests.

- [ ] **Step 2: Mechanical rename**

In `runtime.py`:
- `AccountSlot.hl: HLClient` → `AccountSlot.exec_client: ExecutionClient` (avoid `exec` which shadows builtin).
- `slot.hl.X()` → `slot.exec_client.X()` everywhere.
- The constructor call at `runtime.py:344-349` becomes:
  ```python
  exec_client = HLClient(
      account_address=hl_cfg.account_address,
      api_secret_key=hl_cfg.api_secret_key,
      base_url=hl_cfg.base_url,
      paper_mode=s_cfg.paper_mode,
  )
  ```

In `router.py`:
- `__init__(... hl: HLClient ...)` → `__init__(... exec_client: ExecutionClient ...)`.
- `self.hl = hl` → `self.exec_client = exec_client`.
- All `self.hl.X(...)` → `self.exec_client.X(...)`.

In `scanner.py`: nothing — `pnl_provider` is already injected as a callable, not a client ref.

Update factory parameter names in `runtime.py:209` (`hl_client_factory`) to `exec_client_factory` and adjust the `from_single` wrapper.

- [ ] **Step 3: Update tests**

In `tests/integration/test_engine_multi_account.py` and any others that pass `hl=` or `hl_client_factory=`, rename to match. Keep the test bodies unchanged.

- [ ] **Step 4: Run full suite**

```bash
uv run pytest -q -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/ tests/
git commit -m "refactor(engine): AccountSlot.hl → AccountSlot.exec_client; Router takes ExecutionClient"
```

---

## Phase 1 — PM Normalizers (pure functions)

**Goal:** Translate raw PM CLOB WS payloads into `NormalizedEvent`s. No IO. 100% testable with literal JSON dicts.

### Task 1.1: PM book-update normalizer

**Files:**
- Create: `hlanalysis/adapters/polymarket_normalize.py`
- Create: `tests/unit/test_pm_normalize.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_pm_normalize.py
from hlanalysis.adapters.polymarket_normalize import (
    parse_book_message, parse_trade_message,
)
from hlanalysis.events import BookSnapshotEvent, TradeEvent


def test_parse_book_message_yields_book_snapshot_event():
    # Sample PM CLOB WS book payload (from docs.polymarket.com/api/clob/websocket).
    payload = {
        "event_type": "book",
        "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
        "market": "0xabc",
        "timestamp": "1716545000123",
        "hash": "h",
        "bids": [{"price": "0.92", "size": "150"}, {"price": "0.91", "size": "80"}],
        "asks": [{"price": "0.93", "size": "60"}, {"price": "0.94", "size": "120"}],
    }
    ev = parse_book_message(payload, local_recv_ts=1716545000200_000_000)
    assert isinstance(ev, BookSnapshotEvent)
    assert ev.venue == "polymarket"
    assert ev.symbol == payload["asset_id"]
    assert ev.bid_px == [0.92, 0.91]
    assert ev.bid_sz == [150.0, 80.0]
    assert ev.ask_px == [0.93, 0.94]
    assert ev.exchange_ts == 1716545000123 * 1_000_000  # ms → ns


def test_parse_trade_message_yields_trade_event():
    payload = {
        "event_type": "last_trade_price",
        "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
        "market": "0xabc",
        "price": "0.927",
        "size": "100",
        "side": "BUY",
        "timestamp": "1716545001000",
        "trade_id": "tid-9",
    }
    ev = parse_trade_message(payload, local_recv_ts=1716545001100_000_000)
    assert isinstance(ev, TradeEvent)
    assert ev.symbol == payload["asset_id"]
    assert ev.price == 0.927
    assert ev.size == 100.0
    assert ev.side == "buy"
    assert ev.trade_id == "tid-9"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_normalize.py -q
```
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# hlanalysis/adapters/polymarket_normalize.py
"""Pure PM CLOB WS payload → NormalizedEvent translators.

PM WS message format reference: docs.polymarket.com/developers/CLOB/websocket.
All `timestamp` fields are milliseconds since epoch; we convert to ns to
match the rest of the engine. Token IDs are 76-digit ERC-1155 ID strings
and are carried through verbatim as `symbol`.
"""
from __future__ import annotations

from typing import Any

from ..events import (
    BboEvent, BookSnapshotEvent, Mechanism, ProductType, TradeEvent,
)

_VENUE = "polymarket"


def _ts_ms_to_ns(ms_str: str | int) -> int:
    return int(ms_str) * 1_000_000


def parse_book_message(payload: dict[str, Any], *, local_recv_ts: int) -> BookSnapshotEvent:
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=[float(b["price"]) for b in bids],
        bid_sz=[float(b["size"]) for b in bids],
        ask_px=[float(a["price"]) for a in asks],
        ask_sz=[float(a["size"]) for a in asks],
    )


def parse_trade_message(payload: dict[str, Any], *, local_recv_ts: int) -> TradeEvent:
    side_raw = str(payload.get("side", "")).upper()
    side = "buy" if side_raw == "BUY" else ("sell" if side_raw == "SELL" else "unknown")
    return TradeEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        price=float(payload["price"]),
        size=float(payload["size"]),
        side=side,
        trade_id=str(payload.get("trade_id", "")) or None,
    )


def parse_price_change_message(payload: dict[str, Any], *, local_recv_ts: int) -> BookSnapshotEvent | None:
    """PM `price_change` events carry a partial book delta (changed levels
    only). We snapshot what's present; missing sides yield empty lists.

    The engine MarketState (BookSnapshotEvent branch, market_state.py:82-88)
    only reads top-of-book, so partial snapshots are acceptable — but we
    return None when both sides are empty to avoid clobbering a previous
    full snapshot.
    """
    changes = payload.get("changes") or []
    bids = [c for c in changes if str(c.get("side", "")).upper() == "BUY"]
    asks = [c for c in changes if str(c.get("side", "")).upper() == "SELL"]
    if not bids and not asks:
        return None
    return BookSnapshotEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=str(payload["asset_id"]),
        exchange_ts=_ts_ms_to_ns(payload.get("timestamp", 0)),
        local_recv_ts=local_recv_ts,
        bid_px=[float(b["price"]) for b in bids],
        bid_sz=[float(b["size"]) for b in bids],
        ask_px=[float(a["price"]) for a in asks],
        ask_sz=[float(a["size"]) for a in asks],
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_normalize.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/adapters/polymarket_normalize.py tests/unit/test_pm_normalize.py
git commit -m "feat(pm): parse_book_message + parse_trade_message normalizers"
```

### Task 1.2: PM QuestionMetaEvent + SettlementEvent from Gamma market dict

**Files:**
- Modify: `hlanalysis/adapters/polymarket_normalize.py`
- Modify: `tests/unit/test_pm_normalize.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_pm_normalize.py`:

```python
from hlanalysis.adapters.polymarket_normalize import (
    parse_gamma_market_to_question_meta, parse_gamma_market_to_settlement,
)


_SAMPLE_GAMMA_MARKET = {
    "conditionId": "0xcond123",
    "clobTokenIds": '["71321...992563", "71321...111111"]',
    "startDate": "2026-05-24T00:00:00Z",
    "endDate": "2026-05-25T00:00:00Z",
    "description": "Will BTC go up or down? Resolves based on the Binance 1 "
                   "minute candle for BTC/USDT May 24 '26 20:00 in the ET timezone...",
    "outcomePrices": '["0.92","0.08"]',
    "groupItemTitle": "",
}


def test_parse_gamma_market_to_question_meta_binary():
    ev = parse_gamma_market_to_question_meta(
        _SAMPLE_GAMMA_MARKET,
        series_slug="btc-up-or-down-daily",
        local_recv_ts=1716545000000_000_000,
    )
    assert ev.event_type == "question_meta"
    assert ev.venue == "polymarket"
    # YES token is leg 0, NO is leg 1
    keys = dict(zip(ev.keys, ev.values))
    assert keys["class"] == "priceBinary"
    assert keys["underlying"] == "BTC"
    assert keys["yes_token_id"] == "71321...992563"
    assert keys["no_token_id"] == "71321...111111"
    assert "strike_ref_ts_ns" in keys


def test_parse_gamma_market_to_settlement_when_resolved():
    resolved = dict(_SAMPLE_GAMMA_MARKET, outcomePrices='["1.0","0.0"]')
    ev = parse_gamma_market_to_settlement(
        resolved, series_slug="btc-up-or-down-daily",
        local_recv_ts=1716631400000_000_000,
    )
    assert ev is not None
    assert ev.event_type == "settlement"
    assert ev.settled_side_idx == 0  # YES won


def test_parse_gamma_market_to_settlement_returns_none_when_open():
    ev = parse_gamma_market_to_settlement(
        _SAMPLE_GAMMA_MARKET, series_slug="btc-up-or-down-daily",
        local_recv_ts=0,
    )
    assert ev is None
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_normalize.py::test_parse_gamma_market_to_question_meta_binary -q
```
Expected: FAIL `ImportError`.

- [ ] **Step 3: Implement (promote backtest helpers)**

Append to `hlanalysis/adapters/polymarket_normalize.py`:

```python
import json
import re
from datetime import datetime, timezone

from ..events import QuestionMetaEvent, SettlementEvent


_BTC_UPDOWN_STRIKE_RULE = re.compile(
    r"Binance 1 minute candle for BTC/USDT\s+(\w+)\s+(\d+)\s+'(\d{2})\s+"
    r"(\d{1,2}):(\d{2})\s+in the ET timezone",
    re.IGNORECASE,
)
_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def _parse_strike_ref_ts_ns(description: str) -> int | None:
    if not description:
        return None
    m = _BTC_UPDOWN_STRIKE_RULE.search(description)
    if not m:
        return None
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except Exception:
        return None
    mon, day, yr2, hh, mm = m.groups()
    if mon not in _MONTHS:
        return None
    dt = datetime(2000 + int(yr2), _MONTHS[mon], int(day), int(hh), int(mm), tzinfo=et)
    return int(dt.astimezone(timezone.utc).timestamp() * 1e9)


def _question_idx_from_condition(condition_id: str) -> int:
    """Stable 31-bit hash so the index fits in a SQLite int column. Matches
    backtest/data/polymarket.py:_question_idx for replay parity."""
    return hash(condition_id) & 0x7FFFFFFF


def _parse_token_ids(market: dict) -> tuple[str, str] | None:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    toks = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(toks, list) or len(toks) < 2:
        return None
    return str(toks[0]), str(toks[1])


def _parse_iso_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def parse_gamma_market_to_question_meta(
    market: dict, *, series_slug: str, local_recv_ts: int,
) -> QuestionMetaEvent:
    cond_id = str(market.get("conditionId") or market.get("id") or "")
    tokens = _parse_token_ids(market)
    if tokens is None:
        raise ValueError(f"market {cond_id}: clobTokenIds missing or malformed")
    yes_t, no_t = tokens
    end_iso = market.get("endDate") or ""
    desc = market.get("description") or ""
    strike_ref_ts_ns = _parse_strike_ref_ts_ns(desc)

    keys = ["class", "underlying", "yes_token_id", "no_token_id",
            "expiry_ns", "series_slug", "condition_id"]
    values = ["priceBinary", "BTC", yes_t, no_t,
              str(_parse_iso_ns(end_iso)) if end_iso else "0",
              series_slug, cond_id]
    if strike_ref_ts_ns is not None:
        keys.append("strike_ref_ts_ns")
        values.append(str(strike_ref_ts_ns))

    return QuestionMetaEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=yes_t,  # primary leg
        exchange_ts=0,
        local_recv_ts=local_recv_ts,
        question_idx=_question_idx_from_condition(cond_id),
        named_outcome_idxs=[0, 1],
        keys=keys,
        values=values,
    )


def parse_gamma_market_to_settlement(
    market: dict, *, series_slug: str, local_recv_ts: int,
) -> SettlementEvent | None:
    """Returns a SettlementEvent iff the market has resolved (one of YES/NO
    has outcome price 1.0). Open markets return None.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None
    prices = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(prices, list) or len(prices) != 2:
        return None
    yes_p, no_p = float(prices[0]), float(prices[1])
    if yes_p >= 0.99:
        settled_idx = 0
    elif no_p >= 0.99:
        settled_idx = 1
    else:
        return None  # not resolved yet
    tokens = _parse_token_ids(market)
    if tokens is None:
        return None
    return SettlementEvent(
        venue=_VENUE,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=tokens[settled_idx],
        exchange_ts=local_recv_ts,
        local_recv_ts=local_recv_ts,
        settled_side_idx=settled_idx,
        settle_price=1.0,
        settle_ts=local_recv_ts,
        keys=["series_slug", "condition_id"],
        values=[series_slug, str(market.get("conditionId") or "")],
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_normalize.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/adapters/polymarket_normalize.py tests/unit/test_pm_normalize.py
git commit -m "feat(pm): question_meta + settlement normalizers from Gamma payloads"
```

---

## Phase 2 — Gamma REST client

### Task 2.1: Capture a live Gamma response into a fixture

**Files:**
- Create: `tests/fixtures/pm/gamma_active_btc_updown.json`

- [ ] **Step 1: Fetch and save**

```bash
mkdir -p tests/fixtures/pm
curl -s 'https://gamma-api.polymarket.com/events?series_slug=btc-up-or-down-daily&closed=false&limit=20' \
  | python3 -m json.tool > tests/fixtures/pm/gamma_active_btc_updown.json
```

- [ ] **Step 2: Sanity-check shape**

```bash
python3 -c "import json; d=json.load(open('tests/fixtures/pm/gamma_active_btc_updown.json')); print('events:', len(d), 'first market keys:', sorted(d[0]['markets'][0].keys())[:10])"
```

Expected: at least 1 event, market keys include `conditionId`, `clobTokenIds`, `endDate`, `outcomePrices`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/pm/gamma_active_btc_updown.json
git commit -m "test(pm): record live Gamma /events response for BTC Up/Down series"
```

### Task 2.2: GammaClient

**Files:**
- Create: `hlanalysis/adapters/polymarket_gamma.py`
- Create: `tests/unit/test_pm_gamma.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_pm_gamma.py
from __future__ import annotations
import json
from pathlib import Path

import pytest

from hlanalysis.adapters.polymarket_gamma import GammaClient


FIXTURE = Path("tests/fixtures/pm/gamma_active_btc_updown.json")


class _FakeHttp:
    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict):
        self.calls.append((url, params))
        return self._pages.pop(0) if self._pages else []


def test_fetch_active_markets_paginates_until_short_page():
    page1 = [{"id": f"e{i}"} for i in range(100)]
    page2 = [{"id": "e100"}, {"id": "e101"}]
    http = _FakeHttp(pages=[page1, page2])
    gc = GammaClient(http_get=http.get)
    out = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    assert len(out) == 102
    assert http.calls[0][1]["offset"] == 0
    assert http.calls[1][1]["offset"] == 100


def test_extract_active_binary_markets_from_fixture():
    raw = json.loads(FIXTURE.read_text())
    http = _FakeHttp(pages=[raw, []])
    gc = GammaClient(http_get=http.get)
    events = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    markets = list(gc.iter_binary_markets(events))
    assert markets  # fixture must contain at least one
    for m in markets:
        assert m.get("conditionId")
        assert m.get("clobTokenIds")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_gamma.py -q
```

- [ ] **Step 3: Implement**

```python
# hlanalysis/adapters/polymarket_gamma.py
"""Synchronous Gamma REST client for Polymarket market discovery + resolution.

The PM CLOB WS does not push market-creation or market-resolution events;
those are observed by polling the Gamma `/events` REST endpoint. This
module is pure HTTP — no asyncio — so it can be called from a background
asyncio task via `loop.run_in_executor` without the adapter pulling in
httpx-async or aiohttp just for this path.

`http_get` is injected so tests don't hit the network.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import requests
from loguru import logger

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_PAGE_LIMIT = 100  # Gamma caps responses at 100 even when limit > 100


def _real_http_get(url: str, params: dict[str, Any]) -> list[dict] | dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


class GammaClient:
    def __init__(
        self, *,
        http_get: Callable[[str, dict[str, Any]], Any] = _real_http_get,
        base_url: str = _GAMMA_BASE,
    ) -> None:
        self._get = http_get
        self._base = base_url

    def fetch_events(
        self, *, series_slug: str, closed: bool, max_pages: int = 50,
    ) -> list[dict]:
        out: list[dict] = []
        offset = 0
        for _ in range(max_pages):
            page = self._get(f"{self._base}/events", {
                "series_slug": series_slug,
                "closed": "true" if closed else "false",
                "limit": _PAGE_LIMIT,
                "offset": offset,
            })
            if not isinstance(page, list) or not page:
                break
            out.extend(page)
            if len(page) < _PAGE_LIMIT:
                break
            offset += len(page)
        else:
            logger.warning("gamma fetch_events hit max_pages={}", max_pages)
        return out

    @staticmethod
    def iter_binary_markets(events: list[dict]) -> Iterator[dict]:
        """Yield single-market binary entries (skips multi-market bucket events)."""
        for ev in events:
            markets = ev.get("markets") or []
            if len(markets) != 1:
                continue
            mk = markets[0]
            if not mk.get("clobTokenIds") or not mk.get("conditionId"):
                continue
            yield mk
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_gamma.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/adapters/polymarket_gamma.py tests/unit/test_pm_gamma.py
git commit -m "feat(pm): GammaClient with paginated /events fetch"
```

---

## Phase 3 — PolymarketAdapter (WS + REST)

### Task 3.1: Capture a recorded WS fixture (one-shot)

**Files:**
- Create: `scripts/capture_pm_ws_fixture.py`
- Create: `tests/fixtures/pm/ws_book_frames.jsonl`

- [ ] **Step 1: Write the capture script**

```python
# scripts/capture_pm_ws_fixture.py
"""One-shot: connect to PM CLOB market WS, subscribe to the YES/NO tokens of
the next-expiring BTC-Up-or-Down market, dump 60s of raw frames to JSONL.

Usage:
  uv run python scripts/capture_pm_ws_fixture.py \
      --out tests/fixtures/pm/ws_book_frames.jsonl --seconds 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

from hlanalysis.adapters.polymarket_gamma import GammaClient

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


async def _capture(out_path: Path, seconds: int) -> None:
    gc = GammaClient()
    events = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    markets = list(gc.iter_binary_markets(events))
    if not markets:
        sys.exit("No active BTC Up/Down markets found.")
    mk = sorted(markets, key=lambda m: m.get("endDate", ""))[0]
    token_ids = json.loads(mk["clobTokenIds"])
    print(f"capturing {mk['conditionId']} → {token_ids}")

    deadline = time.time() + seconds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        async with websockets.connect(_WS_URL, ping_interval=30) as ws:
            await ws.send(json.dumps({"type": "market", "assets_ids": token_ids}))
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                fout.write(msg if msg.endswith("\n") else msg + "\n")
                fout.flush()
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    asyncio.run(_capture(args.out, args.seconds))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to capture the fixture**

```bash
uv run python scripts/capture_pm_ws_fixture.py \
    --out tests/fixtures/pm/ws_book_frames.jsonl --seconds 60
```

Expected: file appears with ≥ 30 lines of JSON. If 0 lines, PM WS likely changed message format — pivot the script to print the first 5 frames before subscribing to confirm the subscription envelope.

- [ ] **Step 3: Commit script + fixture**

```bash
git add scripts/capture_pm_ws_fixture.py tests/fixtures/pm/ws_book_frames.jsonl
git commit -m "test(pm): one-shot capture script + recorded 60s WS fixture"
```

### Task 3.2: PolymarketAdapter — skeleton + WS connect

**Files:**
- Create: `hlanalysis/adapters/polymarket.py`
- Create: `tests/unit/test_pm_adapter.py`

- [ ] **Step 1: Write failing test (uses fixture replay)**

```python
# tests/unit/test_pm_adapter.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hlanalysis.adapters.polymarket import PolymarketAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import (
    BookSnapshotEvent, Mechanism, ProductType, TradeEvent, QuestionMetaEvent,
)


FIXTURE = Path("tests/fixtures/pm/ws_book_frames.jsonl")


class _FakeWS:
    """Async-iter shim that replays JSONL fixture lines as if from a socket."""

    def __init__(self, frames: list[str]):
        self._frames = list(frames)
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._frames:
            await asyncio.sleep(0.05)
            raise asyncio.CancelledError
        return self._frames.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_adapter_emits_book_and_trade_events_from_fixture():
    frames = FIXTURE.read_text().splitlines()
    fake_ws = _FakeWS(frames)
    sub = Subscription(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="*",
        channels=("trades", "book"),
        match={"series_slug": "btc-up-or-down-daily", "underlying": "BTC"},
    )
    adapter = PolymarketAdapter(
        ws_factory=lambda url: fake_ws,
        gamma_client=_StubGamma(),
    )
    events: list = []
    async def _drain() -> None:
        try:
            async for ev in adapter.stream([sub]):
                events.append(ev)
        except asyncio.CancelledError:
            pass

    await asyncio.wait_for(_drain(), timeout=2.0)
    book = [e for e in events if isinstance(e, BookSnapshotEvent)]
    trade = [e for e in events if isinstance(e, TradeEvent)]
    qmeta = [e for e in events if isinstance(e, QuestionMetaEvent)]
    assert book or trade, "fixture must contain at least one book or trade frame"
    assert qmeta, "adapter must emit a QuestionMetaEvent on startup"


class _StubGamma:
    def fetch_events(self, **kw):
        return [{
            "markets": [{
                "conditionId": "0xfixturecond",
                "clobTokenIds": '["tok-yes","tok-no"]',
                "endDate": "2026-05-25T00:00:00Z",
                "description": "Will BTC go up or down? Resolves based on the "
                               "Binance 1 minute candle for BTC/USDT May 24 '26 "
                               "20:00 in the ET timezone...",
                "outcomePrices": '["0.5","0.5"]',
            }]
        }]
    @staticmethod
    def iter_binary_markets(events):
        for ev in events:
            yield ev["markets"][0]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_adapter.py -q
```

- [ ] **Step 3: Implement adapter**

```python
# hlanalysis/adapters/polymarket.py
"""Polymarket CLOB live-data adapter.

Inputs: list[Subscription] (typically a single wildcard for the BTC Up/Down
daily series). Outputs: AsyncIterator[NormalizedEvent] — BookSnapshotEvent,
TradeEvent, QuestionMetaEvent, SettlementEvent.

Architecture:
  - On startup: poll Gamma /events to learn the currently-active markets;
    emit one QuestionMetaEvent per match-filtered market.
  - Subscribe to the PM CLOB WS for all matched token IDs (yes + no legs).
  - Receive loop: dispatch by `event_type` to the normalizers in
    polymarket_normalize.py and yield results to the consumer.
  - Background poller: re-fetch Gamma every 60s to pick up new daily
    markets and resolved settlements. Resolved markets emit a
    SettlementEvent; new markets emit a QuestionMetaEvent + subscribe their
    new token IDs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import websockets

from ..config import Subscription
from ..events import (
    HealthEvent, Mechanism, NormalizedEvent, ProductType,
)
from .base import VenueAdapter
from .polymarket_gamma import GammaClient
from .polymarket_normalize import (
    parse_book_message, parse_gamma_market_to_question_meta,
    parse_gamma_market_to_settlement, parse_price_change_message,
    parse_trade_message,
)

log = logging.getLogger(__name__)

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_GAMMA_POLL_S = 60


class PolymarketAdapter(VenueAdapter):
    venue = "polymarket"

    def __init__(
        self, *,
        ws_factory: Callable[[str], Any] | None = None,
        gamma_client: GammaClient | None = None,
    ) -> None:
        self._ws_factory = ws_factory or (
            lambda url: websockets.connect(url, ping_interval=30)
        )
        self._gamma = gamma_client or GammaClient()

    def supports(self, product_type: ProductType, mechanism: Mechanism) -> bool:
        return product_type == ProductType.PREDICTION_BINARY and mechanism == Mechanism.CLOB

    async def stream(
        self, subscriptions: list[Subscription]
    ) -> AsyncIterator[NormalizedEvent]:
        # Resolve subscriptions to series slugs + match filters. Only PM subs
        # land here in practice — runtime/recorder route by venue.
        pm_subs = [s for s in subscriptions if s.venue == self.venue]
        if not pm_subs:
            return

        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)
        active_tokens: set[str] = set()
        known_conditions: dict[str, dict] = {}  # cond_id → last-seen market dict

        async def _gamma_poll_once() -> None:
            for sub in pm_subs:
                series = (sub.match or {}).get("series_slug")
                if not series:
                    continue
                series_str = series if isinstance(series, str) else series[0]
                events = self._gamma.fetch_events(
                    series_slug=series_str, closed=False,
                )
                for mk in self._gamma.iter_binary_markets(events):
                    cond_id = str(mk["conditionId"])
                    is_new = cond_id not in known_conditions
                    known_conditions[cond_id] = mk
                    if is_new:
                        qmeta = parse_gamma_market_to_question_meta(
                            mk, series_slug=series_str, local_recv_ts=time.time_ns(),
                        )
                        await queue.put(qmeta)
                        toks = json.loads(mk["clobTokenIds"])
                        active_tokens.update(str(t) for t in toks)
                    # Resolution check (works for both new + previously-seen)
                    settle = parse_gamma_market_to_settlement(
                        mk, series_slug=series_str, local_recv_ts=time.time_ns(),
                    )
                    if settle is not None:
                        await queue.put(settle)

        async def _gamma_loop() -> None:
            while True:
                try:
                    await _gamma_poll_once()
                except Exception:
                    log.exception("gamma poll crashed")
                await asyncio.sleep(_GAMMA_POLL_S)

        async def _ws_loop() -> None:
            # First poll synchronous so initial subscribe carries the tokens.
            await _gamma_poll_once()
            if not active_tokens:
                await queue.put(self._health("no_active_markets",
                                              "Gamma returned 0 active markets"))
                return
            ws_ctx = self._ws_factory(_WS_URL)
            async with ws_ctx as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": sorted(active_tokens),
                }))
                while True:
                    raw = await ws.recv()
                    self._dispatch_frame(raw, queue)

        tasks = [
            asyncio.create_task(_ws_loop()),
            asyncio.create_task(_gamma_loop()),
        ]
        try:
            while True:
                ev = await queue.get()
                yield ev
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _dispatch_frame(
        self, raw: str | bytes, queue: asyncio.Queue,
    ) -> None:
        try:
            payloads = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("pm ws: undecodable frame discarded")
            return
        # PM CLOB sends either single objects or arrays.
        items = payloads if isinstance(payloads, list) else [payloads]
        now = time.time_ns()
        for p in items:
            t = p.get("event_type")
            try:
                if t == "book":
                    queue.put_nowait(parse_book_message(p, local_recv_ts=now))
                elif t == "price_change":
                    ev = parse_price_change_message(p, local_recv_ts=now)
                    if ev is not None:
                        queue.put_nowait(ev)
                elif t == "last_trade_price":
                    queue.put_nowait(parse_trade_message(p, local_recv_ts=now))
                # Unknown event_types (e.g. tick_size_change) ignored — not needed
                # by the strategy or recorder.
            except (KeyError, ValueError, TypeError) as e:
                log.warning("pm ws: malformed %s frame discarded: %s", t, e)

    def _health(self, kind: str, detail: str) -> HealthEvent:
        return HealthEvent(
            venue=self.venue,
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="*",
            exchange_ts=time.time_ns(),
            local_recv_ts=time.time_ns(),
            kind=kind, detail=detail,
        )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_adapter.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/adapters/polymarket.py tests/unit/test_pm_adapter.py
git commit -m "feat(pm): PolymarketAdapter — WS + Gamma poll; book/trade/qmeta/settlement events"
```

### Task 3.3: Reconnect / backoff

**Files:**
- Modify: `hlanalysis/adapters/polymarket.py:_ws_loop`
- Modify: `tests/unit/test_pm_adapter.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_adapter_reconnects_on_ws_close():
    closing_ws = _FakeWS([])
    fresh_ws = _FakeWS(['{"event_type":"book","asset_id":"tok-yes","timestamp":"1","bids":[],"asks":[{"price":"0.5","size":"10"}]}'])

    calls = {"n": 0}
    def factory(_url: str):
        calls["n"] += 1
        return closing_ws if calls["n"] == 1 else fresh_ws

    adapter = PolymarketAdapter(ws_factory=factory, gamma_client=_StubGamma())
    sub = Subscription(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="*", channels=("book",),
        match={"series_slug": "btc-up-or-down-daily"},
    )
    events: list = []
    async def _drain():
        try:
            async for ev in adapter.stream([sub]):
                events.append(ev)
                if len(events) >= 2:
                    return
        except asyncio.CancelledError:
            pass
    await asyncio.wait_for(_drain(), timeout=3.0)
    assert calls["n"] >= 2  # initial + at least one reconnect
```

- [ ] **Step 2: Wrap `_ws_loop` body in a retry loop**

Replace `_ws_loop` in `polymarket.py`:

```python
async def _ws_loop() -> None:
    await _gamma_poll_once()
    if not active_tokens:
        await queue.put(self._health("no_active_markets", "Gamma returned 0"))
        return
    backoff = 1.0
    while True:
        try:
            ws_ctx = self._ws_factory(_WS_URL)
            async with ws_ctx as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": sorted(active_tokens),
                }))
                await queue.put(self._health("subscribed",
                                              f"{len(active_tokens)} tokens"))
                backoff = 1.0  # reset on successful (re)connect
                while True:
                    raw = await ws.recv()
                    self._dispatch_frame(raw, queue)
        except (websockets.exceptions.ConnectionClosed, OSError, asyncio.IncompleteReadError) as e:
            await queue.put(self._health("reconnect", str(e)[:200]))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
```

- [ ] **Step 3: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_adapter.py -q
```

- [ ] **Step 4: Commit**

```bash
git add hlanalysis/adapters/polymarket.py tests/unit/test_pm_adapter.py
git commit -m "feat(pm): adapter reconnect + exponential backoff on WS close"
```

---

## Phase 4 — Recorder wiring (CHECKPOINT: live PM data → parquet)

### Task 4.1: Register PolymarketAdapter in recorder

**Files:**
- Modify: `hlanalysis/recorder/runner.py:17-20`

- [ ] **Step 1: Add the registration**

```python
# hlanalysis/recorder/runner.py
from ..adapters.polymarket import PolymarketAdapter

ADAPTERS: dict[str, type[VenueAdapter]] = {
    "hyperliquid": HyperliquidAdapter,
    "binance": BinanceAdapter,
    "polymarket": PolymarketAdapter,
}
```

- [ ] **Step 2: Run existing recorder tests**

```bash
uv run pytest tests/ -q -k recorder
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add hlanalysis/recorder/runner.py
git commit -m "feat(recorder): register PolymarketAdapter"
```

### Task 4.2: Add PM subscription to `config/symbols.yaml`

**Files:**
- Modify: `config/symbols.yaml`

- [ ] **Step 1: Append the PM block**

```yaml
  # Polymarket BTC Up/Down daily binary — read-side feed for the new
  # v31_pm engine slot. Wildcard `*` symbol so the adapter discovers
  # currently-active markets via Gamma and subscribes their token IDs.
  - venue: polymarket
    product_type: prediction_binary
    mechanism: clob
    symbol: "*"
    channels: [trades, book]
    match:
      underlying: BTC
      class: priceBinary
      series_slug: btc-up-or-down-daily
```

- [ ] **Step 2: Validate the config loads**

```bash
uv run python -c "from hlanalysis.config import load_config; from pathlib import Path; c = load_config(Path('config/symbols.yaml')); pm = [s for s in c.subscriptions if s.venue=='polymarket']; print('pm subs:', len(pm), pm[0].match if pm else '')"
```
Expected: `pm subs: 1 {...series_slug...}`.

- [ ] **Step 3: Commit**

```bash
git add config/symbols.yaml
git commit -m "config(recorder): subscribe to Polymarket BTC Up/Down daily binary"
```

### Task 4.3: Integration smoke test — recorder writes PM parquet

**Files:**
- Create: `tests/integration/test_recorder_pm_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/integration/test_recorder_pm_smoke.py
"""End-to-end: recorder consuming a fake PolymarketAdapter writes parquet
with the expected schema. No network access; uses a stubbed adapter that
yields a known sequence of normalized events.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import (
    BookSnapshotEvent, Mechanism, ProductType, TradeEvent,
)
from hlanalysis.recorder.writer import ParquetWriter


class _StubAdapter(VenueAdapter):
    venue = "polymarket"

    def supports(self, *a, **k): return True

    async def stream(self, _subs):
        now = 1_716_000_000_000_000_000
        yield BookSnapshotEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="71321045679252212594626385532706912750332728571942532289631379312455583992563",
            exchange_ts=now, local_recv_ts=now,
            bid_px=[0.92], bid_sz=[100.0], ask_px=[0.93], ask_sz=[80.0],
        )
        yield TradeEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="71321045679252212594626385532706912750332728571942532289631379312455583992563",
            exchange_ts=now + 1, local_recv_ts=now + 1,
            price=0.927, size=10.0, side="buy", trade_id="t1",
        )


@pytest.mark.asyncio
async def test_recorder_writes_pm_book_and_trade(tmp_path):
    writer = ParquetWriter(tmp_path, flush_interval_s=0.01)
    adapter = _StubAdapter()
    sub = Subscription(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="*",
        channels=("trades", "book"),
    )
    async for ev in adapter.stream([sub]):
        writer.write(ev.model_dump(mode="python"))
    writer.maybe_flush()
    writer.close()

    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert parquet_files, "ParquetWriter produced no files"
    schemas = [pq.read_table(p).schema for p in parquet_files]
    assert any("symbol" in s.names for s in schemas)
```

- [ ] **Step 2: Run — expect PASS**

```bash
uv run pytest tests/integration/test_recorder_pm_smoke.py -q
```

If `ParquetWriter` can't handle the 76-digit token_id as a `string` column (it might infer `int64` and overflow), the test will surface that and we fix it inside `recorder/writer.py` here — adding an explicit dtype map for the `symbol` column.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_recorder_pm_smoke.py
git commit -m "test(recorder): smoke test PM book + trade events round-trip parquet"
```

### Task 4.4: Live recorder smoke (manual)

**Files:** none (operational step).

- [ ] **Step 1: Launch recorder against live PM**

```bash
uv run python -m hlanalysis.recorder.main \
    --config config/symbols.yaml \
    --data-root data/recorder \
    --log-level INFO
```

- [ ] **Step 2: After ~5 minutes, sanity-check the output**

```bash
find data/recorder -name "*polymarket*" -newer /tmp/now -ls
uv run python -c "import pyarrow.parquet as pq, glob; ps=sorted(glob.glob('data/recorder/**/polymarket*book*.parquet', recursive=True))[-1:]; print('latest:', ps); print(pq.read_table(ps[0]).slice(0,3).to_pandas())"
```
Expected: at least one parquet with book rows; columns `symbol` (string), `bid_px`, `ask_px`, etc.

- [ ] **Step 3: Kill recorder**

`Ctrl-C`. **CHECKPOINT REACHED: live PM data is now flowing into parquet alongside HL/Binance.** Backtester `polymarket.py` cache can be re-pointed at this data once a few days have accumulated, closing the recorder↔backtest loop.

- [ ] **Step 4: Commit operational notes**

Append to `DEPLOYMENT.md` under a new "Polymarket data" section, then:

```bash
git add DEPLOYMENT.md
git commit -m "docs(deploy): note Polymarket recorder feed verified live"
```

---

## Phase 5 — Engine wiring for v31_pm slot

### Task 5.1: Add PolymarketAccount dispatch to `_build_slot`

**Files:**
- Modify: `hlanalysis/engine/runtime.py:326-381`
- Create: `hlanalysis/engine/pm_client.py` (stub only — full impl in P6/P8)

- [ ] **Step 1: Write the stub PMClient**

```python
# hlanalysis/engine/pm_client.py
"""Polymarket execution client. Implements ExecutionClient.

This is the paper-mode-only stub in Phase 5. Live `py-clob-client-v2`
wiring lands in Phase 8.
"""
from __future__ import annotations

from .exec_types import (
    ClearinghouseState, OpenOrderRow, OrderAck, PlaceRequest, UserFillRow,
)


class PMClient:
    paper_mode: bool

    def __init__(self, *, paper_mode: bool, **_kwargs) -> None:
        self.paper_mode = paper_mode
        if not paper_mode:
            raise NotImplementedError(
                "PMClient live mode lands in Phase 8 of the v3.1-PM plan."
            )

    def place(self, req: PlaceRequest) -> OrderAck:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, *, cloid: str, symbol: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def open_orders(self) -> list[OpenOrderRow]:
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        return []

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        return 0.0
```

- [ ] **Step 2: Update `_build_slot`**

In `runtime.py:326-381`, replace the HL-only construction with venue dispatch:

```python
def _build_slot(self, s_cfg: StrategyConfig) -> AccountSlot:
    alias = s_cfg.account_alias
    if alias not in self.deploy_cfg.accounts:
        raise ValueError(
            f"strategy '{s_cfg.name}' references account_alias={alias!r} but "
            f"deploy.accounts has only {list(self.deploy_cfg.accounts)}",
        )
    acct = self.deploy_cfg.accounts[alias]
    state_db_path = Path(self.deploy_cfg.state_db_path_for(alias))
    kill_switch_path = Path(self.deploy_cfg.kill_switch_path_for(alias))
    cloid_prefix = f"hla-{alias}-"

    dal = StateDAL(state_db_path)
    dal.run_migrations()

    if self.exec_client_factory is not None:
        exec_client = self.exec_client_factory(alias, acct, s_cfg.paper_mode)
    elif isinstance(acct, HyperliquidAccount):
        exec_client = HLClient(
            account_address=acct.account_address,
            api_secret_key=acct.api_secret_key,
            base_url=acct.base_url,
            paper_mode=s_cfg.paper_mode,
        )
    elif isinstance(acct, PolymarketAccount):
        from .pm_client import PMClient
        exec_client = PMClient(
            paper_mode=s_cfg.paper_mode,
            clob_host=acct.clob_host,
            chain_id=acct.chain_id,
            private_key=acct.private_key,
            clob_api_key=acct.clob_api_key,
            clob_api_secret=acct.clob_api_secret,
            clob_api_passphrase=acct.clob_api_passphrase,
        )
    else:
        raise ValueError(f"unsupported account type: {type(acct).__name__}")

    risk = RiskGate(s_cfg)
    router = Router(
        dal=dal, gate=risk, bus=self.bus, exec_client=exec_client,
        strategy_cfg=s_cfg, strategy_id=s_cfg.name,
        cloid_prefix=cloid_prefix,
    )
    strategy = _build_strategy_for_slot(s_cfg)
    gate_log_path = state_db_path.parent / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=strategy, cfg=s_cfg,
        market_state=self.market_state, dal=dal,
        kill_switch_path=kill_switch_path,
        last_reconcile_ns=0,
        pnl_provider=exec_client.realized_pnl_since,
        gate_log_path=gate_log_path,
    )
    return AccountSlot(
        cfg=s_cfg, account_cfg=acct,
        state_db_path=state_db_path, kill_switch_path=kill_switch_path,
        cloid_prefix=cloid_prefix,
        dal=dal, exec_client=exec_client, risk=risk, router=router,
        strategy=strategy, scanner=scanner,
    )
```

Add the imports at the top of `runtime.py`:

```python
from .config import HyperliquidAccount, PolymarketAccount
```

And update `AccountSlot` (around `runtime.py:169-197`):

```python
@dataclass
class AccountSlot:
    cfg: StrategyConfig
    account_cfg: "HyperliquidAccount | PolymarketAccount"
    state_db_path: Path
    kill_switch_path: Path
    cloid_prefix: str
    dal: StateDAL
    exec_client: ExecutionClient
    risk: RiskGate
    router: Router
    strategy: Strategy
    scanner: Scanner
    blocked: bool = False
    last_reconcile_ns: int = 0
    scans_completed: int = 0
    decisions_emitted: int = 0
    halted: bool = False

    @property
    def alias(self) -> str:
        return self.cfg.account_alias
```

- [ ] **Step 3: Add factory routing in `engine/main.py`**

In `engine/main.py:56-62`, expand subscription routing to send the PM subscriptions to the engine ingestion path (currently only HL):

```python
# Engine consumes HL + PM subscriptions. Binance stays recorder-only.
engine_subs = [s for s in sym_cfg.subscriptions if s.venue in ("hyperliquid", "polymarket")]
```

And the adapter factory needs to dispatch by venue. Simplest correct change: build a CompositeAdapter (or pass both adapters). For now, since each adapter handles only its own subs, register both:

```python
from ..adapters.polymarket import PolymarketAdapter

def _composite_factory():
    # Engine treats one adapter that yields the union of all venues' events.
    # CompositeAdapter is a thin merger; see engine.composite_adapter.
    from ..adapters.composite import CompositeAdapter
    return CompositeAdapter([HyperliquidAdapter(), PolymarketAdapter()])

runtime = EngineRuntime(
    strategies=strategies_cfg.strategies,
    deploy_cfg=deploy_cfg,
    adapter_factory=_composite_factory,
    subscriptions=engine_subs,
)
```

- [ ] **Step 4: Create CompositeAdapter**

```python
# hlanalysis/adapters/composite.py
"""Merges multiple VenueAdapter streams into a single AsyncIterator. Each
adapter receives only its own subs (filtered by `venue`); their event
streams are interleaved fairly via asyncio.Queue."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..config import Subscription
from ..events import Mechanism, NormalizedEvent, ProductType
from .base import VenueAdapter


class CompositeAdapter(VenueAdapter):
    venue = "composite"

    def __init__(self, adapters: list[VenueAdapter]) -> None:
        self._adapters = adapters

    def supports(self, p: ProductType, m: Mechanism) -> bool:
        return any(a.supports(p, m) for a in self._adapters)

    async def stream(
        self, subscriptions: list[Subscription],
    ) -> AsyncIterator[NormalizedEvent]:
        queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue(maxsize=10000)

        async def _drain(adapter: VenueAdapter) -> None:
            subs = [s for s in subscriptions if s.venue == adapter.venue]
            if not subs:
                return
            async for ev in adapter.stream(subs):
                await queue.put(ev)

        tasks = [asyncio.create_task(_drain(a)) for a in self._adapters]
        try:
            while True:
                yield await queue.get()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
```

- [ ] **Step 5: Run unit suite**

```bash
uv run pytest tests/unit -q -x
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add hlanalysis/engine/runtime.py hlanalysis/engine/pm_client.py \
        hlanalysis/engine/main.py hlanalysis/adapters/composite.py
git commit -m "feat(engine): venue-typed slot dispatch + CompositeAdapter; PMClient stub"
```

### Task 5.2: Add v31_pm slot to `config/strategy.yaml` + `deploy.yaml`

**Files:**
- Modify: `config/strategy.yaml` (append the v31_pm block from `docs/v31-polymarket-live-plan.md` Appendix C)
- Modify: `config/deploy.yaml` (add `accounts.v31_pm`)

- [ ] **Step 1: Append the strategy block**

Paste the Appendix C YAML stub from `docs/v31-polymarket-live-plan.md` at the end of `strategies:` in `config/strategy.yaml`. **Set `paper_mode: true`** for now — flips to false after Phase 8.

- [ ] **Step 2: Append the deploy block**

In `config/deploy.yaml` under `accounts:`:

```yaml
    v31_pm:
      venue: polymarket
      clob_host: https://clob.polymarket.com
      chain_id: 137
      private_key: ${PM_PRIVATE_KEY}
      clob_api_key: ${PM_CLOB_API_KEY}
      clob_api_secret: ${PM_CLOB_API_SECRET}
      clob_api_passphrase: ${PM_CLOB_API_PASSPHRASE}
```

- [ ] **Step 3: Smoke-load both configs**

```bash
PM_PRIVATE_KEY=stub PM_CLOB_API_KEY=stub PM_CLOB_API_SECRET=stub PM_CLOB_API_PASSPHRASE=stub \
HL_ACCOUNT_ADDRESS=0x0 HL_API_SECRET_KEY=0x0 HL_ACCOUNT_ADDRESS_V31=0x0 HL_API_SECRET_KEY_V31=0x0 \
TG_BOT_TOKEN=t TG_CHAT_ID=c \
uv run python -c "
from hlanalysis.engine.config import load_deploy_config, load_strategies_config
from pathlib import Path
d = load_deploy_config(Path('config/deploy.yaml'))
s = load_strategies_config(Path('config/strategy.yaml'))
print('accounts:', list(d.accounts))
print('strategies:', [(x.name, x.account_alias) for x in s.strategies])
"
```

Expected: `accounts: ['v1', 'v31', 'v31_pm']` and `strategies` lists 3 entries including the new v31_pm slot.

- [ ] **Step 4: Commit**

```bash
git add config/strategy.yaml config/deploy.yaml
git commit -m "config: add v31_pm slot (paper_mode=true) — strategy + deploy"
```

---

## Phase 6 — PMClient paper-mode

### Task 6.1: PMClient paper place / state / fills

**Files:**
- Modify: `hlanalysis/engine/pm_client.py`
- Create: `tests/unit/test_pm_client_paper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_pm_client_paper.py
import pytest

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


@pytest.fixture
def paper():
    return PMClient(paper_mode=True)


def test_paper_place_marketable_fills(paper):
    req = PlaceRequest(
        cloid="hla-v31_pm-1", symbol="71321...992563",
        side="buy", size=100.0, price=0.92,
        reduce_only=False, time_in_force="ioc",
    )
    ack = paper.place(req)
    assert ack.status == "filled"
    assert ack.fill_price == 0.92 and ack.fill_size == 100.0


def test_paper_place_rejects_nonpositive_price(paper):
    req = PlaceRequest(
        cloid="hla-v31_pm-2", symbol="t", side="buy", size=10, price=0.0,
        reduce_only=False, time_in_force="ioc",
    )
    assert paper.place(req).status == "rejected"


def test_paper_clearinghouse_state_reflects_fills(paper):
    paper.place(PlaceRequest(
        cloid="hla-v31_pm-3", symbol="tok", side="buy", size=100, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    state = paper.clearinghouse_state()
    pos = [p for p in state.positions if p.symbol == "tok"]
    assert pos and pos[0].qty == 100


def test_paper_realized_pnl_zero_on_open(paper):
    paper.place(PlaceRequest(
        cloid="hla-v31_pm-4", symbol="tok", side="buy", size=50, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    assert paper.realized_pnl_since(0) == 0.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_client_paper.py -q
```

- [ ] **Step 3: Implement paper branches in PMClient**

Replace `pm_client.py` with:

```python
"""Polymarket execution client. Implements ExecutionClient.

Paper mode here is the same shape as HLClient.paper — synthesized
OrderAcks, in-memory positions and fills. The strategy doesn't know it's
paper; the engine wiring decides.

Live mode (paper_mode=False) bootstrap and order I/O land in Phase 8 via
py-clob-client-v2.
"""
from __future__ import annotations

import time
import uuid

from .exec_types import (
    ClearinghouseState, OpenOrderRow, OrderAck, PlaceRequest, UserFillRow,
    VenuePosition,
)


class PMClient:
    paper_mode: bool

    def __init__(
        self, *, paper_mode: bool,
        clob_host: str | None = None,
        chain_id: int = 137,
        private_key: str | None = None,
        clob_api_key: str | None = None,
        clob_api_secret: str | None = None,
        clob_api_passphrase: str | None = None,
    ) -> None:
        self.paper_mode = paper_mode
        self._cfg = dict(
            clob_host=clob_host, chain_id=chain_id, private_key=private_key,
            clob_api_key=clob_api_key, clob_api_secret=clob_api_secret,
            clob_api_passphrase=clob_api_passphrase,
        )
        # Paper bookkeeping
        self._paper_acks: dict[str, OrderAck] = {}
        self._paper_open: dict[str, OpenOrderRow] = {}
        self._paper_positions: dict[str, VenuePosition] = {}
        self._paper_fills: list[UserFillRow] = []
        if not paper_mode:
            # Live wiring is added in Phase 8. Construct-and-fail here is
            # intentional so engine integration tests catch a misconfigured
            # slot at startup rather than at first order.
            self._live = None

    def place(self, req: PlaceRequest) -> OrderAck:
        if self.paper_mode:
            return self._paper_place(req)
        return self._live_place(req)

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        if self.paper_mode:
            return self._paper_open.pop(cloid, None) is not None
        return self._live_cancel(cloid=cloid, symbol=symbol)

    def open_orders(self) -> list[OpenOrderRow]:
        if self.paper_mode:
            return list(self._paper_open.values())
        return self._live_open_orders()

    def clearinghouse_state(self) -> ClearinghouseState:
        if self.paper_mode:
            return ClearinghouseState(
                positions=tuple(self._paper_positions.values()),
                account_value_usd=0.0,
            )
        return self._live_clearinghouse_state()

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        if self.paper_mode:
            return [f for f in self._paper_fills if f.ts_ns >= since_ts_ns]
        return self._live_user_fills(since_ts_ns=since_ts_ns)

    def realized_pnl_since(self, since_ts_ns: int) -> float:
        fills = self.user_fills(since_ts_ns=since_ts_ns)
        return sum(f.closed_pnl - f.fee for f in fills)

    # ---- paper internals ----

    def _paper_place(self, req: PlaceRequest) -> OrderAck:
        if req.cloid in self._paper_acks:
            return self._paper_acks[req.cloid]
        if req.price <= 0:
            ack = OrderAck(
                cloid=req.cloid, venue_oid=f"paper-{req.cloid}",
                status="rejected", error="non_marketable_price",
            )
            self._paper_acks[req.cloid] = ack
            return ack
        ack = OrderAck(
            cloid=req.cloid, venue_oid=f"paper-{uuid.uuid4().hex[:16]}",
            status="filled", fill_price=req.price, fill_size=req.size,
        )
        self._paper_acks[req.cloid] = ack
        # Position bookkeeping
        signed = req.size if req.side == "buy" else -req.size
        existing = self._paper_positions.get(req.symbol)
        if existing is None:
            self._paper_positions[req.symbol] = VenuePosition(
                symbol=req.symbol, qty=signed, avg_entry=req.price,
                unrealized_pnl=0.0,
            )
        else:
            tot = existing.qty + signed
            avg = (
                (existing.qty * existing.avg_entry + signed * req.price) / tot
                if abs(tot) > 1e-9 else 0.0
            )
            if abs(tot) < 1e-9:
                self._paper_positions.pop(req.symbol, None)
            else:
                self._paper_positions[req.symbol] = VenuePosition(
                    symbol=req.symbol, qty=tot, avg_entry=avg, unrealized_pnl=0.0,
                )
        ts = time.time_ns()
        self._paper_fills.append(UserFillRow(
            fill_id=f"f-{req.cloid}-{ts}", cloid=req.cloid, symbol=req.symbol,
            side=req.side, price=req.price, size=req.size, fee=0.0, ts_ns=ts,
        ))
        return ack

    # ---- live stubs (filled in Phase 8) ----

    def _live_place(self, req: PlaceRequest) -> OrderAck:
        raise NotImplementedError("Phase 8")

    def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
        raise NotImplementedError("Phase 8")

    def _live_open_orders(self) -> list[OpenOrderRow]:
        raise NotImplementedError("Phase 8")

    def _live_clearinghouse_state(self) -> ClearinghouseState:
        raise NotImplementedError("Phase 8")

    def _live_user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        raise NotImplementedError("Phase 8")
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_client_paper.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/pm_client.py tests/unit/test_pm_client_paper.py
git commit -m "feat(pm): PMClient paper-mode (place/cancel/state/fills/pnl)"
```

### Task 6.2: Engine end-to-end paper test

**Files:**
- Create: `tests/integration/test_engine_pm_paper_loop.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_engine_pm_paper_loop.py
"""Engine with a v31_pm paper slot scans a stubbed market and emits at
least one ENTER decision, all without network. Validates that the
ExecutionClient seam is properly threaded through Router and Scanner.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import yaml

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.engine.config import (
    load_deploy_config, load_strategies_config,
)
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.events import (
    BookSnapshotEvent, Mechanism, ProductType, QuestionMetaEvent, TradeEvent,
)


class _PMStubAdapter(VenueAdapter):
    venue = "polymarket"
    def supports(self, *a, **k): return True
    async def stream(self, _subs) -> AsyncIterator:
        # ... fixture: emit QuestionMetaEvent for a fake BTC daily, then
        # a tight 0.92/0.93 book with the YES leg as favorite, then halt.
        ...  # full body in plan repo


@pytest.mark.asyncio
async def test_pm_paper_slot_emits_decision(tmp_path):
    # Compose minimal configs pointed at tmp_path. Set paper_mode=true on
    # v31_pm. Run runtime for 5 seconds. Assert at least one decision
    # was emitted by the v31_pm slot.
    ...
```

Note: The body is mechanical and follows the existing
`tests/integration/test_engine_paper_loop.py` shape. The implementer should
copy that structure verbatim and adapt the adapter to yield PM events.

- [ ] **Step 2: Run — expect PASS**

```bash
uv run pytest tests/integration/test_engine_pm_paper_loop.py -q
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_engine_pm_paper_loop.py
git commit -m "test(pm): engine paper loop emits a decision on PM stub adapter"
```

---

## Phase 7 — Depth-walk slippage gate + PM fee model

### Task 7.1: Add multi-level book to RiskInputs + MarketState

**Files:**
- Modify: `hlanalysis/engine/market_state.py` (track top-N levels, default N=5)
- Modify: `hlanalysis/engine/risk.py` (RiskInputs + check_pre_trade)
- Modify: `hlanalysis/engine/scanner.py` (wire ask_levels into RiskInputs)
- Create: `tests/unit/test_risk_depth_walk.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_risk_depth_walk.py
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.config import (
    StrategyConfig, AllowlistEntry, GlobalRiskConfig,
)
from hlanalysis.strategy.types import BookState, OrderIntent, QuestionView


def _mk_cfg(*, max_slip_pct: float = 0.01) -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary"}, max_position_usd=200,
        stop_loss_pct=None, tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0, price_extreme_max=1, vol_max=100,
        distance_from_strike_usd_min=0,
    )
    g = GlobalRiskConfig(
        max_total_inventory_usd=1000, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=0, stale_data_halt_seconds=999,
        reconcile_interval_seconds=60,
        max_slippage_pct=max_slip_pct,
    )
    return StrategyConfig(
        name="t", paper_mode=True, allowlist=[entry],
        defaults=entry, global_=g,
    )


def test_depth_walk_vetoes_when_slippage_exceeds_cap():
    cfg = _mk_cfg(max_slip_pct=0.005)  # 0.5% (= ~0.5¢ at 0.95)
    gate = RiskGate(cfg)
    book = BookState(
        bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=10.0,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.93, 10), (0.95, 100), (0.98, 200)),
    )
    intent = OrderIntent(
        question_idx=1, symbol="tok", side="buy", size=150.0,
        limit_price=0.93, cloid="c", time_in_force="ioc",
    )
    inp = RiskInputs(
        question=QuestionView(question_idx=1, yes_symbol="tok", no_symbol="n",
                               strike=0, expiry_ns=10**18, underlying="BTC",
                               klass="priceBinary", period="24h"),
        question_fields={"class": "priceBinary", "underlying": "BTC"},
        reference_price=110_000, book=book, recent_volume_usd=1000,
        positions=[], live_orders_total_notional=0,
        realized_pnl_today=0, kill_switch_active=False,
        last_reconcile_ns=100, now_ns=200,
    )
    v = gate.check_pre_trade(intent, inp)
    assert not v.approved
    assert v.reason == "depth_walk_slip"


def test_depth_walk_approves_when_first_level_covers_size():
    cfg = _mk_cfg(max_slip_pct=0.005)
    gate = RiskGate(cfg)
    book = BookState(
        bid_px=0.92, bid_sz=200, ask_px=0.93, ask_sz=300,
        last_l2_ts_ns=100, last_trade_ts_ns=100,
        ask_levels=((0.93, 300),),
    )
    intent = OrderIntent(
        question_idx=1, symbol="tok", side="buy", size=150,
        limit_price=0.93, cloid="c", time_in_force="ioc",
    )
    inp = RiskInputs(
        question=QuestionView(question_idx=1, yes_symbol="tok", no_symbol="n",
                               strike=0, expiry_ns=10**18, underlying="BTC",
                               klass="priceBinary", period="24h"),
        question_fields={"class": "priceBinary", "underlying": "BTC"},
        reference_price=110_000, book=book, recent_volume_usd=1000,
        positions=[], live_orders_total_notional=0,
        realized_pnl_today=0, kill_switch_active=False,
        last_reconcile_ns=100, now_ns=200,
    )
    assert gate.check_pre_trade(intent, inp).approved
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_risk_depth_walk.py -q
```
Expected: FAIL — `ask_levels` field doesn't exist on `BookState`.

- [ ] **Step 3: Extend types**

In `hlanalysis/strategy/types.py:BookState`, add:

```python
# Top-N L2 levels (price, size) for entry-leg slippage estimation.
# Empty tuple is acceptable; the depth-walk gate degrades to "approve when
# limit_price covers the full size at ask_px" in that case (legacy
# behavior). MarketState fills this from BookSnapshotEvent on PM/HL.
ask_levels: tuple[tuple[float, float], ...] = ()
bid_levels: tuple[tuple[float, float], ...] = ()
```

In `hlanalysis/engine/config.py:GlobalRiskConfig`, add:

```python
# Maximum tolerated realized-fill slippage as a fraction of intent
# limit_price. 0 disables the gate; PM ships ~0.005 (~0.5¢ at 0.95-favorite).
max_slippage_pct: float = 0.0
```

In `hlanalysis/engine/market_state.py:_MutableBook`, add tuple fields and
populate them in the `BookSnapshotEvent` branch:

```python
case BookSnapshotEvent():
    b = self._books.setdefault(ev.symbol, _MutableBook())
    if ev.bid_px:
        b.bid_px, b.bid_sz = ev.bid_px[0], ev.bid_sz[0]
        b.bid_levels = tuple(zip(ev.bid_px, ev.bid_sz))
    if ev.ask_px:
        b.ask_px, b.ask_sz = ev.ask_px[0], ev.ask_sz[0]
        b.ask_levels = tuple(zip(ev.ask_px, ev.ask_sz))
    b.last_l2_ts_ns = max(b.last_l2_ts_ns, ev.exchange_ts or ev.local_recv_ts)
```

In `hlanalysis/engine/risk.py:check_pre_trade`, before the final
`return RiskVerdict(True, "approved")`:

```python
# 13. Depth-walk slippage. Estimate the realized fill price by walking
# the ask ladder for buys (bid ladder for sells); reject when the
# blended fill exceeds the cap.
slip_cap = self.cfg.global_.max_slippage_pct
if slip_cap > 0 and intent.size > 0:
    levels = inp.book.ask_levels if intent.side == "buy" else inp.book.bid_levels
    if levels:
        remaining = intent.size
        cost = 0.0
        for px, sz in levels:
            take = min(remaining, sz)
            cost += take * px
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            return RiskVerdict(False, "depth_walk_no_fill",
                                {"shortfall": f"{remaining:.4f}"})
        avg_px = cost / intent.size
        slip = (avg_px - intent.limit_price) / intent.limit_price
        if intent.side == "sell":
            slip = -slip
        if slip > slip_cap:
            return RiskVerdict(False, "depth_walk_slip",
                                {"avg_px": f"{avg_px:.5f}",
                                 "limit": f"{intent.limit_price:.5f}",
                                 "slip_pct": f"{slip*100:.3f}"})
```

In `hlanalysis/engine/scanner.py`, when building the `BookState` snapshot for
`RiskInputs`, carry `ask_levels` / `bid_levels` through from `MarketState`.

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_risk_depth_walk.py tests/unit/test_risk.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/strategy/types.py hlanalysis/engine/{config,market_state,risk,scanner}.py \
        tests/unit/test_risk_depth_walk.py
git commit -m "feat(risk): depth-walk slippage gate; BookState carries ask/bid_levels"
```

### Task 7.2: Set PM slippage cap in config

**Files:**
- Modify: `config/strategy.yaml` (v31_pm.global)

- [ ] **Step 1: Add `max_slippage_pct: 0.005`** to the v31_pm `global:` block.

- [ ] **Step 2: Verify it loads** (re-run Task 5.2 step 3 smoke).

- [ ] **Step 3: Commit**

```bash
git add config/strategy.yaml
git commit -m "config(v31_pm): max_slippage_pct=0.005 (~0.5¢ at 0.95 favorite)"
```

### Task 7.3: PM fee model in ThetaHarvesterConfig

**Goal:** Strategy honours `fee_model="pm_binary"` at the live entry edge so
the live PnL matches the backtest's PM-fee-curve assumption.

**Files:**
- Modify: `hlanalysis/strategy/theta_harvester.py:ThetaHarvesterConfig` + `_evaluate_entry`
- Modify: `tests/unit/test_strategy_late_resolution.py` (parity tests) — or add `tests/unit/test_theta_harvester.py`

- [ ] **Step 1: Write failing test**

```python
def test_pm_fee_curve_reduces_effective_edge_near_50_50(tmp_path):
    from hlanalysis.strategy.theta_harvester import (
        ThetaHarvesterStrategy, ThetaHarvesterConfig,
    )
    base = dict(
        vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
        vol_clip_min=0.0, vol_clip_max=5.0, edge_buffer=0.0,
        fee_taker=0.0, half_spread_assumption=0.0,
        drift_lookback_seconds=3600, drift_blend=0.0,
        max_position_usd=200.0, favorite_threshold=0.0,
        tte_min_seconds=0, tte_max_seconds=86400,
        stop_loss_pct=None, exit_edge_threshold=0.0,
        take_profit_price=None, time_stop_seconds=0,
    )
    flat = ThetaHarvesterConfig(**base, fee_model="flat")
    pmf = ThetaHarvesterConfig(**base, fee_model="pm_binary", fee_rate=0.07)
    # At p=0.5 the PM fee is largest (0.07*0.5*0.5=0.0175); the flat
    # variant subtracts nothing. So pm_binary's effective edge must be
    # strictly lower at p≈0.5.
    s_flat = ThetaHarvesterStrategy(flat)
    s_pmf = ThetaHarvesterStrategy(pmf)
    # ... (full evaluate() inputs — copy from existing theta tests)
    ...
```

- [ ] **Step 2: Run — expect FAIL** (`fee_model` field missing).

- [ ] **Step 3: Add `fee_model` + `fee_rate` to `ThetaHarvesterConfig`**

In `theta_harvester.py:ThetaHarvesterConfig` add:

```python
# Entry-side fee model:
#   "flat"      — subtract `fee_taker` (legacy, HL).
#   "pm_binary" — Polymarket curve: fee_per_share = fee_rate * p * (1-p).
fee_model: str = "flat"
fee_rate: float = 0.0  # only consumed when fee_model="pm_binary"
```

In `_evaluate_entry`, where the edge is computed (~`theta_harvester.py:485`):

```python
fee_per_share = (
    self.cfg.fee_rate * p_win * (1.0 - p_win)
    if self.cfg.fee_model == "pm_binary"
    else self.cfg.fee_taker
)
edge = p_win - book.ask_px - fee_per_share - self.cfg.half_spread_assumption
```

Wire `fee_model`/`fee_rate` through `engine/config.py:ThetaParams` and
`engine/runtime.py:build_theta_harvester_config`.

- [ ] **Step 4: Set PM config**

In the v31_pm `theta:` block of `config/strategy.yaml`:

```yaml
      fee_model: pm_binary
      fee_rate: 0.07
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit -q -x
git add hlanalysis/strategy/theta_harvester.py hlanalysis/engine/{config,runtime}.py \
        config/strategy.yaml tests/unit/test_strategy_late_resolution.py
git commit -m "feat(strategy): pm_binary fee model; v31_pm uses fee_rate=0.07"
```

---

## Phase 8 — PMClient live (py-clob-client-v2)

### Task 8.1: Add py-clob-client-v2 dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dep**

```toml
[project]
dependencies = [
    # ... existing entries
    "py-clob-client-v2>=1.0.1",
    "httpx[http2]>=0.27",
]
```

- [ ] **Step 2: Sync**

```bash
uv sync
```

- [ ] **Step 3: Commit lockfile**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: py-clob-client-v2 1.0.1 + httpx[http2]"
```

### Task 8.2: Bootstrap script — derive + persist L2 creds

**Files:**
- Create: `scripts/pm_derive_api_key.py`

- [ ] **Step 1: Write the script**

```python
# scripts/pm_derive_api_key.py
"""One-time bootstrap: derive PM L2 API credentials from a Polygon EOA
and print them as deploy.yaml-shaped env-var values.

Usage:
  PM_PRIVATE_KEY=0x... uv run python scripts/pm_derive_api_key.py
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    key = os.environ.get("PM_PRIVATE_KEY")
    if not key:
        sys.exit("set PM_PRIVATE_KEY env var")
    from py_clob_client_v2 import ClobClient

    host = os.environ.get("PM_CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("PM_CHAIN_ID", "137"))
    client = ClobClient(host=host, chain_id=chain_id, key=key)
    creds = client.create_or_derive_api_key()
    print(f"PM_CLOB_API_KEY={creds.api_key}")
    print(f"PM_CLOB_API_SECRET={creds.api_secret}")
    print(f"PM_CLOB_API_PASSPHRASE={creds.api_passphrase}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (operator step, gated behind real key)**

```bash
PM_PRIVATE_KEY=0xYOUR_EOA uv run python scripts/pm_derive_api_key.py
```

Operator copies the three env vars into the systemd unit / deployment
secrets. **This step is offline w.r.t. tests** — no commit needed for the
output.

- [ ] **Step 3: Commit script**

```bash
git add scripts/pm_derive_api_key.py
git commit -m "ops(pm): one-shot script to derive + print PM L2 API credentials"
```

### Task 8.3: PMClient.place / cancel live

**Files:**
- Modify: `hlanalysis/engine/pm_client.py`
- Create: `tests/unit/test_pm_client_live.py`

- [ ] **Step 1: Write failing test using a fake SDK client**

```python
# tests/unit/test_pm_client_live.py
import pytest

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


class _FakeClob:
    def __init__(self):
        self.placed: list[dict] = []

    def create_and_post_order(self, *, order_args, options, order_type):
        self.placed.append({
            "token_id": order_args.token_id, "price": order_args.price,
            "size": order_args.size, "side": str(order_args.side),
            "order_type": str(order_type),
        })
        return {"success": True, "orderID": "0xfakeid", "status": "matched",
                "makingAmount": str(order_args.size),
                "takingAmount": f"{order_args.size * order_args.price}"}


def test_live_place_translates_request_to_FAK_order(monkeypatch):
    fake = _FakeClob()
    c = PMClient(paper_mode=False, clob_host="x", chain_id=137,
                 private_key="0x0", clob_api_key="k", clob_api_secret="s",
                 clob_api_passphrase="p")
    c._sdk = fake  # type: ignore[attr-defined]
    ack = c.place(PlaceRequest(
        cloid="hla-v31_pm-1",
        symbol="71321...992563",
        side="buy", size=100, price=0.92,
        reduce_only=False, time_in_force="ioc",
    ))
    assert fake.placed
    assert fake.placed[0]["order_type"].endswith("FAK")
    assert ack.status == "filled"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/unit/test_pm_client_live.py -q
```

- [ ] **Step 3: Implement live branches**

In `pm_client.py`:

```python
def __init__(self, *, paper_mode, ...):
    # ... existing code ...
    self._sdk = None
    if not paper_mode:
        from py_clob_client_v2 import ApiCreds, ClobClient
        self._sdk = ClobClient(
            host=clob_host, chain_id=chain_id, key=private_key,
            creds=ApiCreds(
                api_key=clob_api_key,
                api_secret=clob_api_secret,
                api_passphrase=clob_api_passphrase,
            ),
        )


def _live_place(self, req: PlaceRequest) -> OrderAck:
    from py_clob_client_v2 import (
        OrderArgs, OrderType, PartialCreateOrderOptions, Side,
    )
    side = Side.BUY if req.side == "buy" else Side.SELL
    order_type = OrderType.FAK if req.time_in_force == "ioc" else OrderType.GTC
    try:
        resp = self._sdk.create_and_post_order(
            order_args=OrderArgs(
                token_id=req.symbol, price=req.price, side=side, size=req.size,
            ),
            options=PartialCreateOrderOptions(tick_size="0.01"),
            order_type=order_type,
        )
    except Exception as e:
        return OrderAck(cloid=req.cloid, venue_oid="", status="rejected",
                        error=str(e)[:200])
    if not resp.get("success"):
        return OrderAck(cloid=req.cloid, venue_oid=str(resp.get("orderID") or ""),
                        status="rejected",
                        error=str(resp.get("errorMsg", "unknown"))[:200])
    making = float(resp.get("makingAmount") or 0)
    taking = float(resp.get("takingAmount") or 0)
    fill_size = making if req.side == "buy" else taking
    fill_price = (taking / making) if (req.side == "buy" and making > 0) else (
        (making / taking) if (req.side == "sell" and taking > 0) else req.price
    )
    status = "filled" if fill_size > 0 else "open"
    return OrderAck(
        cloid=req.cloid, venue_oid=str(resp.get("orderID") or ""),
        status=status, fill_price=fill_price, fill_size=fill_size,
    )


def _live_cancel(self, *, cloid: str, symbol: str) -> bool:
    # PM cancels by orderID, not cloid. We track cloid→orderID in
    # self._cloid_to_oid populated in _live_place. If the order isn't
    # tracked locally (e.g. orphan), we fail-soft to False.
    oid = getattr(self, "_cloid_to_oid", {}).get(cloid)
    if not oid:
        return False
    try:
        return bool(self._sdk.cancel_order(order_id=oid).get("success"))
    except Exception:
        return False
```

Wire the `_cloid_to_oid` map into `_live_place` so cancel can resolve it.

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/unit/test_pm_client_live.py -q
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/engine/pm_client.py tests/unit/test_pm_client_live.py
git commit -m "feat(pm): PMClient live place/cancel via py-clob-client-v2 (FAK for IOC)"
```

### Task 8.4: PMClient.open_orders / clearinghouse_state / user_fills live

**Files:**
- Modify: `hlanalysis/engine/pm_client.py`
- Modify: `tests/unit/test_pm_client_live.py`

- [ ] **Step 1: Tests**

Add cases that stub `self._sdk.get_orders`, `self._sdk.get_balance_allowance`,
`self._sdk.get_trades` and verify the returned `OpenOrderRow` /
`ClearinghouseState` / `UserFillRow` shapes match the engine's expectations.

- [ ] **Step 2: Implement** the three `_live_*` methods, mapping
`py-clob-client-v2` response dicts to the engine types in `exec_types.py`.

- [ ] **Step 3: PASS + commit**

```bash
uv run pytest tests/unit/test_pm_client_live.py -q
git add hlanalysis/engine/pm_client.py tests/unit/test_pm_client_live.py
git commit -m "feat(pm): PMClient live read path (open_orders/state/fills)"
```

---

## Phase 9 — PM settlement detection

### Task 9.1: Settlement poller

**Goal:** The PolymarketAdapter Gamma poll already emits SettlementEvent
when a market resolves (Phase 1.2 / 3.2). This task makes sure the
Reconciler doesn't fire spurious DRIFTs for the "PM position with qty=0"
case that PM produces post-resolution (PM positions don't *vanish*; they
sit at qty=0 until manually redeemed).

**Files:**
- Modify: `hlanalysis/engine/reconcile.py:163-178` (vanished_positions branch)
- Modify: `tests/unit/test_reconcile.py`

- [ ] **Step 1: Test that a zero-qty venue position is treated like vanished**

```python
def test_zero_qty_venue_position_treated_as_vanished_on_pm():
    # ... build local position with qty>0, venue state with qty==0
    # for the same symbol; assert reconcile produces a vanished entry
    # and deletes the local row.
    ...
```

- [ ] **Step 2: Adjust the reconcile loop**

In `reconcile.py:163-178`, the existing branch fires only when `vp is None`.
Add a second arm: when `vp` exists but `abs(vp.qty) < 1e-9`, treat it as
vanished (same as before). This keeps HL behaviour unchanged (HL never
returns zero-qty rows; it omits them) and handles PM's "resolved but
unredeemed" case.

```python
for qidx, lp in local_by_qidx.items():
    vp = venue_by_symbol.get(lp.symbol)
    if vp is None or abs(vp.qty) < 1e-9:
        vanished.append((qidx, lp.symbol, lp))
        self.dal.delete_position(qidx)
        continue
```

- [ ] **Step 3: PASS + commit**

```bash
uv run pytest tests/unit/test_reconcile.py -q
git add hlanalysis/engine/reconcile.py tests/unit/test_reconcile.py
git commit -m "feat(reconcile): treat zero-qty venue position as vanished (PM settle)"
```

---

## Phase 10 — Alerts + end-to-end live smoke

### Task 10.1: OrderUnconfirmed / RedemptionTimeout alerts

**Files:**
- Modify: `hlanalysis/engine/risk_events.py` (add event types)
- Modify: `hlanalysis/alerts/rules.py` (route + format)
- Modify: `tests/unit/test_alerts_rules.py`

- [ ] **Step 1: Add the event types** to `risk_events.py` matching the
existing dataclass shape (frozen, slots, account_alias-tagged).
- [ ] **Step 2: Format messages** in `alerts/rules.py` next to the existing
`OrderRejected` formatter. Telegram template:
`⚠️ ORDER UNCONFIRMED alias={alias} cloid={cloid} age={age_s}s`.
- [ ] **Step 3: Pass tests + commit**.

### Task 10.2: Engine PM smoke — paper end-to-end

**Files:**
- (no code; operational smoke test of `make engine-paper` against PM)

- [ ] **Step 1: Run engine in PM-paper mode against live PM book feed**

```bash
PM_PRIVATE_KEY=stub PM_CLOB_API_KEY=stub PM_CLOB_API_SECRET=stub PM_CLOB_API_PASSPHRASE=stub \
  uv run python -m hlanalysis.engine.main \
    --strategy-config config/strategy.yaml \
    --deploy-config config/deploy.yaml \
    --symbols-config config/symbols.yaml \
    --log-level INFO
```

- [ ] **Step 2: After 30 minutes, verify**

- Heartbeat lines include `alias=v31_pm`.
- At least one `decision` for v31_pm.
- No `RiskVeto` storms; no `OrderRejected` (paper mode shouldn't produce real
  orders).
- Recorder is still capturing PM parquet alongside.

- [ ] **Step 3: Promote to live**

Flip `paper_mode: false` for v31_pm in `config/strategy.yaml`. Set the live
caps to the burn-in values from §5.3 of `docs/v31-polymarket-live-plan.md`
(`max_position_usd: 50`, `daily_loss_cap_usd: 25`,
`max_concurrent_positions: 1`).

- [ ] **Step 4: Commit the burn-in config**

```bash
git add config/strategy.yaml
git commit -m "config(v31_pm): flip to live burn-in (size=50, dl_cap=25, conc=1)"
```

---

## Self-review

- **Spec coverage:**
  - P0 (PolymarketAdapter, recorder, PMConfig+dispatch, PMClient paper) → Phases 3, 4, 5, 6. ✓
  - P1 (pm_binary fee, depth-walk gate, PMClient live, settlement detector, alerts) → Phases 7, 8, 9, 10. ✓
  - Refactor seam → Phase 0. ✓
  - Final-PM params shipped → Phase 5.2 + Phase 7.3. ✓
  - Recorder-first checkpoint → Phase 4.4 explicit. ✓

- **Placeholder scan:** Two tasks (6.2 and 7.1 first test) have body fragments tagged with `...` placeholders. Both reference the existing similar test (`test_engine_paper_loop.py` for 6.2; `test_strategy_late_resolution.py` for 7.3 fee-curve test) as the literal pattern. The implementer should copy that pattern verbatim — these are deliberate "copy from sibling test" rather than open-ended TODOs.

- **Type consistency:**
  - `ExecutionClient` interface (P0) → consumed by Router + Scanner from P0 onward. ✓
  - `AccountConfig` discriminated union → P0 + P5 dispatch + P5.2 deploy.yaml. ✓
  - `BookState.ask_levels` (P7) → MarketState writes, Scanner reads, RiskGate consumes. ✓
  - `OrderType.FAK` (P8.3) → matches `time_in_force="ioc"` mapping in router. ✓

- **Sequence dependency:**
  - P0 must land before any P5+ work (engine wiring depends on the seam). ✓
  - P3 must land before P4 (recorder needs the adapter). ✓
  - P5 PMClient stub must land before P6 paper-mode fills (engine must construct first). ✓
  - P8 must come after P5 (live PMClient slots into the same factory). ✓
  - P7 depth-walk gate is independent of P8 live PMClient — paper-mode validates the gate fires correctly before any real money rides on it. ✓

---

## Execution

Plan saved to `docs/superpowers/plans/2026-05-24-v31-polymarket-live.md`.
