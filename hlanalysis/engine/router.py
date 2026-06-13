from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import replace as _dc_replace
from pathlib import Path

from loguru import logger

from ..marketdata.position_math import PositionState, apply_fill, settle, stop_price
from ..strategy.render import outcome_description, question_description
from ..strategy.types import Action, Decision, OrderIntent, QuestionView
from .config import StrategyConfig, match_question
from .event_bus import EventBus
from .exec_client import ExecutionClient
from .exec_types import PlaceRequest
from .risk import RiskGate, RiskInputs
from .risk_events import Entry, Exit, OrderRejected, ReconcileDrift, RiskVeto
from .state import Fill, OpenOrder, Position, StateDAL
from .trade_journal import HaltSnapshot, TradeJournal

# Parses Polymarket's CTF *balance* shortfall rejection (SHR-109). The CLOB
# rejects a reduce-only sell we can't cover with:
#   "not enough balance / allowance: the balance is not enough ->
#    balance: 7900, order amount: 58120000"
# where both numbers are USDC/CTF 1e6 base units. We deliberately require the
# literal "balance is not enough" branch and balance < order_amount, so an
# ALLOWANCE shortfall ("the allowance is not enough") does NOT match — there we
# still HOLD the shares and must not detrack, only fix the on-chain CTF approval.
_PM_BALANCE_SHORTFALL_RE = re.compile(
    r"balance is not enough.*?balance:\s*(\d+).*?order amount:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)

# How long _close_settled will defer publishing a settlement Exit while the
# PnL source (PM winning leg / HL settlement closedPnl fill) lags the `settled`
# flag. The strategy emits the settlement EXIT the instant `question.settled`
# flips, which can race AHEAD of the venue delivering that source — publishing
# then yields a misleading $0 Telegram alert. We retry next tick until the
# source lands, but cap the wait so a never-arriving source can't wedge the
# position open: after the cap we close with the best-available PnL (the prior
# behaviour). Settled markets resolve their PnL source in seconds-to-minutes,
# so 5 min is generous; the 6h RedemptionTimeout watchdog backstops a stuck
# PM redemption independently. (Incident 2026-06-12: HL legs settling at $0.)
_SETTLEMENT_DEFER_TIMEOUT_NS = 300 * 1_000_000_000


def _pm_settlement_winner_known(qv: QuestionView | None) -> bool:
    """True once the winning leg of a settled PM market is known, so
    ``settlement_pnl_usd`` can compute the real PnL instead of falling back to
    ``prior_realized`` (≈$0). ``settled`` can flip True (meta poll) before the
    SettlementEvent resolves the winner, hence the separate check."""
    return qv is not None and qv.settled and bool(qv.settled_symbols or qv.settled_symbol or qv.settled_side)


def _pm_balance_shortfall(error: str | None) -> bool:
    """True iff a PM CLOB rejection authoritatively reports the on-chain CTF
    *balance* (not the allowance) is short of the order amount — i.e. the venue
    says we no longer hold the shares a reduce-only exit is trying to close."""
    if not error:
        return False
    m = _PM_BALANCE_SHORTFALL_RE.search(error)
    if not m:
        return False
    try:
        balance, order_amount = int(m.group(1)), int(m.group(2))
    except (TypeError, ValueError):
        return False
    return balance < order_amount


class Router:
    """Decision → HL API call. Risk gate is the hard veto.

    Phase 1 single-shot: each intent becomes one place call. We do NOT manage
    queue position, partial fills, or order amends — IOC is fire-and-forget.
    """

    def __init__(
        self,
        *,
        dal: StateDAL,
        gate: RiskGate,
        bus: EventBus,
        exec_client: ExecutionClient,
        strategy_cfg: StrategyConfig,
        strategy_id: str = "late_resolution",
        cloid_prefix: str = "hla-",
        reject_breaker_threshold: int = 5,
        reject_breaker_reset_seconds: float = 300.0,
        reduce_close_atol: float = 1e-9,
        journal: TradeJournal | None = None,
    ) -> None:
        self.dal = dal
        # Durable trade journal (SHR-83). Optional + best-effort: when present,
        # the router records the decision the moment the cloid is final, then
        # the send_ts, the reject/veto reason, and the venue fill. None in tests
        # and any caller that doesn't wire it — behaviour is then unchanged.
        self.journal = journal
        # A reduce that lands within this many shares of flat is treated as a
        # full close, and a reduce-only sell against a holding this small is
        # suppressed (un-sellable dust). Defaults to ~exact (1e-9) for HL, where
        # the venue fills the exact size; PM slots pass DUST_QTY_ABS_TOL because
        # PM market sells floor the share amount to 2dp and strand a sub-0.01
        # residual that would otherwise wedge the position open forever
        # (2026-06-06 v31_pm incident). See marketdata.position_math.
        self._reduce_close_atol = reduce_close_atol
        self.gate = gate
        self.bus = bus
        self.exec_client = exec_client
        self.strategy_cfg = strategy_cfg
        self.strategy_id = strategy_id
        # Per-account prefix used for both DB storage and venue-side identification
        # of orders this Router placed. e.g. 'hla-v1-' or 'hla-v31-'. Default
        # 'hla-' preserves legacy single-account behavior.
        self.cloid_prefix = cloid_prefix
        # account_alias derived from the cloid_prefix. We strip the leading
        # "hla-" and trailing "-" so messages carry the bare alias (e.g. "v1").
        # Legacy single-account default ("hla-") becomes "" — alerts read as
        # cross-slot, matching the pre-multi-account behavior.
        self.account_alias = cloid_prefix.removeprefix("hla-").rstrip("-")
        # Per-question last-exit timestamp (ns). Populated from _book_fill's
        # close branch. Used by the post-exit cooldown guard in handle() to
        # block re-entries on the same question_idx within
        # `cfg.defaults.entry_cooldown_seconds` after a close. Persisted to
        # `<state_db_dir>/exit_cooldowns.json` and reloaded on init so the
        # cooldown survives restarts (deploys, OOM, crashes) — without this,
        # the first scan tick after restart sees an empty map and lets
        # re-entries through within the cooldown window.
        self._last_exit_ts: dict[int, int] = self._load_cooldowns()
        # Per-question timestamp of the first deferred settlement-close attempt
        # (ns). _close_settled defers publishing the settlement Exit while the
        # PnL source lags the `settled` flag; this bounds that wait to
        # _SETTLEMENT_DEFER_TIMEOUT_NS so a never-arriving source can't wedge a
        # settled position open. In-memory only: a restart re-derives it on the
        # next settled scan, and the settled market is closed by reconcile/
        # redemption regardless.
        self._settled_first_seen: dict[int, int] = {}
        # Reject circuit-breaker (incident 2026-06-04). A stale-open position
        # whose exit can never fill (price out of band / shares already gone)
        # re-fires every scan tick — 1,200+ identical rejects in ~1h, flooding
        # Telegram and the PM API. After `reject_breaker_threshold` consecutive
        # rejects for the same (question_idx, side) we stop placing until a
        # fill on that question clears the count. Keyed per (qidx, side) so an
        # unfillable exit doesn't gag unrelated orders; reset on any fill for
        # the question (a fill proves the leg is tradeable again).
        self._reject_breaker_threshold = reject_breaker_threshold
        self._consecutive_rejects: dict[tuple[int, str], int] = {}
        # Time-based auto-reset (SHR-109). A fill is the ONLY other reset, so a
        # leg whose exit can never fill — e.g. a stale orphan the venue no longer
        # holds — would otherwise stay tripped FOREVER, wedged with no recovery.
        # After this many seconds since the last reject for a tripped key we let
        # exactly ONE order through as a re-probe (the count is not cleared, so a
        # fresh reject re-arms the full window → at most one order per window,
        # vs. the pre-fix ~1,200/h flood; a fill clears it via _book_fill). 0
        # disables the auto-reset (legacy permanent-until-fill behaviour).
        self._reject_breaker_reset_seconds = reject_breaker_reset_seconds
        # Wall-clock (ns) of the most recent reject per (qidx, side), driving the
        # auto-reset window above.
        self._last_reject_ts: dict[tuple[int, str], int] = {}

    def _cooldown_path(self) -> Path:
        return Path(self.dal.db_path).parent / "exit_cooldowns.json"

    def _load_cooldowns(self) -> dict[int, int]:
        path = self._cooldown_path()
        try:
            with open(path) as f:
                raw = json.load(f)
            return {int(k): int(v) for k, v in raw.items()}
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("failed to load exit_cooldowns.json ({}); starting fresh", e)
            return {}

    def _save_cooldowns(self) -> None:
        path = self._cooldown_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump({str(k): v for k, v in self._last_exit_ts.items()}, f)
                # Finding #38: flush the userspace buffer then fsync the fd
                # before os.replace so a power-loss between write and rename
                # cannot produce a zero-length (or corrupt) cooldown file. The
                # with-block keeps the fd open until after the flush+fsync so
                # the OS doesn't coalesce the writes with later I/O.
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist exit_cooldowns.json: {}", e)

    def _stamp_cloid(self, intent: OrderIntent) -> OrderIntent:
        """Rewrite a strategy-issued `hla-<uuid>` cloid into `<prefix><uuid_hex>`.

        Strategies don't know which account they're routed to, so they emit a
        neutral `hla-<uuid>` cloid. The router stamps the account prefix here so
        the cloid is account-tagged from the moment it lands in the DB and on
        the venue. Idempotent: if the cloid already starts with self.cloid_prefix
        the intent is returned unchanged.
        """
        if intent.cloid.startswith(self.cloid_prefix):
            return intent
        # Extract a hex tail: strip 'hla-' if present, then collapse remaining
        # hyphens (uuid str form has 4 of them) to land on a contiguous hex run.
        tail = intent.cloid
        if tail.startswith("hla-"):
            tail = tail[len("hla-") :]
        tail = tail.replace("-", "")
        new_cloid = f"{self.cloid_prefix}{tail}"
        return _dc_replace(intent, cloid=new_cloid)

    async def handle(
        self,
        decision: Decision,
        *,
        inputs: RiskInputs,
        now_ns: int,
        recent_returns: tuple[float, ...] = (),
        halt: HaltSnapshot | None = None,
    ) -> None:
        if decision.action is Action.HOLD:
            return
        for intent in decision.intents:
            intent = self._stamp_cloid(intent)
            # SHR-83: journal the decision the instant the cloid is final, so
            # every emitted order has a row regardless of whether it is later
            # vetoed, rejected, or filled. Best-effort, off the hot path.
            self._journal_decision(decision, intent, inputs, now_ns, recent_returns, halt)
            # Post-exit cooldown — applies to ENTRIES only (reduce_only exits
            # are always allowed, otherwise we couldn't close fast on a flip).
            # Catches v1-style churn where the strategy re-enters at the top
            # of the book ~1 second after exiting at the bid, paying the
            # spread on every cycle. Lives in the router because the strategy
            # doesn't see realized fills; the router does, in _book_fill.
            cooldown_s = getattr(self.strategy_cfg.defaults, "entry_cooldown_seconds", 0)
            if cooldown_s > 0 and not intent.reduce_only and decision.action is Action.ENTER:
                last_exit = self._last_exit_ts.get(intent.question_idx, 0)
                elapsed_s = (now_ns - last_exit) / 1e9
                if last_exit > 0 and elapsed_s < cooldown_s:
                    await self.bus.publish(
                        RiskVeto(
                            ts_ns=now_ns,
                            account_alias=self.account_alias,
                            reason="post_exit_cooldown",
                            question_idx=intent.question_idx,
                            detail={
                                "cooldown_s": f"{cooldown_s}",
                                "elapsed_s": f"{elapsed_s:.1f}",
                            },
                        )
                    )
                    logger.info(
                        "post_exit_cooldown veto q={} elapsed_s={:.1f} cooldown_s={}",
                        intent.question_idx,
                        elapsed_s,
                        cooldown_s,
                    )
                    self._journal_reject(intent.cloid, "post_exit_cooldown")
                    continue
            verdict = self.gate.check_pre_trade(intent, inputs)
            if not verdict.approved:
                self._journal_reject(intent.cloid, verdict.reason)
                await self.bus.publish(
                    RiskVeto(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        reason=verdict.reason,
                        question_idx=intent.question_idx,
                        detail=verdict.detail or {},
                    )
                )
                logger.info(
                    "risk veto cloid={} reason={} detail={}",
                    intent.cloid,
                    verdict.reason,
                    verdict.detail,
                )
                continue
            # Depth-walk may approve with a smaller size when at-limit liquidity
            # is below the intended quantity. Resize the intent before placement;
            # the strategy's topup loop closes the residual shortfall next tick.
            if verdict.clamped_size is not None and verdict.clamped_size < intent.size:
                logger.info(
                    "depth_walk_clamp cloid={} intent_size={} clamped_size={} limit={}",
                    intent.cloid,
                    intent.size,
                    verdict.clamped_size,
                    intent.limit_price,
                )
                intent = _dc_replace(intent, size=verdict.clamped_size)
            await self._place(
                intent,
                now_ns=now_ns,
                decision=decision,
                question=inputs.question,
                question_fields=inputs.question_fields,
            )
        # Settlement-driven EXIT with no intents: book the close locally.
        if decision.action is Action.EXIT and not decision.intents:
            await self._close_settled(inputs.question.question_idx, now_ns=now_ns, question=inputs.question)

    # ---- trade journal hooks (SHR-83) ----
    # All best-effort: TradeJournal swallows its own write errors, and these
    # guards keep the journal entirely optional. The journal observes the order
    # lifecycle; it must never gate or delay it.

    def _journal_decision(
        self,
        decision: Decision,
        intent: OrderIntent,
        inputs: RiskInputs,
        now_ns: int,
        recent_returns: tuple[float, ...],
        halt: HaltSnapshot | None,
    ) -> None:
        if self.journal is None:
            return
        self.journal.record_decision(
            cloid=intent.cloid,
            question_idx=intent.question_idx,
            decision_ts_ns=now_ns,
            action=decision.action.value,
            side=intent.side,
            symbol=intent.symbol,
            intended_size=intent.size,
            intended_price=intent.limit_price,
            book=inputs.book,
            reference_price=inputs.reference_price,
            recent_volume_usd=inputs.recent_volume_usd,
            recent_returns=recent_returns,
            diagnostics=decision.diagnostics,
            halt=self._augment_halt(halt, intent, inputs),
        )

    def _augment_halt(
        self,
        halt: HaltSnapshot | None,
        intent: OrderIntent,
        inputs: RiskInputs,
    ) -> HaltSnapshot:
        """Fill in the two halt-state bits only the router knows: whether this
        (question, side) has tripped the reject circuit-breaker, and whether the
        reference feed was stale at decision time (per the slot's
        stale_data_halt_seconds)."""
        base = halt or HaltSnapshot()
        tripped = self._consecutive_rejects.get((intent.question_idx, intent.side), 0) >= self._reject_breaker_threshold
        stale_ns = self.strategy_cfg.global_.stale_data_halt_seconds * 1_000_000_000
        stale = inputs.reference_age_ns > stale_ns if inputs.reference_age_ns else False
        return _dc_replace(base, reject_breaker_tripped=tripped, stale_reference=stale)

    def _journal_send(self, cloid: str, now_ns: int) -> None:
        if self.journal is not None:
            self.journal.record_send(cloid=cloid, send_ts_ns=now_ns)

    def _journal_reject(self, cloid: str, reason: str) -> None:
        if self.journal is not None:
            self.journal.record_reject(cloid=cloid, reject_reason=reason)

    def _journal_fill(self, cloid: str, ts_ns: int, px: float, sz: float) -> None:
        if self.journal is not None:
            self.journal.record_fill(
                cloid=cloid,
                fill_ts_ns=ts_ns,
                fill_px=px,
                fill_sz=sz,
            )

    async def _place(
        self,
        intent: OrderIntent,
        *,
        now_ns: int,
        decision: Decision,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
    ) -> None:
        # 0. Reduce-only clamp (SHR-47). The stop-loss enforcer re-fires a
        # full-size IOC every ~1s; if a prior fill already reduced/closed the
        # position, an unclamped reduce-only order oversells into a NAKED SHORT
        # on the outcome leg. HIP-4 makes this load-bearing: HL strips
        # reduce_only for `#` symbols, so the venue never enforces it and this
        # router-side check is the only guard. Read held qty from the DAL
        # (venue-truth via reconcile); once flat, further stop-loss fires are
        # no-ops, and a reduce-only order can never exceed what we hold.
        if intent.reduce_only:
            pos = self.dal.get_position(intent.question_idx)
            held = pos.qty if pos is not None else 0.0
            reducing = (held > 0 and intent.side == "sell") or (held < 0 and intent.side == "buy")
            if not reducing or abs(held) <= self._reduce_close_atol:
                # `abs(held) <= reduce_close_atol` catches un-sellable PM dust:
                # PM floors a sell to 2dp, so a 0.0079-share residual rounds to
                # 0.00 → guaranteed `invalid maker amount` every scan tick. Don't
                # send it. The dust row is cleared by the reconciler.
                logger.info(
                    "reduce_only suppressed cloid={} qidx={} side={} size={} "
                    "held={} (nothing to reduce / wrong direction / dust)",
                    intent.cloid,
                    intent.question_idx,
                    intent.side,
                    intent.size,
                    held,
                )
                return
            if intent.size > abs(held):
                logger.info(
                    "reduce_only clamp cloid={} qidx={} size={}->{} held={}",
                    intent.cloid,
                    intent.question_idx,
                    intent.size,
                    abs(held),
                    held,
                )
                intent = _dc_replace(intent, size=abs(held))
        # 0.5 Reject circuit-breaker. Once a (question, side) has rejected
        # `threshold` times in a row with no intervening fill, stop placing —
        # the leg is wedged (price out of band, shares gone) and further IOCs
        # just flood. Suppressing here skips both the DB write and the network
        # call. A fill on the question resets the count (see _book_fill).
        breaker_key = (intent.question_idx, intent.side)
        if self._consecutive_rejects.get(breaker_key, 0) >= self._reject_breaker_threshold:
            # Tripped. Allow a single re-probe once the auto-reset window has
            # elapsed since the last reject, so a wedged-but-now-tradeable leg
            # can recover without a fill (and a permanently-dead one floods at
            # most one order per window instead of every tick). The count is
            # NOT cleared here: if the re-probe rejects, _last_reject_ts advances
            # and the next probe waits another full window; a fill clears it.
            last = self._last_reject_ts.get(breaker_key, 0)
            elapsed_s = (now_ns - last) / 1e9 if last else float("inf")
            if self._reject_breaker_reset_seconds > 0 and elapsed_s >= self._reject_breaker_reset_seconds:
                logger.info(
                    "reject breaker re-probe qidx={} side={} after {:.0f}s idle (tripped, window elapsed)",
                    intent.question_idx,
                    intent.side,
                    elapsed_s,
                )
            else:
                self._journal_reject(intent.cloid, "reject_breaker_suppressed")
                return
        # 1. Persist pending row before the network call (spec §5.5 idempotency).
        self.dal.upsert_order(
            OpenOrder(
                cloid=intent.cloid,
                venue_oid=None,
                question_idx=intent.question_idx,
                symbol=intent.symbol,
                side=intent.side,
                price=intent.limit_price,
                size=intent.size,
                status="pending",
                placed_ts_ns=now_ns,
                last_update_ts_ns=now_ns,
                strategy_id=self.strategy_id,
            )
        )
        # SHR-83: stamp the send timestamp on the journal row right before the
        # network call so decision_ts → send_ts measures the in-engine latency.
        self._journal_send(intent.cloid, now_ns)
        # 2. Send. The ExecutionClient is a synchronous, requests-backed SDK
        # call wrapped in tenacity retries; running it inline would park the
        # shared event loop (WS ingest, heartbeat, stop-loss enforcer) for the
        # whole network round-trip. Offload to a worker thread (SHR-41).
        ack = await asyncio.to_thread(
            self.exec_client.place,
            PlaceRequest(
                cloid=intent.cloid,
                symbol=intent.symbol,
                side=intent.side,
                size=intent.size,
                price=intent.limit_price,
                reduce_only=intent.reduce_only,
                time_in_force=intent.time_in_force,
            ),
        )
        # 3. Update DB row from ack.
        self.dal.update_order_status(intent.cloid, status=ack.status, venue_oid=ack.venue_oid, now_ns=now_ns)
        # Surface venue-side rejections. The ack.error is dropped on the floor
        # by upsert_order_status (no error column on the openorder table), so
        # without this log the rejection reason is invisible — symptoms are
        # "orders placed silently never fill", which is what bit us live with
        # the HYPE-short eating all margin and HIP-4 buys getting rejected.
        if ack.status == "rejected":
            self._journal_reject(intent.cloid, ack.error or "rejected")
            n = self._consecutive_rejects.get(breaker_key, 0) + 1
            self._consecutive_rejects[breaker_key] = n
            self._last_reject_ts[breaker_key] = now_ns
            if n == self._reject_breaker_threshold:
                logger.warning(
                    "reject circuit-breaker TRIPPED qidx={} side={} after {} "
                    "consecutive rejects; suppressing further placements until a "
                    "fill clears it or the {:.0f}s auto-reset re-probes (last err: {})",
                    intent.question_idx,
                    intent.side,
                    n,
                    self._reject_breaker_reset_seconds,
                    ack.error or "<no_error_field>",
                )
            # SHR-109: an un-sellable stale orphan. Once we have enough evidence
            # (breaker tripped) AND the venue authoritatively reports the CTF
            # balance is short for THIS reduce-only exit, the shares are gone —
            # retrying can never fill, so detrack the local position to stop the
            # perpetual flood and let the slot resume trading. Scoped to a
            # balance (not allowance) shortfall so we never abandon a position we
            # actually hold; _pm_balance_shortfall only matches PM's error, so HL
            # is unaffected.
            if n >= self._reject_breaker_threshold and intent.reduce_only and _pm_balance_shortfall(ack.error):
                await self._quarantine_unsellable_position(intent, now_ns=now_ns)
            logger.warning(
                "order rejected cloid={} symbol={} side={} size={} price={} err={}",
                intent.cloid,
                intent.symbol,
                intent.side,
                intent.size,
                intent.limit_price,
                ack.error or "<no_error_field>",
            )
            await self.bus.publish(
                OrderRejected(
                    ts_ns=now_ns,
                    account_alias=self.account_alias,
                    cloid=intent.cloid,
                    question_idx=intent.question_idx,
                    symbol=intent.symbol,
                    side=intent.side,
                    size=intent.size,
                    price=intent.limit_price,
                    error=ack.error or "",
                )
            )
        # 4. If filled, update Position + emit Entry/Exit.
        # Use `is not None` rather than truthy: a falsy 0.0 is a malformed ack
        # that should be logged as a problem, not silently treated as 'no fill'
        # which would create an unmanaged live position (DB filled, no Position).
        if ack.status == "filled":
            if ack.fill_size is None or ack.fill_price is None or ack.fill_size <= 0 or ack.fill_price <= 0:
                logger.warning(
                    "filled ack with missing/zero size or price; cloid={} size={} price={}",
                    intent.cloid,
                    ack.fill_size,
                    ack.fill_price,
                )
                return
            # Surface IOC partial fills explicitly. Strategy-side topup
            # (theta_harvester / late_resolution) closes the resulting size gap
            # on subsequent ticks, but the partial itself is invisible without
            # this log — symptoms are "position smaller than intended; no
            # visible reason". Threshold ≥ 1.0 avoids logspam on rounding.
            shortfall = intent.size - ack.fill_size
            if shortfall >= 1.0:
                logger.info(
                    "partial_fill cloid={} intent_size={} fill_size={} shortfall={}",
                    intent.cloid,
                    intent.size,
                    ack.fill_size,
                    shortfall,
                )
            await self._book_fill(
                intent, ack.fill_price, ack.fill_size, now_ns=now_ns, question=question, question_fields=question_fields
            )
            # SHR-83: record the venue fill (ts + px/sz) for latency calibration.
            self._journal_fill(intent.cloid, now_ns, ack.fill_price, ack.fill_size)

    async def _quarantine_unsellable_position(
        self,
        intent: OrderIntent,
        *,
        now_ns: int,
    ) -> None:
        """Detrack a stale local position the venue says we no longer hold
        (SHR-109). The CLOB rejects its reduce-only exit with a CTF *balance*
        shortfall, so the shares are gone (most likely a sell/redeem fill the
        engine missed). Retrying floods forever, so delete the row: the strategy
        stops re-emitting the exit, the breaker state clears, and the slot can
        trade again. We stamp the post-exit cooldown to avoid an instant
        re-entry, and emit a position_mismatch DRIFT so the detrack is auditable.
        No realized PnL is booked here — the closing trade (if any) was never
        ours to observe; the loud alert is the operator's signal to reconcile.
        """
        pos = self.dal.get_position(intent.question_idx)
        if pos is None:
            return
        self.dal.delete_position(intent.question_idx)
        self._last_exit_ts[intent.question_idx] = now_ns
        self._save_cooldowns()
        self._consecutive_rejects = {k: v for k, v in self._consecutive_rejects.items() if k[0] != intent.question_idx}
        self._last_reject_ts = {k: v for k, v in self._last_reject_ts.items() if k[0] != intent.question_idx}
        logger.warning(
            "quarantined un-sellable position qidx={} symbol={} qty={:g} "
            "avg_entry={:g}: venue reports insufficient CTF balance for the "
            "reduce-only exit (stale orphan). Detracked to stop the reject flood.",
            intent.question_idx,
            pos.symbol,
            pos.qty,
            pos.avg_entry,
        )
        await self.bus.publish(
            ReconcileDrift(
                ts_ns=now_ns,
                account_alias=self.account_alias,
                case="position_mismatch",
                question_idx=intent.question_idx,
                detail={
                    "resolution": "quarantined_unsellable_orphan",
                    "symbol": pos.symbol,
                    "db_qty": f"{pos.qty}",
                },
            )
        )

    def prune(self, active_question_idxs: set[int], *, now_ns: int | None = None) -> None:
        """Drop per-question cache entries for question_idxs no longer active,
        and expire elapsed cooldown entries by TTL (finding #38).

        Wiring choice: same as Scanner.prune — an explicit method rather than
        hooking into MarketState's eviction, so all eviction logic stays within
        the owned files (no edits to runtime.py required). The caller passes the
        current active set; Router removes stale entries from _last_exit_ts and
        the reject-breaker dicts (keyed by (qidx, side)).

        TTL expiry (finding #38): entries whose cooldown window has already
        elapsed are removed regardless of whether the question is still active.
        Without this, _last_exit_ts grows unbounded — every closed question
        leaves a permanent entry in memory and on disk. The entry is only useful
        while the cooldown is active; once elapsed it is dead weight. If
        entry_cooldown_seconds is 0 (disabled), no entries are TTL-pruned (the
        gate is already bypassed in handle()). After removing any expired entries
        the pruned map is persisted to disk so a restart doesn't reload them.
        """
        _now_ns = now_ns if now_ns is not None else time.time_ns()
        cooldown_s = getattr(self.strategy_cfg.defaults, "entry_cooldown_seconds", 0) or 0

        # 1. Inactivity eviction (existing behaviour): drop entries for question
        #    idxs that are no longer in the active market set.
        stale_qidxs = {idx for idx in self._last_exit_ts if idx not in active_question_idxs}
        for idx in stale_qidxs:
            self._last_exit_ts.pop(idx, None)

        # 2. TTL expiry (finding #38): remove entries whose cooldown window has
        #    already elapsed, even if the question is still active. This keeps
        #    the map bounded and the on-disk JSON compact.
        if cooldown_s > 0:
            expired_qidxs = [
                idx for idx, last_exit in self._last_exit_ts.items() if (_now_ns - last_exit) / 1e9 >= cooldown_s
            ]
            for idx in expired_qidxs:
                self._last_exit_ts.pop(idx, None)

        # 3. Persist the pruned map so a restart loads the trimmed version.
        #    Only write when something was actually removed to avoid a disk write
        #    on every prune call in the common no-change case.
        if stale_qidxs or (cooldown_s > 0 and expired_qidxs):
            self._save_cooldowns()

        stale_breaker_keys = [k for k in self._consecutive_rejects if k[0] not in active_question_idxs]
        for k in stale_breaker_keys:
            self._consecutive_rejects.pop(k, None)
            self._last_reject_ts.pop(k, None)

    def _stop_loss_price_for(
        self,
        fill_price: float,
        question_idx: int,
        question_fields: dict[str, str] | None,
    ) -> float:
        """Resolve the per-trade stop loss from the matched allowlist entry.

        Returns the price at which the risk gate should trigger a stop-loss
        exit, or the shared STOP_DISABLED_SENTINEL when the entry's
        `stop_loss_pct` is None (or no entry matches — defaults catch this in
        practice).
        """
        matched = None
        if question_fields:
            matched = match_question(
                self.strategy_cfg,
                question_idx=question_idx,
                fields=question_fields,
            )
        entry = matched or self.strategy_cfg.defaults
        return stop_price(fill_price, entry.stop_loss_pct)

    async def _book_fill(
        self,
        intent: OrderIntent,
        price: float,
        size: float,
        *,
        now_ns: int,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
    ) -> None:
        # A fill proves this question's legs are tradeable again — clear any
        # reject-breaker state for it (both sides) so a fresh position can exit.
        self._consecutive_rejects = {k: v for k, v in self._consecutive_rejects.items() if k[0] != intent.question_idx}
        self._last_reject_ts = {k: v for k, v in self._last_reject_ts.items() if k[0] != intent.question_idx}
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, intent.symbol) if question else ""
        existing = self.dal.get_position(intent.question_idx)
        signed = size if intent.side == "buy" else -size
        # Position-write audit (incident 2026-06-04). Log every fill-driven
        # position transition with before/after qty so a stale/re-inflated
        # position can be traced to the exact fill that wrote it — the evidence
        # that was missing when the v31_pm position stuck at 56.1685.
        _qty_before = existing.qty if existing is not None else 0.0
        logger.info(
            "position_write cloid={} qidx={} symbol={} side={} fill_size={:g} "
            "fill_price={:g} qty_before={:g} qty_after={:g}",
            intent.cloid,
            intent.question_idx,
            intent.symbol,
            intent.side,
            size,
            price,
            _qty_before,
            _qty_before + signed,
        )
        # Realized PnL + position update come from the shared, pure
        # ``apply_fill`` (the single source of truth the backtest runner also
        # calls, so live and sim accounting are bit-identical by construction).
        # ``realized_this_fill`` is 0 on opens/add-ons and the closed-lot PnL on
        # reduce/close legs, recorded on the Fill row and the Exit event.
        prev_state = (
            PositionState(
                existing.qty,
                existing.avg_entry,
                existing.realized_pnl,
                existing.closed_qty,
            )
            if existing is not None
            else None
        )
        new_state, realized_this_fill = apply_fill(
            prev_state,
            intent.side,
            size,
            price,
            close_atol=self._reduce_close_atol,
        )
        if existing is None:
            assert new_state is not None
            self.dal.upsert_position(
                Position(
                    question_idx=intent.question_idx,
                    symbol=intent.symbol,
                    qty=new_state.qty,
                    avg_entry=new_state.avg_entry,
                    realized_pnl=0.0,
                    last_update_ts_ns=now_ns,
                    stop_loss_price=self._stop_loss_price_for(
                        price,
                        intent.question_idx,
                        question_fields,
                    ),
                    closed_qty=new_state.closed_qty,
                )
            )
            await self.bus.publish(
                Entry(
                    ts_ns=now_ns,
                    account_alias=self.account_alias,
                    cloid=intent.cloid,
                    question_idx=intent.question_idx,
                    symbol=intent.symbol,
                    side=intent.side,
                    size=size,
                    price=price,
                    question_description=q_desc,
                    outcome_description=o_desc,
                )
            )
        elif new_state is None:
            # Closed. On a full close ``realized_this_fill == (price-avg)*qty``,
            # so the trade's realized PnL is this fill's plus prior partials'.
            self.dal.delete_position(intent.question_idx)
            # Stamp the post-exit cooldown clock. Read in handle() before
            # the next ENTER on the same question_idx is allowed through.
            self._last_exit_ts[intent.question_idx] = now_ns
            self._save_cooldowns()
            # Use the strategy-supplied exit_reason when present so the
            # Telegram alert distinguishes safety_d / edge / time_stop /
            # true stop_loss exits. Legacy fallback: reduce_only without a
            # reason was historically tagged "stop_loss" — preserved for
            # callers that haven't been updated.
            exit_reason = intent.exit_reason or ("stop_loss" if intent.reduce_only else "manual")
            # Report the TOTAL quantity closed over the trade (prior partial
            # reduces + this closing lot), so qty and the cumulative realized_pnl
            # describe the same scope. `existing.qty` alone is just the final lot
            # and pairs misleadingly with the whole-trade PnL when the exit
            # filled across several partial reduces.
            await self.bus.publish(
                Exit(
                    ts_ns=now_ns,
                    account_alias=self.account_alias,
                    question_idx=intent.question_idx,
                    symbol=intent.symbol,
                    qty=existing.closed_qty + existing.qty,
                    realized_pnl=realized_this_fill + existing.realized_pnl,
                    reason=exit_reason,
                    question_description=q_desc,
                    outcome_description=o_desc,
                )
            )
        else:
            # Reduce-only partial or add-on. ``apply_fill`` carried realized PnL
            # forward (add-on weighted-averages the basis; partial reduce leaves
            # it unchanged) so later closes don't lose this partial's PnL. The
            # stop price is preserved from the original entry on add-ons.
            is_addon = (existing.qty > 0 and intent.side == "buy") or (existing.qty < 0 and intent.side == "sell")
            self.dal.upsert_position(
                Position(
                    question_idx=intent.question_idx,
                    symbol=intent.symbol,
                    qty=new_state.qty,
                    avg_entry=new_state.avg_entry,
                    realized_pnl=new_state.realized_pnl,
                    last_update_ts_ns=now_ns,
                    stop_loss_price=existing.stop_loss_price,
                    closed_qty=new_state.closed_qty,
                )
            )
            # Topups / add-on buys reuse the ENTRY Telegram alert so operators
            # see every size-increasing fill, not just the initial open.
            if is_addon:
                await self.bus.publish(
                    Entry(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        cloid=intent.cloid,
                        question_idx=intent.question_idx,
                        symbol=intent.symbol,
                        side=intent.side,
                        size=size,
                        price=price,
                        question_description=q_desc,
                        outcome_description=o_desc,
                    )
                )
        # Persist a Fill row regardless of branch so the local DB has a
        # coherent trade history for diagnostics. fill_id is derived from the
        # cloid + timestamp to be unique per actual venue fill even when one
        # cloid produces multiple partial fills.
        self.dal.append_fill(
            Fill(
                fill_id=f"{intent.cloid}-{now_ns}",
                cloid=intent.cloid,
                question_idx=intent.question_idx,
                symbol=intent.symbol,
                side=intent.side,
                price=price,
                size=size,
                fee=0.0,  # HL fee is reported on user_fills; not surfaced in OrderAck
                ts_ns=now_ns,
                closed_pnl=realized_this_fill,
            )
        )

    async def _close_settled(self, question_idx: int, *, now_ns: int, question: QuestionView | None = None) -> None:
        p = self.dal.get_position(question_idx)
        if p is None:
            return
        is_hl = getattr(self.exec_client, "settlement_reported_as_fill", True)
        # Defer the close until the settlement PnL source has landed. The
        # strategy emits this EXIT the instant `question.settled` flips, which
        # can race AHEAD of the venue delivering the PnL source — the winning
        # leg (PM) or the settlement closedPnl fill (HL). Closing then publishes
        # a $0 settlement Exit (the daily-loss gate self-heals later from venue
        # truth; the Telegram alert does not). So retry next tick until the
        # source is ready, bounded by _SETTLEMENT_DEFER_TIMEOUT_NS so a
        # never-arriving source can't wedge the position open. (Incident
        # 2026-06-12: HL legs settling at $0 because the closedPnl fill lagged
        # the settled flag.)
        if is_hl:
            from .scanner import Scanner

            window_start_ns = Scanner._daily_window_start_ns(
                now_ns,
                hour=self.strategy_cfg.global_.daily_window_start_hour_utc,
            )
            venue_realized = self.exec_client.realized_pnl_for_symbol(
                p.symbol,
                since_ts_ns=window_start_ns,
            )
            # HL binaries pay 0/1 to legs and avg is strictly between, so a held
            # leg's settlement closedPnl is never exactly 0 — venue_realized==0
            # means the settlement fill hasn't been ingested yet.
            ready = venue_realized != 0.0
        else:
            ready = _pm_settlement_winner_known(question)
        # Paper/sim has no real venue lag — the settlement closedPnl fill is
        # never synthesized and the winner is deterministic, so deferring would
        # just stall the loop until the timeout. Only defer in live mode.
        paper = getattr(self.exec_client, "paper_mode", False)
        if not ready and not paper:
            first = self._settled_first_seen.setdefault(question_idx, now_ns)
            if now_ns - first <= _SETTLEMENT_DEFER_TIMEOUT_NS:
                logger.info(
                    "settlement_close_deferred qidx={} symbol={} venue={} — PnL "
                    "source not ready (winner/closedPnl lagging settled flag); "
                    "retry next tick",
                    question_idx,
                    p.symbol,
                    "hl" if is_hl else "pm",
                )
                return
            logger.warning(
                "settlement_close_defer_timeout qidx={} symbol={} venue={} — PnL "
                "source still missing after {}s; closing with best-available PnL",
                question_idx,
                p.symbol,
                "hl" if is_hl else "pm",
                _SETTLEMENT_DEFER_TIMEOUT_NS // 1_000_000_000,
            )
        self._settled_first_seen.pop(question_idx, None)
        # Settlement closes are conceptually exits too — stamp the cooldown so
        # we don't immediately re-enter on a question whose post-settlement
        # rollover gets ingested in the same tick.
        self._last_exit_ts[question_idx] = now_ns
        self._save_cooldowns()
        # Settlement: caller has already confirmed the question settled. We
        # delete the local position; the venue payout is final at 1.0 for
        # the winning leg and 0.0 elsewhere, so the alert's PnL is
        # `qty * (payout - avg_entry) + prior_realized` — see render.
        self.dal.delete_position(question_idx)
        if is_hl:
            # HL: settlement is a venue fill (dir="Settlement", closedPnl set),
            # so source realized PnL from the venue — it matches HL exactly and
            # avoids re-deriving a winning leg, which mislabels multi-leg
            # buckets (a winner booked as a total loss). Do NOT persist:
            # realized_pnl_since already counts this fill, so persisting would
            # double-count it in the daily-loss gate.
            # SHR-88: book the HL settlement through the shared position_math
            # settle() with the venue-truth `closedPnl` override. The payoff
            # indices are unused under the override (HL gives us no clean winner
            # index — that is precisely why we trust the venue's own closedPnl),
            # so settle() returns `venue_realized` verbatim. This is the LIVE
            # override path; the sim runs the same settle() with venue override
            # None (compute path).
            _, realized = settle(
                PositionState(p.qty, p.avg_entry, p.realized_pnl, p.closed_qty),
                position_side_idx=0,
                settled_side_idx=0,
                venue_closed_pnl=venue_realized,
            )
        else:
            # PM: the redeem is not a CLOB fill, so re-derive from the
            # (price-sourced, venue-correct) winner and persist for the
            # daily-loss gate. Idempotent per qidx — if the reconcile
            # vanished-position path already recorded it, this overwrites.
            from ..strategy.render import settlement_pnl_usd

            realized = settlement_pnl_usd(
                question,
                p.symbol,
                p.qty,
                p.avg_entry,
                prior_realized=p.realized_pnl,
            )
            self.dal.record_settlement(
                question_idx=question_idx,
                symbol=p.symbol,
                realized_pnl=realized,
                ts_ns=now_ns,
            )
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, p.symbol) if question else ""
        await self.bus.publish(
            Exit(
                ts_ns=now_ns,
                account_alias=self.account_alias,
                question_idx=question_idx,
                symbol=p.symbol,
                qty=p.closed_qty + p.qty,
                realized_pnl=realized,
                reason="settlement",
                question_description=q_desc,
                outcome_description=o_desc,
            )
        )
