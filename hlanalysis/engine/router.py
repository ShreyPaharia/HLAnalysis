from __future__ import annotations

import json
import os
import time
from dataclasses import replace as _dc_replace
from pathlib import Path

from loguru import logger

from .config import StrategyConfig, match_question
from .event_bus import EventBus
from .hl_client import HLClient, PlaceRequest
from .risk import RiskGate, RiskInputs
from .risk_events import Entry, Exit, OrderRejected, RiskVeto
from .state import Fill, OpenOrder, Position, StateDAL
from ..strategy.render import outcome_description, question_description
from ..strategy.types import Action, Decision, OrderIntent, QuestionView


# Mirrors hlanalysis.backtest.runner.hftbt_runner._STOP_DISABLED_SENTINEL — a
# negative price the risk gate's stop-loss check (px <= stop_loss_price for
# longs) can never trigger on, since real bid prices are ≥ 0.
_STOP_DISABLED_SENTINEL = -1.0


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
        hl: HLClient,
        strategy_cfg: StrategyConfig,
        strategy_id: str = "late_resolution",
        cloid_prefix: str = "hla-",
    ) -> None:
        self.dal = dal
        self.gate = gate
        self.bus = bus
        self.hl = hl
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
            tail = tail[len("hla-"):]
        tail = tail.replace("-", "")
        new_cloid = f"{self.cloid_prefix}{tail}"
        return _dc_replace(intent, cloid=new_cloid)

    async def handle(self, decision: Decision, *, inputs: RiskInputs, now_ns: int) -> None:
        if decision.action is Action.HOLD:
            return
        for intent in decision.intents:
            intent = self._stamp_cloid(intent)
            # Post-exit cooldown — applies to ENTRIES only (reduce_only exits
            # are always allowed, otherwise we couldn't close fast on a flip).
            # Catches v1-style churn where the strategy re-enters at the top
            # of the book ~1 second after exiting at the bid, paying the
            # spread on every cycle. Lives in the router because the strategy
            # doesn't see realized fills; the router does, in _book_fill.
            cooldown_s = getattr(self.strategy_cfg.defaults, "entry_cooldown_seconds", 0)
            if (
                cooldown_s > 0
                and not intent.reduce_only
                and decision.action is Action.ENTER
            ):
                last_exit = self._last_exit_ts.get(intent.question_idx, 0)
                elapsed_s = (now_ns - last_exit) / 1e9
                if last_exit > 0 and elapsed_s < cooldown_s:
                    await self.bus.publish(RiskVeto(
                        ts_ns=now_ns, account_alias=self.account_alias,
                        reason="post_exit_cooldown",
                        question_idx=intent.question_idx,
                        detail={
                            "cooldown_s": f"{cooldown_s}",
                            "elapsed_s": f"{elapsed_s:.1f}",
                        },
                    ))
                    logger.info(
                        "post_exit_cooldown veto q={} elapsed_s={:.1f} cooldown_s={}",
                        intent.question_idx, elapsed_s, cooldown_s,
                    )
                    continue
            verdict = self.gate.check_pre_trade(intent, inputs)
            if not verdict.approved:
                await self.bus.publish(RiskVeto(
                    ts_ns=now_ns, account_alias=self.account_alias,
                    reason=verdict.reason,
                    question_idx=intent.question_idx,
                    detail=verdict.detail or {},
                ))
                logger.info(
                    "risk veto cloid={} reason={} detail={}",
                    intent.cloid, verdict.reason, verdict.detail,
                )
                continue
            await self._place(intent, now_ns=now_ns, decision=decision,
                              question=inputs.question,
                              question_fields=inputs.question_fields)
        # Settlement-driven EXIT with no intents: book the close locally.
        if decision.action is Action.EXIT and not decision.intents:
            await self._close_settled(inputs.question.question_idx, now_ns=now_ns, question=inputs.question)

    async def _place(
        self, intent: OrderIntent, *, now_ns: int, decision: Decision,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
    ) -> None:
        # 1. Persist pending row before the network call (spec §5.5 idempotency).
        self.dal.upsert_order(OpenOrder(
            cloid=intent.cloid, venue_oid=None, question_idx=intent.question_idx,
            symbol=intent.symbol, side=intent.side, price=intent.limit_price,
            size=intent.size, status="pending", placed_ts_ns=now_ns,
            last_update_ts_ns=now_ns, strategy_id=self.strategy_id,
        ))
        # 2. Send.
        ack = self.hl.place(PlaceRequest(
            cloid=intent.cloid, symbol=intent.symbol, side=intent.side,
            size=intent.size, price=intent.limit_price,
            reduce_only=intent.reduce_only, time_in_force=intent.time_in_force,
        ))
        # 3. Update DB row from ack.
        self.dal.update_order_status(intent.cloid, status=ack.status,
                                      venue_oid=ack.venue_oid, now_ns=now_ns)
        # Surface venue-side rejections. The ack.error is dropped on the floor
        # by upsert_order_status (no error column on the openorder table), so
        # without this log the rejection reason is invisible — symptoms are
        # "orders placed silently never fill", which is what bit us live with
        # the HYPE-short eating all margin and HIP-4 buys getting rejected.
        if ack.status == "rejected":
            logger.warning(
                "order rejected cloid={} symbol={} side={} size={} price={} err={}",
                intent.cloid, intent.symbol, intent.side, intent.size,
                intent.limit_price, ack.error or "<no_error_field>",
            )
            await self.bus.publish(OrderRejected(
                ts_ns=now_ns, account_alias=self.account_alias,
                cloid=intent.cloid,
                question_idx=intent.question_idx,
                symbol=intent.symbol,
                side=intent.side,
                size=intent.size,
                price=intent.limit_price,
                error=ack.error or "",
            ))
        # 4. If filled, update Position + emit Entry/Exit.
        # Use `is not None` rather than truthy: a falsy 0.0 is a malformed ack
        # that should be logged as a problem, not silently treated as 'no fill'
        # which would create an unmanaged live position (DB filled, no Position).
        if ack.status == "filled":
            if ack.fill_size is None or ack.fill_price is None or ack.fill_size <= 0 or ack.fill_price <= 0:
                logger.warning(
                    "filled ack with missing/zero size or price; cloid={} size={} price={}",
                    intent.cloid, ack.fill_size, ack.fill_price,
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
                    intent.cloid, intent.size, ack.fill_size, shortfall,
                )
            await self._book_fill(intent, ack.fill_price, ack.fill_size,
                                  now_ns=now_ns, question=question,
                                  question_fields=question_fields)

    def _stop_loss_price_for(
        self, fill_price: float, question_idx: int,
        question_fields: dict[str, str] | None,
    ) -> float:
        """Resolve the per-trade stop loss from the matched allowlist entry.

        Returns the price at which the risk gate should trigger a stop-loss
        exit, or _STOP_DISABLED_SENTINEL when the entry's `stop_loss_pct` is
        None (or no entry matches — defaults catch this in practice).
        """
        matched = None
        if question_fields:
            matched = match_question(
                self.strategy_cfg,
                question_idx=question_idx,
                fields=question_fields,
            )
        entry = matched or self.strategy_cfg.defaults
        pct = entry.stop_loss_pct
        if pct is None:
            return _STOP_DISABLED_SENTINEL
        return max(0.0, fill_price * (1.0 - pct / 100.0))

    async def _book_fill(
        self, intent: OrderIntent, price: float, size: float, *, now_ns: int,
        question: QuestionView | None = None,
        question_fields: dict[str, str] | None = None,
    ) -> None:
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, intent.symbol) if question else ""
        existing = self.dal.get_position(intent.question_idx)
        signed = size if intent.side == "buy" else -size
        # Realized PnL on this fill (0 on opens, signed on reduces). Computed
        # here so we can record it on the Fill row and the Exit event with one
        # source of truth. For reduce/close legs, PnL = (close_px − avg_entry) × |closed_qty|
        # with sign matching the direction of the position being reduced.
        realized_this_fill = 0.0
        if existing is not None and (
            (existing.qty > 0 and intent.side == "sell")
            or (existing.qty < 0 and intent.side == "buy")
        ):
            closed_qty = min(size, abs(existing.qty))
            if existing.qty > 0:
                realized_this_fill = (price - existing.avg_entry) * closed_qty
            else:
                realized_this_fill = (existing.avg_entry - price) * closed_qty
        if existing is None:
            self.dal.upsert_position(Position(
                question_idx=intent.question_idx, symbol=intent.symbol,
                qty=signed, avg_entry=price, realized_pnl=0.0,
                last_update_ts_ns=now_ns,
                stop_loss_price=self._stop_loss_price_for(
                    price, intent.question_idx, question_fields,
                ),
            ))
            await self.bus.publish(Entry(
                ts_ns=now_ns, account_alias=self.account_alias,
                cloid=intent.cloid,
                question_idx=intent.question_idx, symbol=intent.symbol,
                side=intent.side, size=size, price=price,
                question_description=q_desc, outcome_description=o_desc,
            ))
        else:
            new_qty = existing.qty + signed
            if abs(new_qty) < 1e-9:
                # Closed
                realized = (price - existing.avg_entry) * existing.qty
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
                exit_reason = intent.exit_reason or (
                    "stop_loss" if intent.reduce_only else "manual"
                )
                await self.bus.publish(Exit(
                    ts_ns=now_ns, account_alias=self.account_alias,
                    question_idx=intent.question_idx,
                    symbol=intent.symbol, qty=existing.qty,
                    realized_pnl=realized + existing.realized_pnl,
                    reason=exit_reason,
                    question_description=q_desc, outcome_description=o_desc,
                ))
            else:
                # Reduce-only partial or add-on. Carry realized PnL forward on
                # the position so later closes don't lose this partial's PnL.
                is_addon = (
                    (existing.qty > 0 and intent.side == "buy")
                    or (existing.qty < 0 and intent.side == "sell")
                )
                avg = (existing.qty * existing.avg_entry + signed * price) / new_qty if new_qty else 0.0
                self.dal.upsert_position(Position(
                    question_idx=intent.question_idx, symbol=intent.symbol,
                    qty=new_qty, avg_entry=avg,
                    realized_pnl=existing.realized_pnl + realized_this_fill,
                    last_update_ts_ns=now_ns, stop_loss_price=existing.stop_loss_price,
                ))
                # Topups / add-on buys reuse the ENTRY Telegram alert so operators
                # see every size-increasing fill, not just the initial open.
                if is_addon:
                    await self.bus.publish(Entry(
                        ts_ns=now_ns, account_alias=self.account_alias,
                        cloid=intent.cloid,
                        question_idx=intent.question_idx, symbol=intent.symbol,
                        side=intent.side, size=size, price=price,
                        question_description=q_desc, outcome_description=o_desc,
                    ))
        # Persist a Fill row regardless of branch so the local DB has a
        # coherent trade history for diagnostics. fill_id is derived from the
        # cloid + timestamp to be unique per actual venue fill even when one
        # cloid produces multiple partial fills.
        self.dal.append_fill(Fill(
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
        ))

    async def _close_settled(self, question_idx: int, *, now_ns: int, question: QuestionView | None = None) -> None:
        p = self.dal.get_position(question_idx)
        if p is None:
            return
        # Settlement closes are conceptually exits too — stamp the cooldown so
        # we don't immediately re-enter on a question whose post-settlement
        # rollover gets ingested in the same tick.
        self._last_exit_ts[question_idx] = now_ns
        self._save_cooldowns()
        # Settlement: caller has already confirmed the question settled. We
        # simply delete the local position; venue truth handles PnL.
        self.dal.delete_position(question_idx)
        q_desc = question_description(question) if question else ""
        o_desc = outcome_description(question, p.symbol) if question else ""
        await self.bus.publish(Exit(
            ts_ns=now_ns, account_alias=self.account_alias,
            question_idx=question_idx, symbol=p.symbol,
            qty=p.qty, realized_pnl=p.realized_pnl, reason="settlement",
            question_description=q_desc, outcome_description=o_desc,
        ))
