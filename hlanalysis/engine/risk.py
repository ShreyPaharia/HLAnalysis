from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import StrategyConfig, match_question
from ..risk.caps import (
    concurrent_cap_exceeded,
    daily_loss_exceeded,
    inventory_cap_exceeded,
)
from ..strategy.types import BookState, OrderIntent, Position, QuestionView


@dataclass(frozen=True, slots=True)
class RiskInputs:
    question: QuestionView
    question_fields: dict[str, str]      # for allowlist matcher
    reference_price: float
    book: BookState                       # the leg being targeted
    recent_volume_usd: float              # last-hour notional on this question
    positions: list[Position]             # all currently held positions
    live_orders_total_notional: float     # USD across all live (pending/open) orders
    realized_pnl_today: float
    kill_switch_active: bool
    last_reconcile_ns: int
    now_ns: int
    # Age of the latest reference-feed (Binance/HL) price tick at decision time.
    # Defaults to 0 (fresh) so callers that don't track it — the stop-loss
    # enforcer, tests — are unaffected; the Scanner sets it for entries so the
    # gate can veto on a stale reference (SHR-60).
    reference_age_ns: int = 0


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    approved: bool
    reason: str = ""
    detail: dict[str, str] | None = None
    # Depth-walk gate may approve while clamping the intent down to whatever
    # liquidity is available at-or-better-than the limit price. Router applies
    # the clamp before submission; None means "no clamp, use intent.size".
    clamped_size: float | None = None


class RiskGate:
    """Spec §6 — independent of strategy. Final guard on real-money invariants."""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    # ---- pre-trade ----

    def check_pre_trade(self, intent: OrderIntent, inp: RiskInputs) -> RiskVerdict:
        # Normalise: an exit intent (reduce_only) only runs the trivial sanity
        # checks. Daily-loss / inventory caps are entry-only.
        is_exit = intent.reduce_only

        # 12. Settled-market refusal — applies to entries AND exits (the venue
        # has already cash-settled, so don't push orders).
        if inp.question.settled:
            return RiskVerdict(False, "settled")

        # 6. Order size sanity — only hard zero/negative check here; USD cap is
        # enforced by per-position and global checks below.
        if intent.size <= 0:
            return RiskVerdict(False, "size_invalid")

        if is_exit:
            # Exits skip entry gates, but MUST respect the slippage budget so a
            # stop/exit IOC can't walk a thin PM book down (SHR-48).  The depth-walk
            # clamp is reused from the entry path; on a budget breach we clamp size
            # (a partial reduce beats none) rather than veto outright.
            # min_order_notional is deliberately skipped for exits — stopping at
            # whatever partial fill is available is always better than not stopping.
            return self._depth_walk_clamp(intent, inp, approve_reason="approved_exit")

        # 1. Allowlist
        matched = match_question(
            self.cfg, question_idx=inp.question.question_idx, fields=inp.question_fields,
        )
        if matched is None:
            if inp.question.question_idx in self.cfg.blocklist_question_idxs:
                return RiskVerdict(False, "blocklist")
            return RiskVerdict(False, "allowlist_no_match")

        # 2. Per-position cap (using matched-class override)
        notional = intent.size * intent.limit_price
        if notional > matched.max_position_usd:
            return RiskVerdict(False, "max_position_usd",
                                {"notional": f"{notional:.2f}", "cap": f"{matched.max_position_usd}"})

        # 3. Global inventory cap
        held_plus_orders = inp.live_orders_total_notional + sum(
            abs(p.qty) * p.avg_entry for p in inp.positions
        )
        if inventory_cap_exceeded(
            held_plus_orders, notional, self.cfg.global_.max_total_inventory_usd
        ):
            new_total = held_plus_orders + notional
            return RiskVerdict(False, "max_total_inventory",
                                {"new_total": f"{new_total:.2f}"})

        # 4. Concurrent-positions cap
        is_topup = any(p.question_idx == intent.question_idx for p in inp.positions)
        if concurrent_cap_exceeded(
            len(inp.positions), is_topup, self.cfg.global_.max_concurrent_positions
        ):
            return RiskVerdict(False, "max_concurrent_positions")

        # 5. Daily loss cap
        if daily_loss_exceeded(
            inp.realized_pnl_today, self.cfg.global_.daily_loss_cap_usd
        ):
            return RiskVerdict(False, "daily_loss_cap")

        # 7. TTE bounds
        tte_s = (inp.question.expiry_ns - inp.now_ns) / 1e9
        if not (matched.tte_min_seconds <= tte_s <= matched.tte_max_seconds):
            return RiskVerdict(False, "tte_out_of_window", {"tte_s": f"{tte_s:.0f}"})

        # 8. Strike-proximity
        if inp.reference_price > 0:
            distance = abs(inp.question.strike - inp.reference_price) / inp.reference_price
            if distance > (self.cfg.global_.max_strike_distance_pct / 100.0):
                return RiskVerdict(False, "strike_distance",
                                    {"pct": f"{distance*100:.2f}"})

        # 9. Min recent volume
        if inp.recent_volume_usd < self.cfg.global_.min_recent_volume_usd:
            return RiskVerdict(False, "low_volume",
                                {"usd": f"{inp.recent_volume_usd:.0f}"})

        # 10. Engine health
        if inp.kill_switch_active:
            return RiskVerdict(False, "kill_switch_active")
        stale_ns = self.cfg.global_.stale_data_halt_seconds * 1_000_000_000
        if inp.now_ns - inp.book.last_l2_ts_ns > stale_ns:
            return RiskVerdict(False, "stale_data")
        # Reference-feed staleness (SHR-60). The book gate above covers only the
        # HL/PM trading leg, not the Binance/HL reference feed that sets
        # reference_price + σ. A silently dead reference freezes edge computation
        # while the strategy keeps firing; veto entries when it's stale. Exits
        # are short-circuited earlier, so this never blocks a close.
        if inp.reference_age_ns > stale_ns:
            return RiskVerdict(False, "stale_reference")
        # last_reconcile_ns: tolerate 2x the configured interval
        if inp.last_reconcile_ns > 0 and (
            inp.now_ns - inp.last_reconcile_ns
            > 2 * self.cfg.global_.reconcile_interval_seconds * 1_000_000_000
        ):
            return RiskVerdict(False, "stale_reconcile")

        # 11. No conflicting leg
        for p in inp.positions:
            if p.question_idx == intent.question_idx and p.symbol != intent.symbol:
                return RiskVerdict(False, "opposite_leg_held")

        # 13. Depth-walk slippage. Restrict the ladder to levels marketable at
        # our limit (asks ≤ limit for buys, bids ≥ limit for sells) — a CLOB
        # IOC won't lift offers above its limit price, so deeper-and-worse
        # levels were never reachable. If the at-limit ladder can't cover the
        # full size, *clamp* the intent down rather than veto — PM CLOB IOC
        # partial-fills naturally, and the strategy's topup re-fires next tick
        # to close the residual. The slip cap still applies when a strategy
        # explicitly raises the limit to walk multiple price levels. Disabled
        # when cap = 0 (HL default) or when the venue feed doesn't populate
        # book levels.
        clamp_verdict = self._depth_walk_clamp(intent, inp, approve_reason="approved")
        if not clamp_verdict.approved:
            return clamp_verdict
        clamped_size = clamp_verdict.clamped_size

        # Min-order-notional floor. Applied to the *effective* size (clamped
        # if the depth-walk shrank it, otherwise intent.size). Rejects orders
        # too small to be worth submitting — PM CLOB will accept tiny IOCs
        # but each one burns a network round-trip and a cloid for negligible
        # fill notional. HL leaves min_order_notional_usd=0 → no-op.
        min_ntl = self.cfg.global_.min_order_notional_usd
        if min_ntl > 0:
            effective_size = clamped_size if clamped_size is not None else intent.size
            effective_ntl = effective_size * intent.limit_price
            if effective_ntl < min_ntl:
                return RiskVerdict(False, "order_below_min_notional",
                                   {"effective_ntl": f"{effective_ntl:.2f}",
                                    "min_ntl": f"{min_ntl:.2f}"})

        return RiskVerdict(True, "approved", clamped_size=clamped_size)

    # ---- depth-walk helper (shared by entry and exit paths) ----

    def _depth_walk_clamp(
        self, intent: OrderIntent, inp: RiskInputs, *, approve_reason: str,
    ) -> RiskVerdict:
        """Apply the depth-walk slippage clamp to an intent.

        Restricts the book ladder to levels marketable at the intent's limit
        price, then either:
          * returns ``RiskVerdict(True, approve_reason)`` when the full size
            is fillable within the slip cap,
          * returns ``RiskVerdict(True, approve_reason, clamped_size=filled)``
            when only partial depth is available at-limit (caller should use
            the reduced size — a partial fill beats none, and the stop-loss /
            topup loop will re-fire),
          * returns ``RiskVerdict(False, "depth_walk_no_fill")`` when no level
            is at-or-better-than the limit price,
          * returns ``RiskVerdict(True, approve_reason)`` with no clamp when
            the slip cap is 0 (HL default) or no level data is available (BBO
            feeds that don't populate bid/ask_levels).

        Disabled (no-op → approve) when ``max_slippage_pct == 0`` or when the
        relevant side of the book has no level data.
        """
        slip_cap = self.cfg.global_.max_slippage_pct
        if slip_cap <= 0 or intent.size <= 0 or intent.limit_price <= 0:
            return RiskVerdict(True, approve_reason)

        levels = (
            inp.book.ask_levels if intent.side == "buy"
            else inp.book.bid_levels
        )
        if not levels:
            return RiskVerdict(True, approve_reason)

        if intent.side == "buy":
            usable = [(px, sz) for px, sz in levels if px <= intent.limit_price]
        else:
            usable = [(px, sz) for px, sz in levels if px >= intent.limit_price]

        if not usable:
            return RiskVerdict(False, "depth_walk_no_fill",
                               {"shortfall": f"{intent.size:.4f}"})

        remaining = intent.size
        cost = 0.0
        filled = 0.0
        for px, sz in usable:
            take = min(remaining, sz)
            cost += take * px
            filled += take
            remaining -= take
            if remaining <= 1e-9:
                break

        # Guard: every usable level had sz<=0 (e.g. adapter publishes a zero-
        # size level before the real size arrives).  Treat as no fillable
        # liquidity — mirror the empty-book path above.
        if filled <= 0:
            return RiskVerdict(False, "depth_walk_no_fill",
                               {"shortfall": f"{intent.size:.4f}"})

        avg_px = cost / filled
        slip = (avg_px - intent.limit_price) / intent.limit_price
        if intent.side == "sell":
            slip = -slip

        if slip > slip_cap:
            return RiskVerdict(False, "depth_walk_slip",
                               {"avg_px": f"{avg_px:.5f}",
                                "limit": f"{intent.limit_price:.5f}",
                                "slip_pct": f"{slip*100:.3f}"})

        clamped_size = filled if remaining > 1e-9 else None
        return RiskVerdict(True, approve_reason, clamped_size=clamped_size)

    # ---- continuous checks ----

    def breached_stops(
        self, positions: list[Position], books: Mapping[str, BookState],
    ) -> list[Position]:
        """Return positions whose stop-loss has been breached at the current top-of-book."""
        breached: list[Position] = []
        for p in positions:
            b = books.get(p.symbol)
            if b is None:
                continue
            ref = b.bid_px if p.qty > 0 else b.ask_px
            if ref is None:
                continue
            if (p.qty > 0 and ref <= p.stop_loss_price) or (
                p.qty < 0 and ref >= p.stop_loss_price
            ):
                breached.append(p)
        return breached

    def kill_switch_active(self, path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def stale_books(
        self, books: Mapping[str, BookState], *, now_ns: int,
    ) -> list[str]:
        stale_ns = self.cfg.global_.stale_data_halt_seconds * 1_000_000_000
        return [
            sym for sym, b in books.items()
            if b.last_l2_ts_ns > 0 and (now_ns - b.last_l2_ts_ns) > stale_ns
        ]
