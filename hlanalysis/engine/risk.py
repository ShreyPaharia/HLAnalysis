from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import StrategyConfig, match_question
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


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    approved: bool
    reason: str = ""
    detail: dict[str, str] | None = None


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
            return RiskVerdict(True, "approved_exit")

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
        new_total = inp.live_orders_total_notional + notional + sum(
            abs(p.qty) * p.avg_entry for p in inp.positions
        )
        if new_total > self.cfg.global_.max_total_inventory_usd:
            return RiskVerdict(False, "max_total_inventory",
                                {"new_total": f"{new_total:.2f}"})

        # 4. Concurrent-positions cap
        if len(inp.positions) >= self.cfg.global_.max_concurrent_positions:
            # Allow if the intent targets an existing position (top-up); reject otherwise.
            if not any(p.question_idx == intent.question_idx for p in inp.positions):
                return RiskVerdict(False, "max_concurrent_positions")

        # 5. Daily loss cap
        if inp.realized_pnl_today < -self.cfg.global_.daily_loss_cap_usd:
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

        return RiskVerdict(True, "approved")

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
