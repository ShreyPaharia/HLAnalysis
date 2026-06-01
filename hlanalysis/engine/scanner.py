from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .config import StrategyConfig, match_question
from .market_state import MarketState
from .risk import RiskInputs
from .state import StateDAL
from ..strategy.base import Strategy
from ..strategy.types import (
    Action, BookState, Decision, Position, QuestionView,
)


def _binary_favorite_sym(
    question: QuestionView, books: dict[str, BookState],
) -> str | None:
    """Return the higher-mid leg of a binary question, or None if not binary
    or no leg has a usable quote. Used by the gate-log snapshot to identify
    which leg the strategy was actually reasoning about when the diagnostic
    didn't carry an explicit `chosen_leg` field."""
    if question.klass != "priceBinary":
        return None
    y = books.get(question.yes_symbol)
    n = books.get(question.no_symbol)

    def _mid(b: BookState | None) -> float | None:
        if b is None:
            return None
        if b.bid_px is not None and b.ask_px is not None:
            return (b.bid_px + b.ask_px) / 2.0
        return b.ask_px if b.ask_px is not None else b.bid_px

    ym, nm = _mid(y), _mid(n)
    if ym is None and nm is None:
        return None
    if ym is None:
        return question.no_symbol
    if nm is None:
        return question.yes_symbol
    return question.yes_symbol if ym >= nm else question.no_symbol


@dataclass(frozen=True, slots=True)
class ScannedDecision:
    decision: Decision
    inputs: RiskInputs


class Scanner:
    """Per-tick scanner. Walks all known questions, runs allowlist match,
    builds RiskInputs, calls strategy.evaluate(), returns the decisions.
    The runtime owns the timer; this class is pure synchronous logic.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        cfg: StrategyConfig,
        market_state: MarketState,
        dal: StateDAL,
        kill_switch_path: Path,
        last_reconcile_ns: int,
        reference_symbol: str = "BTC",
        pnl_provider: Callable[[int], float] | None = None,
        gate_log_path: Path | None = None,
    ) -> None:
        self.strategy = strategy
        self.cfg = cfg
        self.ms = market_state
        self.dal = dal
        self.kill_switch_path = kill_switch_path
        self.last_reconcile_ns = last_reconcile_ns
        self.ref_symbol = reference_symbol
        # Source of truth for today's realized PnL. When None, falls back to
        # the local DAL — used in tests / paper-mode where there's no HL
        # connection. Live runtime wires this to HLClient.realized_pnl_since
        # so the daily-loss gate sees venue truth rather than the local DB
        # (which loses information on every position close).
        self._pnl_provider = pnl_provider
        # Structured gate-decision log. When set, the scanner writes a JSONL
        # line every time a (question_idx, decision_action, primary_diag_name)
        # tuple changes from the previous tick — i.e. on STATE TRANSITIONS,
        # not every scan. This gives operators a forensic trail of "which
        # gate was blocking each candidate" without flooding the disk at
        # the 1Hz scan rate. Used for the 48h forward test after introducing
        # the bid-gate / cooldown / near-strike filters.
        self.gate_log_path = gate_log_path
        self._last_logged_state: dict[int, tuple[str, str]] = {}
        # PM up/down questions whose open-strike this slot has already resolved
        # and persisted — short-circuits the per-tick capture/persist work.
        self._pm_strike_seen: set[int] = set()
        # Number of 1m log-returns to pull from MarketState each scan tick.
        # Derived from cfg so the YAML's vol_lookback_seconds actually
        # reflects the σ window the strategy sees. Previously hard-coded to
        # 32, which silently capped late_resolution and theta_harvester at
        # 32 min regardless of vol_lookback_seconds=3600 — short windows
        # made σ jumpy enough to bounce p_model across edge_buffer once
        # per minute, churning enter/exit cycles on thin bucket books.
        self._recent_returns_n = self._required_returns_n(cfg)

    @staticmethod
    def _required_returns_n(cfg: StrategyConfig) -> int:
        """How many 1m log-returns to request per scan tick for this strategy.

        Honors vol_lookback_seconds across cfg.defaults and every allowlist
        entry, plus theta.{vol,drift}_lookback_seconds when the strategy is
        theta_harvester. Floored at 32 (the legacy value) so any downstream
        consumer that assumed ≥32 (e.g. v3.4 LM gate's K-of-N) keeps working.
        """
        secs = cfg.defaults.vol_lookback_seconds
        for entry in cfg.allowlist:
            secs = max(secs, entry.vol_lookback_seconds)
        if cfg.theta is not None:
            secs = max(
                secs,
                cfg.theta.vol_lookback_seconds,
                cfg.theta.drift_lookback_seconds,
            )
        # MarketState resamples the ref feed to vol_sampling_dt_seconds bars, so
        # each sample = dt. Derive the divisor from the same config the bucket
        # period is coupled to — at the default dt=60 this is unchanged, but at
        # sub-minute cadences we must request more bars to cover the lookback
        # (else the σ window silently shrinks vs the backtest). Round up to a
        # whole bar; floor at 32 for the legacy behavior.
        dt = cfg.theta.vol_sampling_dt_seconds if cfg.theta is not None else 60
        bars = (secs + dt - 1) // dt
        return max(32, bars)

    def _resolve_pm_strike(self, q: QuestionView, *, now_ns: int) -> QuestionView:
        """Reload a PM up/down strike that the runtime's async capture path
        (EngineRuntime._maybe_capture_pm_strike) already persisted. The scanner
        is pure/sync, so it only reloads an already-known strike here — capture
        (Binance spot 1m close) happens off the scan path."""
        if q.venue != "polymarket" or q.question_idx in self._pm_strike_seen:
            return q
        if q.strike == q.strike:  # already stamped in-memory
            self._pm_strike_seen.add(q.question_idx)
            return q
        persisted = self.dal.get_pm_strike(q.question_idx)
        if persisted is not None and self.ms.set_question_strike(
            q.question_idx, persisted,
        ):
            self._pm_strike_seen.add(q.question_idx)
            return self.ms.question(q.question_idx) or q
        return q

    def scan(self, *, now_ns: int) -> list[ScannedDecision]:
        out: list[ScannedDecision] = []
        ref = self.ms.last_mark(self.ref_symbol)
        if ref is None:
            return out
        positions_db = self.dal.all_positions()
        positions_by_q = {p.question_idx: p for p in positions_db}
        all_positions_strategy = [
            self._db_pos_to_strategy(p) for p in positions_db
        ]
        live_orders = self.dal.live_orders()
        live_notional = sum(o.price * o.size for o in live_orders)
        # Daily loss: realized PnL since the configured daily window start
        # (default UTC midnight, set to 06:00 UTC to align with HL HIP-4
        # binary settlement). Prefer the injected provider (HL truth) over
        # the local DB; see _pnl_provider doc.
        midnight_ns = self._daily_window_start_ns(
            now_ns, hour=self.cfg.global_.daily_window_start_hour_utc,
        )
        if self._pnl_provider is not None:
            try:
                realized_today = self._pnl_provider(midnight_ns)
            except Exception:
                # If HL is unreachable, fall back to local DB rather than
                # halting the scan. The DAL value is a structural under-estimate
                # of losses today (see state.py), so this is the safer side to
                # err on: continue trading rather than freeze on a network blip.
                realized_today = self.dal.realized_pnl_since(midnight_ns)
        else:
            realized_today = self.dal.realized_pnl_since(midnight_ns)
        kill = self.kill_switch_path.exists()

        for q in self.ms.all_questions():
            # `venue` + `series_slug` let a slot's allowlist scope to one venue
            # (and, for PM, one Gamma series): PM and HL share class/underlying
            # but resolve to different books/token namespaces, so without this
            # a PM slot would match HL questions (and submit HL leg symbols as
            # PM token ids — rejected by the CLOB as "Invalid token id").
            series_slug = dict(q.kv).get("series_slug", "")
            fields = {
                "class": q.klass, "underlying": q.underlying, "period": q.period,
                "venue": q.venue, "series_slug": series_slug,
            }
            if match_question(self.cfg, question_idx=q.question_idx, fields=fields) is None:
                continue
            # PM up/down markets carry no static strike. The runtime's async
            # _maybe_capture_pm_strike fetches the Binance spot 1m close once
            # the reference candle has closed. Here we only reload a strike
            # already persisted by that path (pure sync, no IO). May re-stamp
            # the shared QuestionView, so re-read q before using it.
            q = self._resolve_pm_strike(q, now_ns=now_ns)
            # Multi-outcome support: feed every leg of the question to the
            # strategy so it can decide across all sides (priceBucket has 6 legs;
            # priceBinary has 2). Skip the question if no leg has a book yet.
            leg_syms = q.leg_symbols or (
                (q.yes_symbol, q.no_symbol) if q.yes_symbol else ()
            )
            books: dict[str, BookState] = {}
            for sym in leg_syms:
                b = self.ms.book(sym)
                if b is not None:
                    books[sym] = b
            if not books:
                continue

            db_pos = positions_by_q.get(q.question_idx)
            strat_pos = self._db_pos_to_strategy(db_pos) if db_pos else None
            volume_total = sum(
                self.ms.recent_volume_usd(sym, now=now_ns) for sym in leg_syms
            )
            decision = self.strategy.evaluate(
                question=q,
                books=books,
                reference_price=ref,
                recent_returns=self.ms.recent_returns(
                    self.ref_symbol, n=self._recent_returns_n,
                ),
                # Per-bucket (high, low) bars for range-based σ estimators.
                # Previously NEVER passed, so slots configured
                # vol_estimator=parkinson silently fell back to stdev live (the
                # backtest MarketState supplied this all along). Threading it
                # activates Parkinson on the live path — see MarketState.
                # recent_hl_bars and summeries/engine_bbo_sigma_source_2026_05_31.md.
                recent_hl_bars=self.ms.recent_hl_bars(
                    self.ref_symbol, n=self._recent_returns_n,
                ),
                recent_volume_usd=volume_total,
                position=strat_pos,
                now_ns=now_ns,
            )
            self._maybe_log_gate_transition(
                question=q, decision=decision, books=books, now_ns=now_ns,
                position=db_pos,
            )
            if decision.action is Action.HOLD:
                continue
            for intent in decision.intents:
                target = books.get(intent.symbol)
                if target is None:
                    continue
                inputs = RiskInputs(
                    question=q,
                    question_fields=fields,
                    reference_price=ref,
                    book=target,
                    recent_volume_usd=volume_total,
                    positions=all_positions_strategy,
                    live_orders_total_notional=live_notional,
                    realized_pnl_today=realized_today,
                    kill_switch_active=kill,
                    last_reconcile_ns=self.last_reconcile_ns,
                    now_ns=now_ns,
                )
                out.append(ScannedDecision(
                    decision=Decision(action=decision.action, intents=(intent,),
                                       diagnostics=decision.diagnostics),
                    inputs=inputs,
                ))
            # EXIT with no intents (settlement) — pass through with a synthetic input
            if decision.action is Action.EXIT and not decision.intents:
                # Build a stub inputs (book is required; pick any leg we have)
                stub_book = next(iter(books.values()))
                inputs = RiskInputs(
                    question=q, question_fields=fields, reference_price=ref,
                    book=stub_book, recent_volume_usd=0.0,
                    positions=all_positions_strategy,
                    live_orders_total_notional=live_notional,
                    realized_pnl_today=realized_today, kill_switch_active=kill,
                    last_reconcile_ns=self.last_reconcile_ns, now_ns=now_ns,
                )
                out.append(ScannedDecision(decision=decision, inputs=inputs))
        return out

    def _maybe_log_gate_transition(
        self, *, question: QuestionView, decision: Decision,
        books: dict[str, BookState], now_ns: int,
        position: "Position | None" = None,
    ) -> None:
        """Append a JSONL line when the (action, primary_diag) tuple changes
        for this question_idx vs the last scan tick.

        Steady-state ticks (same blocking reason for hours) produce nothing.
        Transitions (entered/exited/changed gate) produce one line. Disk-cheap
        and forensically useful — operators can answer "why didn't we enter
        on #601 at 04:47?" by tailing this file.
        """
        if self.gate_log_path is None:
            return
        # Reduce diagnostics to a stable identifier. Strategies emit one or
        # more Diagnostic objects per decision; the first one is the
        # human-readable label of why this scan tick ended where it did.
        diag_name = (
            decision.diagnostics[0].message if decision.diagnostics else ""
        )
        key = (decision.action.value, diag_name)
        prev = self._last_logged_state.get(question.question_idx)
        if prev == key:
            return
        self._last_logged_state[question.question_idx] = key
        # Snapshot the book of the leg the strategy actually cares about,
        # in priority order:
        #   1) `chosen_leg=<sym>` in the diagnostic — entry-eval path picks
        #      one of N YES legs for buckets and tags the diag.
        #   2) position.symbol — held-position paths (topup hold, edge-hold,
        #      vol_insufficient_data while a position is open, etc.) don't
        #      emit chosen_leg, but the relevant leg is implicitly the
        #      held one.
        #   3) first leg in `books` — pre-existing fallback for diagnostics
        #      with neither chosen_leg nor an open position (e.g.
        #      tte_out_of_window on a fresh question).
        # Without 2) the bucket snapshot would read an arbitrary long-shot
        # leg of the same question and operators would back-solve wildly
        # wrong spreads. See test_gate_log_snapshot_uses_held_position_book_*.
        chosen_sym: str | None = None
        for d in decision.diagnostics:
            for k, v in d.fields:
                if k == "chosen_leg":
                    chosen_sym = v
                    break
            if chosen_sym is not None:
                break
        if chosen_sym is None and position is not None:
            chosen_sym = position.symbol
        if chosen_sym is None:
            # For binary markets every leg-aware gate (no_favorite,
            # bid_notional_too_thin, edge) reasons about the favourite — the
            # higher-mid leg. Snapshotting the favourite's book makes the
            # row reflect the leg the strategy was actually evaluating,
            # instead of an arbitrary insertion-order pick that may be the
            # underdog and mislead operators (was: PM gate log showed YES
            # quotes 0.06–0.10 while the strategy was failing checks against
            # the NO leg at 0.90+).
            chosen_sym = _binary_favorite_sym(question, books)
        sample_book: BookState | None = (
            books.get(chosen_sym) if chosen_sym else None
        ) or next(iter(books.values()), None)
        bid = sample_book.bid_px if sample_book else None
        ask = sample_book.ask_px if sample_book else None
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        row = {
            "ts_ns": now_ns,
            "question_idx": question.question_idx,
            "klass": question.klass,
            "action": decision.action.value,
            "reason": diag_name,
            "bid_px": bid,
            "ask_px": ask,
            "mid_px": mid,
            "bid_sz": sample_book.bid_sz if sample_book else None,
            "ask_sz": sample_book.ask_sz if sample_book else None,
            "strike": question.strike if question.strike == question.strike else None,
            "diag_fields": [
                {"k": k, "v": v}
                for d in decision.diagnostics
                for (k, v) in d.fields
            ] or None,
        }
        try:
            self.gate_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.gate_log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            # Best-effort logging — never block trading on a filesystem hiccup.
            logger.warning("gate log write failed path={} err={}", self.gate_log_path, e)

    @staticmethod
    def _db_pos_to_strategy(p: object | None) -> Position | None:
        if p is None:
            return None
        return Position(
            question_idx=p.question_idx, symbol=p.symbol, qty=p.qty,
            avg_entry=p.avg_entry, stop_loss_price=p.stop_loss_price,
            last_update_ts_ns=p.last_update_ts_ns,
        )

    @staticmethod
    def _utc_midnight_ns(now_ns: int) -> int:
        """Backwards-compatible alias used by older callers / tests; equivalent
        to `_daily_window_start_ns(now_ns, hour=0)`. New callers should pass an
        explicit hour from GlobalRiskConfig.daily_window_start_hour_utc."""
        return Scanner._daily_window_start_ns(now_ns, hour=0)

    @staticmethod
    def _daily_window_start_ns(now_ns: int, *, hour: int) -> int:
        """Most-recent timestamp at HH:00:00 UTC at-or-before `now_ns`.

        If `now_ns` is already past today's HH:00 UTC, returns today's HH:00.
        Otherwise rolls back to yesterday's HH:00. This is the cutoff the
        daily-loss gate uses to query HL fills with — it must be a stable
        boundary so the same fill never appears in two consecutive windows.
        """
        from datetime import datetime, timezone, timedelta
        dt = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc)
        boundary = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        if dt < boundary:
            boundary = boundary - timedelta(days=1)
        return int(boundary.timestamp() * 1_000_000_000)
