from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .config import StrategyConfig, match_question
from .config_builders import reference_sampling_dt_seconds
from .market_state import MarketState
from .risk import RiskInputs
from .state import StateDAL
from ..marketdata.decision_kernel import build_decision_inputs
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
    # recent_returns window the strategy evaluated on, carried through to the
    # trade journal (SHR-83) so the live decision's σ inputs are recorded
    # alongside the book-at-decision. Empty for callers that don't supply it.
    recent_returns: tuple[float, ...] = ()


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
        # P4: memoized allowlist-match verdict per question_idx. The match fields
        # (class/underlying/period/venue/series_slug) are immutable per question
        # and the allowlist/blocklist are static for this slot, so the verdict is
        # stable — compute it once, then skip non-tradeable questions with a dict
        # lookup instead of a full match_question() on every scan tick. This keeps
        # the per-tick cost proportional to the questions this slot actually
        # trades, not the whole ingested registry (which holds many series no slot
        # trades). Decisions are bit-identical to the un-memoized path.
        self._tradeable_cache: dict[int, bool] = {}
        # Number of 1m log-returns to pull from MarketState each scan tick.
        # Derived from cfg so the YAML's vol_lookback_seconds actually
        # reflects the σ window the strategy sees. Previously hard-coded to
        # 32, which silently capped late_resolution and theta_harvester at
        # 32 min regardless of vol_lookback_seconds=3600 — short windows
        # made σ jumpy enough to bounce p_model across edge_buffer once
        # per minute, churning enter/exit cycles on thin bucket books.
        self._recent_returns_n = self._required_returns_n(cfg)
        # Per-class cadence overrides (dt, n) for question classes whose theta
        # override changes vol_sampling_dt_seconds. Classes absent here use the
        # default dt-less read below — bit-identical to pre-refactor behaviour.
        self._cadence_by_class = self.cadence_by_class(cfg)
        # Default σ/drift lookback (seconds) for the non-class-override path.
        # Passed as ``lookback_seconds`` to recent_returns / recent_hl_bars so
        # the live engine time-bounds its window to match the backtest's
        # slice_window semantics (SHR-66). Derived from the same inputs as
        # _required_returns_n so the two stay consistent.
        self._default_lookback_secs: int = self._lookback_secs(cfg)
        # Per-class lookback (seconds) for classes in _cadence_by_class.
        self._lookback_secs_by_class: dict[str, int] = self._class_lookback_secs(cfg)
        # R9: the slot's own default vol_sampling_dt_seconds (seconds). Passed
        # EXPLICITLY to build_decision_inputs in the default (no per-class
        # override) path instead of dt=None. Using None is correct only when
        # the slot is the sole registrant of its symbol — it resolves to
        # cadences[0], which is the first-registered cadence. When two sibling
        # slots share a symbol with DIFFERENT default dts, cadences[0] belongs
        # to whichever slot registered first, so the second slot's dt=None read
        # would alias the wrong (symbol, dt) buffer. Passing the explicit dt
        # fixes this while staying bit-identical for single-slot setups.
        # Use the canonical single-source-of-truth derivation: theta carries
        # vol_sampling_dt_seconds in the `theta:` block; late_resolution carries
        # it on its allowlist/defaults (AllowlistEntry.vol_sampling_dt_seconds).
        # A bare `else 60` would wrongly force late_resolution slots (e.g. v1 at
        # dt=5) to read a dt=60 buffer that was never registered/fed.
        self._default_dt_seconds: int = reference_sampling_dt_seconds(cfg)

    @staticmethod
    def _bars_for(secs: int, dt: int) -> int:
        """Number of dt-spaced bars covering ``secs`` of lookback, floored at 32
        (legacy) so downstream consumers assuming ≥32 (v3.4 LM K-of-N) keep working."""
        return max(32, (secs + dt - 1) // dt)

    @staticmethod
    def _lookback_secs(cfg: StrategyConfig) -> int:
        """Largest σ/drift lookback (seconds) across defaults, allowlist, and the
        theta block — identical inputs to the legacy _required_returns_n.

        Deliberately EXCLUDES theta_overrides (so the default-path `n` stays
        bit-identical). cadence_by_class folds in each class's own override
        lookback on top of this base. The "full" lookback that also spans
        overrides — used to size MarketState history — is
        runtime.reference_vol_lookback_seconds; use that, not this, if you need
        the per-class-inclusive maximum.
        """
        secs = cfg.defaults.vol_lookback_seconds
        for entry in cfg.allowlist:
            secs = max(secs, entry.vol_lookback_seconds)
        if cfg.theta is not None:
            secs = max(secs, cfg.theta.vol_lookback_seconds, cfg.theta.drift_lookback_seconds)
        return secs

    @staticmethod
    def _required_returns_n(cfg: StrategyConfig) -> int:
        """Default bars/tick for the dt-less read. UNCHANGED output: dt divisor is
        theta.vol_sampling_dt_seconds for theta slots, else 60 (legacy). The
        MarketState default series is the slot's first registered cadence, which
        equals this dt for theta slots — so the dt-less read stays bit-identical.

        Honors vol_lookback_seconds across cfg.defaults and every allowlist
        entry, plus theta.{vol,drift}_lookback_seconds when the strategy is
        theta_harvester. Floored at 32 (the legacy value) so any downstream
        consumer that assumed ≥32 (e.g. v3.4 LM gate's K-of-N) keeps working.
        MarketState resamples the ref feed to vol_sampling_dt_seconds bars, so
        each sample = dt. At sub-minute cadences we must request more bars to
        cover the lookback (else the σ window silently shrinks vs the backtest).
        Round up to a whole bar; floor at 32 for the legacy behavior.
        """
        dt = cfg.theta.vol_sampling_dt_seconds if cfg.theta is not None else 60
        return Scanner._bars_for(Scanner._lookback_secs(cfg), dt)

    @staticmethod
    def _class_lookback_secs(cfg: StrategyConfig) -> dict[str, int]:
        """Per-class lookback (seconds) for classes that have a dt override.

        Mirrors the ``secs`` logic inside ``cadence_by_class`` but returns the
        lookback directly (not packed into a tuple) so the scanner can pass it
        to ``recent_returns(lookback_seconds=...)`` without recomputing it.
        """
        out: dict[str, int] = {}
        base_secs = Scanner._lookback_secs(cfg)
        for klass, override in (cfg.theta_overrides or {}).items():
            set_fields = override.model_fields_set
            if "vol_sampling_dt_seconds" not in set_fields:
                continue
            secs = base_secs
            if "vol_lookback_seconds" in set_fields:
                secs = max(secs, override.vol_lookback_seconds)
            if "drift_lookback_seconds" in set_fields:
                secs = max(secs, override.drift_lookback_seconds)
            out[klass] = secs
        return out

    @staticmethod
    def cadence_by_class(cfg: StrategyConfig) -> dict[str, tuple[int, int]]:
        """Map question.klass -> (dt_seconds, n_bars) ONLY for classes whose
        theta_override explicitly sets vol_sampling_dt_seconds (model_fields_set).
        Classes absent here read the default series via the dt-less path, so a
        slot with no dt override is bit-identical to today. Empty for non-theta
        slots or slots with no dt override.

        Live: the guard in build_theta_harvester_configs_by_class that previously
        rejected a per-class vol_sampling_dt_seconds was removed in the (symbol,
        dt) refactor, and the runtime registers each override cadence on the
        shared MarketState (see _register_reference_cadences), so the override
        branch is reachable on a real config.
        """
        out: dict[str, tuple[int, int]] = {}
        base_secs = Scanner._lookback_secs(cfg)
        for klass, override in (cfg.theta_overrides or {}).items():
            set_fields = override.model_fields_set
            if "vol_sampling_dt_seconds" not in set_fields:
                continue
            dt = override.vol_sampling_dt_seconds
            # A class override may also widen its own σ/drift window; size n for
            # the larger of base and the explicitly-set per-class lookbacks so the
            # window isn't truncated. (Over-sizing is harmless — the strategy
            # re-slices to its own vol_lookback; under-sizing truncates σ.)
            secs = base_secs
            if "vol_lookback_seconds" in set_fields:
                secs = max(secs, override.vol_lookback_seconds)
            if "drift_lookback_seconds" in set_fields:
                secs = max(secs, override.drift_lookback_seconds)
            out[klass] = (dt, Scanner._bars_for(secs, dt))
        return out

    def _resolve_pm_strike(self, q: QuestionView) -> QuestionView:
        """Reload a PM up/down strike that the runtime's async capture path
        (EngineRuntime._maybe_capture_pm_strike) already persisted. The scanner
        is pure/sync, so it only reloads an already-known strike here — capture
        (Binance spot 1m close) happens off the scan path."""
        if q.venue != "polymarket" or q.question_idx in self._pm_strike_seen:
            return q
        # Only PM up/down (priceBinary) markets have an async-captured strike.
        # priceBucket prices each leg off `priceThresholds` and never resolves a
        # single strike, so mark it seen to skip the per-tick get_pm_strike read.
        if q.klass != "priceBinary":
            self._pm_strike_seen.add(q.question_idx)
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

    def scan(
        self, *, now_ns: int, realized_pnl_today: float | None = None,
    ) -> list[ScannedDecision]:
        out: list[ScannedDecision] = []
        ref = self.ms.last_mark(self.ref_symbol)
        if ref is None:
            return out
        # Age of the reference feed at decision time, for the gate's
        # stale-reference check (SHR-60). ref is not None here, so a ts exists.
        ref_ts = self.ms.last_mark_ts(self.ref_symbol)
        reference_age_ns = (now_ns - ref_ts) if ref_ts is not None else 0
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
        if realized_pnl_today is not None:
            # SHR-41: the live runtime pre-fetches realized PnL off the event
            # loop and injects it here so this scan tick never blocks the loop
            # on a REST round-trip. Falls through to the provider/DAL paths for
            # backtests and unit tests that don't inject.
            realized_today = realized_pnl_today
        elif self._pnl_provider is not None:
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
            qidx = q.question_idx
            # P4: skip questions already known non-tradeable for this slot with a
            # single dict lookup — no fields build, no match_question call.
            cached = self._tradeable_cache.get(qidx)
            if cached is False:
                continue
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
            if cached is None:
                # First sight of this qidx — compute and memoize the verdict.
                cached = match_question(
                    self.cfg, question_idx=qidx, fields=fields,
                ) is not None
                self._tradeable_cache[qidx] = cached
                if not cached:
                    continue
            # PM up/down markets carry no static strike. The runtime's async
            # _maybe_capture_pm_strike fetches the Binance spot 1m close once
            # the reference candle has closed. Here we only reload a strike
            # already persisted by that path (pure sync, no IO). May re-stamp
            # the shared QuestionView, so re-read q before using it.
            q = self._resolve_pm_strike(q)
            # No strike → cannot price the market. PM up/down (priceBinary)
            # strikes are captured asynchronously (runtime spot 1m-close); until
            # captured the strike is NaN. Skip rather than evaluate — a NaN
            # strike makes safety_d NaN, which the strategy maps to None and
            # would otherwise silently bypass the safety_d gate. "No strike →
            # no trade." This applies ONLY to priceBinary: PM multi-strike
            # priceBucket questions have NO single strike (q.strike is ALWAYS
            # NaN — every leg is priced off `priceThresholds` via
            # winning_region, not question.strike), so a blanket NaN skip would
            # silently swallow every bucket leg and the bucket-only slot would
            # emit zero decisions (v31_pm_eth_ms incident 2026-06-12).
            if (
                q.venue == "polymarket"
                and q.klass == "priceBinary"
                and q.strike != q.strike  # NaN check
            ):
                continue
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
            # Resolve the (dt, n) for σ/OHLC history reads per question class.
            # Classes with a theta_override that sets vol_sampling_dt_seconds read
            # the matching (symbol, dt) bar series; all others use the default
            # dt-less read — byte-identical to pre-refactor behaviour.
            # Per-bucket (high, low) bars are also resolved at the same cadence for
            # range-based σ estimators (Parkinson). Previously NEVER passed, so
            # slots configured vol_estimator=parkinson silently fell back to stdev
            # live (the backtest MarketState supplied this all along). Threading it
            # activates Parkinson on the live path — see MarketState.recent_hl_bars
            # and summeries/engine_bbo_sigma_source_2026_05_31.md.
            cadence = self._cadence_by_class.get(q.klass)
            if cadence is None:
                # Default path: pass the slot's own dt explicitly (R9).
                # Previously used dt=None, which resolves to cadences[0] —
                # the first-registered cadence on the symbol. That is correct
                # only when this slot is the sole registrant of its symbol.
                # When two sibling slots share a symbol with different default
                # dts, dt=None aliases the wrong buffer. Passing the slot's own
                # dt explicitly is bit-identical for single-slot setups (the
                # MarketState resolves explicit dt=N to the same buffer that
                # cadences[0]=N resolves to when there is only one registrant).
                _rets_arr, _hl_arr = build_decision_inputs(
                    self.ms._core,
                    ref_symbol=self.ref_symbol,
                    now_ns=now_ns,
                    lookback_seconds=self._default_lookback_secs,
                    dt=self._default_dt_seconds,
                )
            else:
                dt_s, _ret_n = cadence
                lookback_s = self._lookback_secs_by_class.get(q.klass, self._default_lookback_secs)
                _rets_arr, _hl_arr = build_decision_inputs(
                    self.ms._core,
                    ref_symbol=self.ref_symbol,
                    now_ns=now_ns,
                    lookback_seconds=lookback_s,
                    dt=dt_s,
                )
            # Engine contract: convert numpy arrays to tuples (Strategy.evaluate
            # signature declares tuple[float,...] / tuple[tuple[float,float],...]).
            # The engine's MarketState.recent_returns / recent_hl_bars previously
            # performed this conversion internally; we replicate it here so the
            # call to strategy.evaluate is byte-identical. The backtest historically
            # passed numpy arrays (duck-typed silence); new code follows the contract.
            recent_returns: tuple[float, ...] = tuple(_rets_arr.tolist())
            recent_hl_bars: tuple[tuple[float, float], ...] = tuple(
                (float(h), float(lo)) for h, lo in _hl_arr
            )
            decision = self.strategy.evaluate(
                question=q,
                books=books,
                reference_price=ref,
                recent_returns=recent_returns,
                recent_hl_bars=recent_hl_bars,
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
                    reference_age_ns=reference_age_ns,
                )
                out.append(ScannedDecision(
                    decision=Decision(action=decision.action, intents=(intent,),
                                       diagnostics=decision.diagnostics),
                    inputs=inputs,
                    recent_returns=tuple(recent_returns),
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
                out.append(ScannedDecision(
                    decision=decision, inputs=inputs,
                    recent_returns=tuple(recent_returns),
                ))
        return out

    def prune(self, active_question_idxs: set[int]) -> None:
        """Drop per-question cache entries for question_idxs no longer active.

        Wiring choice: Scanner exposes an explicit ``prune(active_set)`` method
        rather than hooking into eviction directly because ``evict_settled_questions``
        lives on MarketState (not Scanner), and we cannot edit runtime.py.  The
        caller (an integration harness, a test, or a future scan-loop self-prune)
        passes the current active set obtained from ``ms.all_questions()``.  This
        keeps Scanner's eviction logic entirely within Scanner's owned file — the
        runtime can call ``scanner.prune({q.question_idx for q in ms.all_questions()})``
        after each eviction round without Scanner knowing about MarketState.
        """
        stale = {idx for idx in self._tradeable_cache if idx not in active_question_idxs}
        for idx in stale:
            self._tradeable_cache.pop(idx, None)
            self._pm_strike_seen.discard(idx)
            self._last_logged_state.pop(idx, None)

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

        Delegates to the canonical implementation in ``hlanalysis.risk.caps``
        so the engine, the sim, and the risk gate all share one copy.
        """
        from hlanalysis.risk.caps import daily_window_start_ns
        return daily_window_start_ns(now_ns, hour=hour)
