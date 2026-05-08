from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .hl_client import ClearinghouseState, OpenOrderRow, UserFillRow
from .reconcile import Reconciler
from .risk_events import ReconcileDrift
from .state import StateDAL


@dataclass(frozen=True, slots=True)
class RestartDriftResult:
    blocked: bool
    drift_events: list[ReconcileDrift]
    summary: str


class RestartDriftGate:
    """Spec §5.5 — runs the three-way merge at startup; writes a block file if
    any drift case fires. If the block file already exists from a prior run,
    we stay blocked regardless of this run's drift state — operator clears it.

    Conservative default: `auto_clear_on_clean=False` so an operator-set
    block file is never auto-removed. The runtime can pass `True` only on
    `--clean-restart` if we ever add such a flag.
    """

    LOUD_CASES = frozenset({"local_ghost", "venue_orphan", "position_mismatch"})

    def __init__(
        self,
        *,
        dal: StateDAL,
        block_path: Path,
        auto_clear_on_clean: bool = False,
    ) -> None:
        self.dal = dal
        self.block_path = block_path
        self.auto_clear = auto_clear_on_clean

    def run(
        self,
        *,
        venue_open: list[OpenOrderRow],
        venue_state: ClearinghouseState,
        fills_lookup: Callable[[str], list[UserFillRow]],
        now_ns: int,
    ) -> RestartDriftResult:
        rec = Reconciler(self.dal, fills_lookup=fills_lookup)
        res = rec.run(venue_open=venue_open, venue_state=venue_state, now_ns=now_ns)

        loud = [d for d in res.drift_events if d.case in self.LOUD_CASES
                # state_mismatch with a fills resolution is a quiet auto-fix.
                or (d.case == "state_mismatch" and (d.detail or {}).get("resolution") != "filled_via_user_fills")]

        existing_block = self.block_path.exists()
        if loud:
            self.block_path.parent.mkdir(parents=True, exist_ok=True)
            summary = "\n".join(
                f"- {d.case} cloid={d.cloid} q={d.question_idx} {d.detail}"
                for d in loud
            )
            self.block_path.write_text(f"restart_blocked at ns={now_ns}\n{summary}\n")
            return RestartDriftResult(blocked=True, drift_events=res.drift_events, summary=summary)

        if existing_block:
            if self.auto_clear:
                self.block_path.unlink(missing_ok=True)
                return RestartDriftResult(
                    blocked=False, drift_events=res.drift_events,
                    summary="block file auto-cleared on clean restart",
                )
            return RestartDriftResult(
                blocked=True, drift_events=res.drift_events,
                summary="block file present (operator hold)",
            )

        return RestartDriftResult(blocked=False, drift_events=res.drift_events, summary="")
