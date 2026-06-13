"""Per-decision sim-vs-live replay (decision-granularity fidelity).

Where SHR-90 reconciles *market-level PnL*, this exercises the finer question:
at each moment LIVE actually decided (a trade-journal row), did the SIM see the
same evaluate() inputs (σ / reference / p_model / edge) and take the same action?
Because sim and live run the *same* unified evaluate() (SHR-97), a decision
divergence must be an INPUT divergence — so the core measures per-field input
skew at matched decision points and flags phantom (one-sided) actions.
"""

from __future__ import annotations

from hlanalysis.parity.decision_replay import (
    LiveDecision,
    SimTick,
    replay,
)

S = 1_000_000_000  # 1s in ns


def _live(qi, ts, action, *, sigma=0.4, ref=60000.0, p_model=0.04, edge=0.02):
    return LiveDecision(
        question_idx=qi,
        ts_ns=ts,
        action=action,
        symbol="#1",
        sigma=sigma,
        reference_price=ref,
        p_model=p_model,
        edge=edge,
    )


def _tick(qi, ts, action, *, sigma=0.4, ref=60000.0, p_model=0.04, edge=0.02):
    return SimTick(
        question_idx=qi,
        ts_ns=ts,
        action=action,
        sigma=sigma,
        reference_price=ref,
        p_model=p_model,
        edge=edge,
    )


# --------------------------------------------------------------------------- #
# Exact agreement → 100% decision-match, zero skew.
def test_exact_match_zero_skew():
    live = [_live(1, 100 * S, "enter")]
    sim = [
        _tick(1, 100 * S - S, "hold"),
        _tick(1, 100 * S, "enter"),  # same action, same ts, same inputs
        _tick(1, 100 * S + S, "hold"),
    ]
    r = replay(live, sim, ts_tol_ns=2 * S)
    assert r.n_live == 1
    assert r.n_live_matched == 1
    assert r.decision_match_rate() == 1.0
    assert r.n_sim_phantom == 0
    # σ field skew is exactly zero across the one matched pair.
    assert r.field_skews["sigma"].n == 1
    assert r.field_skews["sigma"].max_abs == 0.0


# --------------------------------------------------------------------------- #
# Same action, but the sim saw a different σ → matched decision, non-zero skew.
def test_input_skew_on_sigma():
    live = [_live(1, 100 * S, "enter", sigma=0.40)]
    sim = [_tick(1, 100 * S, "enter", sigma=0.50)]
    r = replay(live, sim, ts_tol_ns=S)
    assert r.n_live_matched == 1
    sk = r.field_skews["sigma"]
    assert sk.n == 1
    assert abs(sk.max_abs - 0.10) < 1e-9
    assert abs(sk.median_rel - 0.25) < 1e-9  # |0.40-0.50|/0.40


# --------------------------------------------------------------------------- #
# Live decided, sim never took that action in-window → unmatched live decision.
def test_unmatched_live_decision():
    live = [_live(1, 100 * S, "enter")]
    sim = [_tick(1, 100 * S, "hold")]  # sim only ever holds here
    r = replay(live, sim, ts_tol_ns=S)
    assert r.n_live_matched == 0
    assert r.decision_match_rate() == 0.0
    assert len(r.unmatched_live) == 1
    # the nearest sim tick is still used for the input comparison
    assert r.field_skews["sigma"].n == 1


# --------------------------------------------------------------------------- #
# Sim took an action live never did → phantom (the SHR-91/over-entry signature).
def test_phantom_sim_action():
    live = [_live(1, 100 * S, "enter")]
    sim = [
        _tick(1, 100 * S, "enter"),  # matches the live enter
        _tick(1, 200 * S, "enter"),  # phantom: no live decision near here
    ]
    r = replay(live, sim, ts_tol_ns=S)
    assert r.n_sim_actions == 2
    assert r.n_live_matched == 1
    assert r.n_sim_phantom == 1
    assert r.phantom_sim[0].ts_ns == 200 * S


# --------------------------------------------------------------------------- #
# Matching respects question_idx: same ts, different question ≠ a match.
def test_question_isolation():
    live = [_live(1, 100 * S, "enter")]
    sim = [_tick(2, 100 * S, "enter")]  # right ts, WRONG question
    r = replay(live, sim, ts_tol_ns=S)
    assert r.n_live_matched == 0
    assert r.n_sim_phantom == 1


# --------------------------------------------------------------------------- #
# Action type must match: a live 'exit' is not satisfied by a sim 'enter'.
def test_action_type_must_match():
    live = [_live(1, 100 * S, "exit")]
    sim = [_tick(1, 100 * S, "enter")]
    r = replay(live, sim, ts_tol_ns=S)
    assert r.n_live_matched == 0
    assert r.n_sim_phantom == 1  # the sim enter is itself unmatched


# --------------------------------------------------------------------------- #
def test_to_dict_roundtrip_and_rate():
    live = [_live(1, 100 * S, "enter"), _live(1, 300 * S, "exit")]
    sim = [_tick(1, 100 * S, "enter"), _tick(1, 300 * S, "hold")]
    r = replay(live, sim, ts_tol_ns=S)
    d = r.to_dict()
    assert d["n_live"] == 2
    assert d["n_live_matched"] == 1
    assert abs(d["decision_match_rate"] - 0.5) < 1e-9
    assert "sigma" in d["field_skews"]
