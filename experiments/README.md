# experiments/ — archived research runners

This directory is a **frozen lab notebook**, not maintained tooling. It holds the
one-off experiment scripts and the per-experiment tuning grids that produced the
results written up in `docs/research/`. They are kept for provenance and
reproducibility of past findings — they are **not** part of the live system and
are not expected to run unmodified.

Nothing in here is imported by the `hlanalysis` package, the four CLIs, the
`Makefile`, or the test suite. Moving or deleting these files cannot break the
build.

## Layout

- `scripts/` — experiment runners, almost all thin subprocess wrappers around the
  `hl-bt` backtest CLI (`run_v31_*`, `run_v35/36/37_*`, `run_v1_cadence_*`,
  `run_variable_tte_*`, `run_v31_hl_{binary,bucket}_*`, `run_v31_ablations*`,
  the `*_sweep.sh` / `*_stack*.sh` drivers, `paper_validate_v1_parkinson.py`,
  `run_btc_ref_equivalence.py`, `analyze_sig5m_tune.py`, the one-off
  `gen_review_report.py`, …).
- `config/` — the `tuning.<variant>.yaml` grids those runners swept over
  (`tuning.v1-*`, `tuning.v3-*`, `tuning.v4-*`, `tuning.v5-*`, plus
  `tuning.coarse/permissive/refined`).

## Caveats

- **Several runners are non-runnable as-is.** About ten hardcode absolute paths
  to ephemeral git worktrees that no longer exist (grep for `.worktrees/`), and
  `gen_review_report.py` additionally points at an ephemeral `/tmp` agent-output
  file. They are archived verbatim, not repaired.
- Each runner was written against the `hl-bt` CLI surface at a specific commit;
  flags and strategy registrations have since evolved. Treat them as historical
  reference, and re-derive any number you care about with the current CLI.

## Where the live equivalents live

- Reusable, argument-taking tooling stayed in `scripts/` at the repo root
  (plotting, fixture capture, data fetch, venue probes, report helpers).
- Canonical, active tuning inputs stayed in `config/` (`tuning.yaml`,
  `strategy.yaml`, `symbols.yaml`, `deploy.yaml`, `config/backtest/`, …).
- The write-ups these experiments produced are in `docs/research/`.
