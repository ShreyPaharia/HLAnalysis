"""``hl-bt`` command line entrypoint for the v2 backtester.

Subcommands:

- ``strategies`` — list registered strategy ids.
- ``run``        — execute one strategy config across all questions a data source
                   discovers in [start, end), writing report.md + diagnostics +
                   fills to --out-dir.
- ``fetch``      — populate the data-source cache (polymarket only).
- ``tune``       — walk-forward parallel grid sweep.
- ``trace``      — per-question diagnostic trace plot.
- ``report``     — re-render a tuning report from results.jsonl.

Module layout
-------------
- ``_cli_plumbing.py``  — data-source construction helpers, shared config
                          utilities, and misc small helpers.
- ``_cli_commands.py``  — ``cmd_*`` subcommand handlers.
- ``_cli_args.py``      — argparse construction (``_add_run_config_args`` +
                          per-subcommand parser builders).
- ``cli.py`` (this)     — ``main()`` + re-exports of all public names so every
                          existing ``from hlanalysis.backtest.cli import …``
                          keeps working.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

# Importing the strategy package triggers strategy auto-registration: each
# strategy module's tail calls `@register(...)` at import time, so this
# populates the registry before any `build()` call.
import hlanalysis.strategy  # noqa: F401

# Argparse helpers (tests import _add_run_config_args directly from cli.py).
from ._cli_args import (  # noqa: F401
    _add_run_config_args,
    _build_fetch_parser,
    _build_report_parser,
    _build_run_parser,
    _build_strategies_parser,
    _build_trace_parser,
    _build_tune_parser,
)

# Command handlers (tests + tuning.py import these directly from cli.py).
from ._cli_commands import (  # noqa: F401
    cmd_fetch,
    cmd_report,
    cmd_run,
    cmd_strategies,
    cmd_trace,
    cmd_tune,
)

# ---------------------------------------------------------------------------
# Re-exports — keep all names that tests + internal callers import from cli.py
# ---------------------------------------------------------------------------
# Plumbing helpers (tests import these directly from cli.py).
from ._cli_plumbing import (  # noqa: F401
    _ENV_HL_DATA,
    _ENV_PM_CACHE,
    _ENV_PM_NBA_CACHE,
    _build_hedge_source,
    _build_strategy_for_cli,
    _concat_parquets,
    _derive_reference_warmup_seconds,
    _extract_hedge_config,
    _hedge_data_path_for,
    _hedge_half_spread_for,
    _load_run_params,
    _resolve_reference_warmup_seconds,
    _run_config_from_args,
    _sim_risk_caps_from_args,
    _source_config_from_args,
    _strike_for_data_source,
    _strike_for_synthetic,
    assert_hl_cadence_match,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hl-bt")
    sp = p.add_subparsers(dest="cmd", required=True)

    _build_strategies_parser(sp)
    _build_run_parser(sp)
    _build_fetch_parser(sp)
    _build_tune_parser(sp)
    _build_trace_parser(sp)
    _build_report_parser(sp)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
