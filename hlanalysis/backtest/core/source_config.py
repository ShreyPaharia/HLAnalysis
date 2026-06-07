"""Picklable description of how to build a backtest ``DataSource``.

This replaces the HLBT_* environment side-channel that used to carry
source-construction parameters to spawn workers. The parent builds a
``SourceConfig`` from the CLI args, and that *same* object is carried in the
work tuple to each worker, which calls :meth:`SourceConfig.build` directly.

Single construction path for both the in-process and subprocess runs — so a
worker can no longer silently revert ``book_source`` to ``synthetic`` or
``reference_resample_seconds`` to ``60`` while the parent used the configured
values (the documented "worker factory config drop" regression).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_source import DataSource


# Polymarket series flavors: which PM series + reference asset to load. The
# values are spread straight into ``PolymarketDataSource(**...)``.
PM_FLAVORS: dict[str, dict[str, str]] = {
    "btc_updown": {
        "reference_symbol": "BTC",
        "series_slug": "btc-up-or-down-daily",
        "klines_subdir": "btc_klines",
        "klines_1s_subdir": "btc_klines_1s",
    },
    "wti_updown": {
        "reference_symbol": "WTI",
        "series_slug": "oil-daily-up-or-down",
        "klines_subdir": "wti_klines",
        # WTI uses Pyth klines — no Binance 1s reference; omit klines_1s_subdir
        # so the ctor default applies (btc_klines_1s, which is also unreachable
        # for WTI since reference_source="klines_1s" is not meaningful here).
    },
    "eth_updown": {
        "reference_symbol": "ETH",
        "series_slug": "eth-up-or-down-daily",
        "klines_subdir": "eth_klines",
        "klines_1s_subdir": "eth_klines_1s",
    },
    "btc_multistrike": {
        "reference_symbol": "BTC",
        "bucket_series_slug": "btc-multi-strikes-weekly",
        "klines_subdir": "btc_klines",
        "klines_1s_subdir": "btc_klines_1s",
    },
    "eth_multistrike": {
        "reference_symbol": "ETH",
        "bucket_series_slug": "ethereum-multi-strikes-weekly",
        "klines_subdir": "eth_klines",
        "klines_1s_subdir": "eth_klines_1s",
    },
}


@dataclass(frozen=True)
class SourceConfig:
    """All source-construction parameters needed to (re)build a ``DataSource``.

    Frozen + plain fields → trivially picklable, so it crosses the spawn
    process boundary in the work tuple. The CLI resolves ``cache_root`` (with
    its env default) once in the parent and bakes the concrete value here; the
    worker reads nothing from the environment.
    """

    kind: str
    cache_root: str | None = None
    # HL HIP-4 reference-price venue.
    hl_ref_source: str = "hl_perp"
    # Polymarket knobs.
    pm_flavor: str = "btc_updown"
    pm_reference_source: str = "klines"
    pm_book_source: str = "synthetic"
    pm_binance_bbo_product_type: str = "perp"
    pm_liquidity_profile_path: str | None = None
    # Shared reference-resample cadence (HL + PM). The CLI couples this to the
    # strategy's ``vol_sampling_dt_seconds``; ``tune`` overrides it per grid cell
    # via :meth:`with_reference_resample` because dt is a sweepable param.
    reference_resample_seconds: int = 60

    def with_reference_resample(self, seconds: int) -> "SourceConfig":
        """Return a copy with ``reference_resample_seconds`` replaced (per-cell)."""
        return replace(self, reference_resample_seconds=int(seconds))

    def build(self) -> "DataSource":
        """Construct the concrete ``DataSource``. Called in the parent and in
        every worker — the single construction path for the backtest."""
        if self.kind == "synthetic":
            from ..data.synthetic import (
                SyntheticDataSource,
                make_default_binary_question,
            )

            ds = SyntheticDataSource()
            ds.add_question(make_default_binary_question())
            return ds
        if self.kind == "polymarket":
            from ..data.polymarket import PolymarketDataSource

            if self.pm_flavor not in PM_FLAVORS:
                raise SystemExit(
                    f"Unknown --pm-flavor: {self.pm_flavor!r}. "
                    f"Choices: {sorted(PM_FLAVORS)}"
                )
            return PolymarketDataSource(
                cache_root=Path(self.cache_root or "data/sim"),
                reference_source=self.pm_reference_source,  # type: ignore[arg-type]
                reference_resample_seconds=self.reference_resample_seconds,
                book_source=self.pm_book_source,  # type: ignore[arg-type]
                binance_bbo_product_type=self.pm_binance_bbo_product_type,  # type: ignore[arg-type]
                liquidity_profile_path=self.pm_liquidity_profile_path,
                **PM_FLAVORS[self.pm_flavor],
            )
        if self.kind == "hl_hip4":
            from ..data.hl_hip4 import HLHip4DataSource

            return HLHip4DataSource(
                data_root=Path(self.cache_root or "data"),
                ref_source=self.hl_ref_source,  # type: ignore[arg-type]
                reference_resample_seconds=self.reference_resample_seconds,
            )
        if self.kind == "pm_nba":
            from ..data.pm_nba import PolymarketNBADataSource

            return PolymarketNBADataSource(
                cache_root=Path(self.cache_root or "data/sim/pm_nba")
            )
        raise SystemExit(f"Unknown --data-source: {self.kind}")


__all__ = ["SourceConfig", "PM_FLAVORS"]
