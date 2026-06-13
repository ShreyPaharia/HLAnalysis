"""Tests for hlanalysis.research.outcome_markets.

Tests using real data are skipped when data is absent (CI-safe).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Real data is at HLBT_HL_DATA_ROOT or ../../data relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
_DATA_ROOT = Path(os.environ.get("HLBT_HL_DATA_ROOT", str(_DEFAULT_DATA_ROOT))).resolve()
_DATA_AVAILABLE = (_DATA_ROOT / "venue=hyperliquid").exists()

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=f"Data root not found at {_DATA_ROOT} — skipping real-data tests",
)


@pytest.fixture(scope="module")
def con():
    """Shared DuckDB connection for this module."""
    from hlanalysis.analysis.helpers import duck

    return duck()


@pytest.fixture(scope="module")
def data_root() -> str:
    return str(_DATA_ROOT)


# ---------------------------------------------------------------------------
# Pure unit tests (no data required)
# ---------------------------------------------------------------------------


class TestBandIndexToRange:
    """Pure-function tests for band_index_to_range — no data needed."""

    # Override module-level skip for this class
    pytestmark = pytest.mark.skipif(False, reason="pure unit test")

    def test_band0_returns_none_lo(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_to_range

        lo_b, hi_b = band_index_to_range(0, lo=75000.0, hi=80000.0)
        assert lo_b is None
        assert hi_b == 75000.0

    def test_band1_returns_lo_hi(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_to_range

        lo_b, hi_b = band_index_to_range(1, lo=75000.0, hi=80000.0)
        assert lo_b == 75000.0
        assert hi_b == 80000.0

    def test_band2_returns_hi_none(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_to_range

        lo_b, hi_b = band_index_to_range(2, lo=75000.0, hi=80000.0)
        assert lo_b == 80000.0
        assert hi_b is None

    def test_invalid_band_raises(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_to_range

        with pytest.raises(ValueError):
            band_index_to_range(3, lo=75000.0, hi=80000.0)

    def test_invalid_n_bands_raises(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_to_range

        with pytest.raises(NotImplementedError):
            band_index_to_range(0, lo=75000.0, hi=80000.0, n_bands=4)


class TestBandIndexWins:
    """Pure-function tests for band_index_wins boundary convention."""

    pytestmark = pytest.mark.skipif(False, reason="pure unit test")

    def test_band0_at_boundary(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        # oracle_px == lo: band 0 wins (oracle_px <= lo)
        assert band_index_wins(0, oracle_px=75000.0, lo=75000.0, hi=80000.0) is True

    def test_band0_above_lo_false(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        assert band_index_wins(0, oracle_px=75001.0, lo=75000.0, hi=80000.0) is False

    def test_band1_strictly_above_lo(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        # lo < oracle_px <= hi
        assert band_index_wins(1, oracle_px=75001.0, lo=75000.0, hi=80000.0) is True

    def test_band1_at_lo_false(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        # oracle_px == lo is NOT in band 1 (that's band 0)
        assert band_index_wins(1, oracle_px=75000.0, lo=75000.0, hi=80000.0) is False

    def test_band1_at_hi_true(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        # oracle_px == hi IS in band 1 (lo < oracle_px <= hi)
        assert band_index_wins(1, oracle_px=80000.0, lo=75000.0, hi=80000.0) is True

    def test_band2_strictly_above_hi(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        assert band_index_wins(2, oracle_px=80001.0, lo=75000.0, hi=80000.0) is True

    def test_band2_at_hi_false(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        # oracle_px == hi is in band 1, NOT band 2
        assert band_index_wins(2, oracle_px=80000.0, lo=75000.0, hi=80000.0) is False

    def test_exactly_one_band_wins(self) -> None:
        """For any oracle_px, exactly one of band 0, 1, 2 should win."""
        from hlanalysis.research.outcome_markets import band_index_wins

        lo, hi = 75000.0, 80000.0
        test_prices = [70000.0, 75000.0, 75001.0, 77500.0, 80000.0, 80001.0, 90000.0]
        for px in test_prices:
            wins = [band_index_wins(i, oracle_px=px, lo=lo, hi=hi) for i in range(3)]
            assert sum(wins) == 1, f"Expected exactly 1 winner for oracle_px={px}, got wins={wins}"

    def test_invalid_band_raises(self) -> None:
        from hlanalysis.research.outcome_markets import band_index_wins

        with pytest.raises(ValueError):
            band_index_wins(5, oracle_px=75000.0, lo=70000.0, hi=80000.0)


# ---------------------------------------------------------------------------
# Real-data tests — skipped if data absent
# ---------------------------------------------------------------------------


class TestLoadMarketReference:
    def test_returns_nonempty_dataframe(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert not ref.empty, "Expected non-empty market_reference"

    def test_required_columns_present(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        required = [
            "symbol",
            "outcome_idx",
            "side_idx",
            "side_name",
            "market_class",
            "expiry",
            "target_price",
            "lo_threshold",
            "hi_threshold",
            "is_yes",
            "question_symbol",
        ]
        for col in required:
            assert col in ref.columns, f"Missing column: {col}"

    def test_no_null_symbols(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert ref["symbol"].notna().all(), "Null symbol found in market_reference"

    def test_symbol_format(self, con, data_root) -> None:
        """All symbols should start with '#'."""
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert ref["symbol"].str.startswith("#").all(), "Some symbols don't start with '#'"

    def test_side_idx_values(self, con, data_root) -> None:
        """side_idx should be 0 or 1."""
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert ref["side_idx"].isin([0, 1]).all(), "Unexpected side_idx values"

    def test_is_yes_consistent_with_side_idx(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert (ref["is_yes"] == (ref["side_idx"] == 0)).all(), "is_yes inconsistent with side_idx"

    def test_binary_legs_have_target_price(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        binary = ref[ref["market_class"] == "priceBinary"]
        if len(binary) > 0:
            assert binary["target_price"].notna().any(), "Binary legs should have at least some target_price"

    def test_counts_sensible(self, con, data_root) -> None:
        """Should have hundreds of unique leg symbols."""
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        assert len(ref) >= 10, f"Too few market_reference rows: {len(ref)}"

    def test_symbol_encoding(self, con, data_root) -> None:
        """Verify #NNN = outcome_idx * 10 + side_idx for known symbols."""
        from hlanalysis.research.outcome_markets import load_market_reference

        ref = load_market_reference(con, data_root)
        for _, row in ref.iterrows():
            sym_num = int(row["symbol"].lstrip("#"))
            expected = row["outcome_idx"] * 10 + row["side_idx"]
            assert sym_num == expected, (
                f"Symbol encoding mismatch: {row['symbol']} -> "
                f"outcome_idx={row['outcome_idx']}, side_idx={row['side_idx']}, "
                f"expected #{expected}"
            )


class TestLoadSettlements:
    def test_returns_dataframe(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        assert df is not None

    def test_required_columns(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        for col in ["symbol", "settled_at", "won", "settlement_price"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_nonempty(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        assert not df.empty, "Expected at least one settlement record"

    def test_all_symbols_yes_legs(self, con, data_root) -> None:
        """All settlement records are for Yes-leg symbols (ending in '0')."""
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        # Symbol ends in '0' (last digit = side_idx = 0 = Yes)
        last_digits = df["symbol"].str[-1]
        assert (last_digits == "0").all(), "Settlement symbols should end in '0' (Yes legs)"

    def test_all_won_true(self, con, data_root) -> None:
        """All settlement records should have won=True."""
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        assert df["won"].all(), "All settlement records should have won=True"

    def test_settlement_join_with_reference(self, con, data_root) -> None:
        """Settlement symbols should all exist in market_reference."""
        from hlanalysis.research.outcome_markets import load_market_reference, load_settlements

        ref = load_market_reference(con, data_root)
        settlements = load_settlements(con, data_root)

        ref_syms = set(ref["symbol"].tolist())
        settle_syms = set(settlements["symbol"].tolist())

        # All settlement symbols should be in reference
        missing = settle_syms - ref_syms
        assert len(missing) == 0, f"Settlement symbols not in market_reference: {missing}"

    def test_settled_at_positive_timestamps(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import load_settlements

        df = load_settlements(con, data_root)
        assert (df["settled_at"] > 0).all(), "settled_at should be positive timestamps"


class TestResolveBinaryOutcomes:
    """Tests for resolve_binary_outcomes() using oracle price."""

    def test_returns_nonempty_dataframe(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert not df.empty, "Expected non-empty binary outcomes"

    def test_required_columns(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        required = [
            "symbol",
            "outcome_idx",
            "expiry_str",
            "expiry",
            "target_price",
            "oracle_px_at_expiry",
            "yes_won",
            "winner_source",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_winner_source_is_oracle(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert (df["winner_source"] == "oracle").all(), "winner_source should be 'oracle'"

    def test_yes_won_is_bool(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert df["yes_won"].dtype == bool, f"yes_won should be bool, got {df['yes_won'].dtype}"

    def test_yes_win_rate_in_reasonable_range(self, con, data_root) -> None:
        """Yes-win rate should be between 30% and 70% for ATM daily binaries (n~38)."""
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert len(df) >= 10, f"Too few binary outcomes: {len(df)}"
        win_rate = df["yes_won"].mean()
        assert 0.30 <= win_rate <= 0.70, (
            f"Yes-win rate {win_rate:.3f} outside expected [0.30, 0.70] range "
            f"(n={len(df)}). This suggests a bug in the oracle ASOF JOIN or "
            f"targetPrice comparison."
        )

    def test_no_null_oracle_prices(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        null_oracle = df["oracle_px_at_expiry"].isna().sum()
        assert null_oracle == 0, (
            f"{null_oracle} binary expiries have null oracle_px_at_expiry — check oracle data coverage"
        )

    def test_target_price_positive(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert (df["target_price"] > 0).all(), "All target prices should be positive"

    def test_expiry_is_utc_datetime(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        import datetime as dt

        for exp in df["expiry"]:
            assert isinstance(exp, dt.datetime), f"expiry should be datetime, got {type(exp)}"
            assert exp.tzinfo is not None, "expiry should be timezone-aware"

    def test_count_matches_expiry_dates(self, con, data_root) -> None:
        """One row per unique expiry date (one binary per day)."""
        from hlanalysis.research.outcome_markets import resolve_binary_outcomes

        df = resolve_binary_outcomes(con, data_root)
        assert len(df) == df["expiry_str"].nunique(), "Duplicate expiry_str rows in binary outcomes"


class TestResolveBucketOutcomes:
    """Tests for resolve_bucket_outcomes() using oracle + terminal-mid cross-check."""

    def test_returns_nonempty_dataframe(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        assert not df.empty, "Expected non-empty bucket outcomes"

    def test_required_columns(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        required = [
            "symbol",
            "outcome_idx",
            "band_index",
            "q_symbol",
            "expiry_str",
            "expiry",
            "lo",
            "hi",
            "lo_bound",
            "hi_bound",
            "oracle_px_at_expiry",
            "terminal_mid",
            "band_won_oracle",
            "band_won_mid",
            "oracle_mid_agree",
            "winner_source",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_winner_source_is_oracle(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        assert (df["winner_source"] == "oracle").all(), "winner_source should be 'oracle'"

    def test_each_question_expiry_has_exactly_one_oracle_winner(self, con, data_root) -> None:
        """For each (q_symbol, expiry_str), exactly one band should win per oracle."""
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        winners_per_q = df.groupby(["q_symbol", "expiry_str"])["band_won_oracle"].sum()
        not_exactly_one = winners_per_q[winners_per_q != 1]
        assert len(not_exactly_one) == 0, (
            f"{len(not_exactly_one)} question/expiry pairs do not have exactly 1 oracle winner:\n{not_exactly_one}"
        )

    def test_oracle_mid_agreement_gte_90pct(self, con, data_root) -> None:
        """Oracle and terminal-mid should agree ≥ 90% on converged legs."""
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        has_mid = df["terminal_mid"].notna()
        if has_mid.sum() == 0:
            pytest.skip("No terminal mid data available")

        agree_rows = df.loc[has_mid, "oracle_mid_agree"]
        agree_pct = agree_rows.mean()
        assert agree_pct >= 0.90, (
            f"Oracle/mid agreement {agree_pct:.1%} < 90% — check band boundary convention or terminal-mid ASOF logic"
        )

    def test_band_index_values(self, con, data_root) -> None:
        """band_index should only be 0, 1, or 2 for named bands."""
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        assert df["band_index"].isin([0, 1, 2]).all(), "Unexpected band_index values"

    def test_lo_less_than_hi(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        assert (df["lo"] < df["hi"]).all(), "lo should be less than hi for all bucket legs"

    def test_lo_bound_hi_bound_consistent_with_band_index(self, con, data_root) -> None:
        """lo_bound/hi_bound should match the band_index_to_range() mapping."""
        from hlanalysis.research.outcome_markets import band_index_to_range, resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        for _, row in df.iterrows():
            lo_b, hi_b = band_index_to_range(int(row["band_index"]), float(row["lo"]), float(row["hi"]))
            if lo_b is None:
                assert row["lo_bound"] is None or (
                    row["lo_bound"] != row["lo_bound"]  # NaN
                ), f"Expected lo_bound=None for band 0, got {row['lo_bound']}"
            else:
                assert row["lo_bound"] == lo_b, f"lo_bound mismatch: {row['lo_bound']} != {lo_b}"
            if hi_b is None:
                assert row["hi_bound"] is None or (
                    row["hi_bound"] != row["hi_bound"]  # NaN
                ), f"Expected hi_bound=None for band 2, got {row['hi_bound']}"
            else:
                assert row["hi_bound"] == hi_b, f"hi_bound mismatch: {row['hi_bound']} != {hi_b}"

    def test_no_null_oracle_prices(self, con, data_root) -> None:
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        null_oracle = df["oracle_px_at_expiry"].isna().sum()
        assert null_oracle == 0, f"{null_oracle} bucket legs have null oracle_px_at_expiry — check oracle data coverage"

    def test_three_bands_per_question(self, con, data_root) -> None:
        """Each question/expiry should have exactly 3 named bands."""
        from hlanalysis.research.outcome_markets import resolve_bucket_outcomes

        df = resolve_bucket_outcomes(con, data_root)
        counts_per_q = df.groupby(["q_symbol", "expiry_str"])["band_index"].count()
        non_three = counts_per_q[counts_per_q != 3]
        assert len(non_three) == 0, f"{len(non_three)} question/expiry pairs don't have exactly 3 bands:\n{non_three}"
