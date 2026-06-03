-- 0005 — persist settlement realized PnL (SHR-53 / SHR-49).
--
-- HIP-4 binary positions are closed by settlement payouts, not by HL fills, so
-- the realized PnL of a settled position lived only on the bus/alert and was
-- never written to the DB. That made the daily-loss gate blind to the dominant
-- PnL component of the binary strategy (and let a transient HL-read failure
-- fall back to a structurally-zero local PnL). Persisting it here, keyed by
-- question_idx, also gives the two close paths (reconcile vanished-position and
-- router._close_settled) a single idempotent owner so they can't double-book.
CREATE TABLE IF NOT EXISTS settlement (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    ts_ns INTEGER NOT NULL
);
