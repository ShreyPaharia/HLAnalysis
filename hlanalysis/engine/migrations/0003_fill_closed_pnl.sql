-- 0003 — persist realized PnL alongside fees on the fill row.
--
-- Daily-loss accounting now reads from HL's user_fills (closedPnl), but we
-- still want the local DB to be diagnosable: Router._book_fill writes a Fill
-- row on every venue fill, including the realized PnL on closes, so an after-
-- the-fact `SELECT SUM(closed_pnl - fee) FROM fill WHERE ts_ns >= ?` matches
-- what the live gate saw. Without this column the local realized_pnl_since
-- helper is structurally near-zero (closed positions get deleted, taking
-- their realized PnL with them).
ALTER TABLE fill ADD COLUMN closed_pnl REAL NOT NULL DEFAULT 0.0;
