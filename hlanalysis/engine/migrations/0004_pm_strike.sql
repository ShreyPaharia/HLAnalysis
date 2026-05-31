-- 0004 — persist captured open-strikes for Polymarket up/down markets.
--
-- PM "BTC Up or Down" markets carry no static strike; they resolve against a
-- Binance reference candle. The engine stamps the strike from the live bbo
-- reference mark at the market open (MarketState.capture_pm_open_strike) and
-- persists it here so a restart can reload it — otherwise a mid-day restart
-- would skip markets whose open it can no longer observe (it never prices a
-- market off a stale/guessed open).
CREATE TABLE IF NOT EXISTS pm_strike (
    question_idx INTEGER PRIMARY KEY,
    strike REAL NOT NULL
);
