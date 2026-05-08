CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS openorder (
    cloid TEXT PRIMARY KEY,
    venue_oid TEXT,
    question_idx INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    status TEXT NOT NULL,
    placed_ts_ns INTEGER NOT NULL,
    last_update_ts_ns INTEGER NOT NULL,
    strategy_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_openorder_status ON openorder(status);
CREATE INDEX IF NOT EXISTS idx_openorder_question ON openorder(question_idx);

CREATE TABLE IF NOT EXISTS position (
    question_idx INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_entry REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    last_update_ts_ns INTEGER NOT NULL,
    stop_loss_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fill (
    fill_id TEXT PRIMARY KEY,
    cloid TEXT NOT NULL,
    question_idx INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    fee REAL NOT NULL,
    ts_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fill_cloid ON fill(cloid);

CREATE TABLE IF NOT EXISTS session (
    session_id TEXT PRIMARY KEY,
    started_ts_ns INTEGER NOT NULL,
    ended_ts_ns INTEGER,
    halt_reason TEXT
);
