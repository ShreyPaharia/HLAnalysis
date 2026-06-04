-- 0006 — append-only events table for engine observability (Component 2).
--
-- Every published BusEvent is persisted here via _events_persist_loop so that
-- "why didn't slot X trade between T1 and T2?" and "how many rejects, of what
-- kind?" are answerable with a single SQL query instead of prose archaeology.
-- Retention is bounded by both age (default 14 days) and row count
-- (default 1,000,000) — the unbounded-_questions→OOM scar is why the ceiling
-- is non-optional on the 1 GiB box (hl_live_eval_2026_05_31).
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns         INTEGER NOT NULL,
    alias         TEXT,
    kind          TEXT NOT NULL,
    question_idx  INTEGER,
    reason        TEXT,
    payload_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_alias_kind_ts
    ON events (alias, kind, ts_ns);

CREATE INDEX IF NOT EXISTS idx_events_question_ts
    ON events (question_idx, ts_ns);
