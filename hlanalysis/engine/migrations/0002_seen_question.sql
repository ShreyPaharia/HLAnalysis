-- 0002 — track question_idxs the engine has emitted NewQuestion alerts for,
-- so restarts don't re-fire for every active market that's already known.
CREATE TABLE IF NOT EXISTS seen_question (
    question_idx INTEGER PRIMARY KEY,
    first_seen_ts_ns INTEGER NOT NULL
);
