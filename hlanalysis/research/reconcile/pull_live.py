"""Pull live engine data from EC2 via AWS SSM send-command (no SSH).

All queries target the **unified** engine state DB
(``/opt/hl-recorder/data/engine/state.db``) — the per-slot
``.../engine/<strategy_id>/state.db`` files are legacy. The unified DB mixes
every slot, so every query MUST filter by ``strategy_id`` (``v1``, ``v31``,
``v1_pm``, ``v31_pm``). HL recycles ``question_idx`` over time, so every query
is also windowed to ``[expiry_ns - 24h, expiry_ns + 60s]``.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INSTANCE_ID = "i-0dc4c0abec85a9eda"
DEFAULT_STRATEGY_ID = "v31"
_VENV_PYTHON = "/opt/hl-recorder/.venv/bin/python"

# Unified engine state DB (read-only). The per-slot v31/state.db files are legacy.
DB_PATH = "/opt/hl-recorder/data/engine/state.db"

# S3 transport for large results (e.g. the per-scan decision trace, which can be
# 137k+ rows / ~11MB). AWS SSM ``GetCommandInvocation.StandardOutputContent`` is
# hard-capped at 24,000 bytes, so anything bigger than ~35 trace rows truncates
# mid-row on the inline path. Instead the box gzips the result and uploads it
# here; we download + decompress locally. Override the bucket via env for other
# accounts. The instance role already writes to the recorder archive bucket.
S3_TRANSPORT_BUCKET = os.environ.get("HLBT_RECON_S3_BUCKET", "hl-recorder-archive-819175935435")
_S3_TRANSPORT_PREFIX = "tmp/recon"

# Sealed decision-trace segments are archived by scripts/sync-engine-to-s3.sh under
# ``<engine-prefix>/date=YYYY-MM-DD/<strategy_id>/traces/*.jsonl.gz`` and then pruned
# from the box. Old reconciles must read them from S3. This mirrors the sync
# script's S3_ENGINE_PREFIX (default "engine").
_TRACE_ARCHIVE_PREFIX = os.environ.get("HLBT_ENGINE_S3_PREFIX", "engine")

_NS_PER_S = 1_000_000_000
_24H_NS = 24 * 3600 * _NS_PER_S
_60S_NS = 60 * _NS_PER_S


def _trace_path(strategy_id: str) -> str:
    """Per-slot decision_trace path on EC2 for a given strategy_id."""
    return f"/opt/hl-recorder/data/engine/{strategy_id}/decision_trace.jsonl"


@dataclass
class LiveData:
    """Container for all live data pulled from EC2.

    Parameters
    ----------
    fills:
        Venue-confirmed fills, cols: ts_ns, symbol, side, price, size, fee, closed_pnl.
    trace:
        Decision trace rows (canonical schema), one row per scan.
    settlement:
        Settlement info dict with keys: question_idx, realized_pnl, ts_ns,
        winner_side, source.
    halts_rejects:
        Engine halt/reject events, cols: ts_ns, kind, reason, payload_json.
    config_hash:
        Config hash from the most recent decision_trace row, or None.
    """

    fills: pd.DataFrame
    trace: pd.DataFrame
    settlement: dict[str, Any]
    halts_rejects: pd.DataFrame
    config_hash: str | None


def _check_aws_cli() -> None:
    """Raise ImportError with a helpful message if AWS CLI is not available."""
    result = subprocess.run(["which", "aws"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ImportError(
            "AWS CLI not found on PATH. Install it with: "
            "pip install awscli  or  brew install awscli. "
            "Then configure credentials with: aws configure"
        )


def _ssm_python(
    script: str,
    instance_id: str = DEFAULT_INSTANCE_ID,
    timeout_s: int = 60,
) -> str:
    """Base64-encode a python script and run it on EC2 via SSM.

    Parameters
    ----------
    script:
        Python source code to execute on the remote instance.
    instance_id:
        EC2 instance ID.
    timeout_s:
        Maximum seconds to wait for the command to complete.

    Returns
    -------
    Standard output from the remote command.
    """
    _check_aws_cli()
    b64 = base64.b64encode(script.encode()).decode()
    cmd = [
        "aws",
        "ssm",
        "send-command",
        "--instance-ids",
        instance_id,
        "--document-name",
        "AWS-RunShellScript",
        "--parameters",
        f'commands=["echo {b64} | base64 --decode | {_VENV_PYTHON}"]',
        "--output",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    command_id = json.loads(result.stdout)["Command"]["CommandId"]

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2)
        poll = subprocess.run(
            [
                "aws",
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                instance_id,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        inv = json.loads(poll.stdout)
        status = inv.get("Status", "")
        if status in ("Success", "Failed", "Cancelled", "TimedOut"):
            if status != "Success":
                raise RuntimeError(f"SSM command {status}: {inv.get('StandardErrorContent', '')}")
            return inv.get("StandardOutputContent", "")
    raise TimeoutError(f"SSM command did not complete in {timeout_s}s")


def _s3_download_bytes(bucket: str, key: str) -> bytes:
    """Download ``s3://bucket/key`` into memory via the AWS CLI (no boto3 dep
    on the client side; mirrors the rest of this module's CLI-only approach)."""
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "obj"
        subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(local)],
            capture_output=True,
            text=True,
            check=True,
        )
        return local.read_bytes()


def _s3_delete(bucket: str, key: str) -> None:
    """Best-effort cleanup of a transport object (never raises)."""
    subprocess.run(
        ["aws", "s3", "rm", f"s3://{bucket}/{key}"],
        capture_output=True,
        text=True,
        check=False,
    )


def _s3_list_keys(bucket: str, prefix: str) -> list[str]:
    """List object keys under ``s3://bucket/prefix`` (best-effort).

    Returns ``[]`` on any failure (missing AWS CLI, no objects, no credentials)
    so a transient listing problem degrades to "no sealed segments" rather than
    raising — the live on-box file path still applies for recent reconciles.
    """
    try:
        out = subprocess.run(
            [
                "aws",
                "s3api",
                "list-objects-v2",
                "--bucket",
                bucket,
                "--prefix",
                prefix,
                "--query",
                "Contents[].Key",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    keys = json.loads(out or "null")
    return keys if isinstance(keys, list) else []


def _window_dates(expiry_ns: int) -> list[str]:
    """date= partitions that can hold rows in ``[expiry-24h, expiry+60s]``.

    Segments are sealed daily (~06:45 UTC) and a single segment can straddle the
    seal boundary, so a row near the window edge may live in the partition for the
    day before or after the expiry date. Cover expiry_date ± 1 day to be safe.
    """
    exp = datetime.fromtimestamp(expiry_ns / 1e9, tz=UTC).date()
    return [(exp + timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 0, 1)]


def _trace_segments_from_s3(
    question_idx: int,
    window_start: int,
    window_end: int,
    strategy_id: str,
    expiry_ns: int,
) -> list[dict[str, Any]]:
    """Read sealed trace segments from S3 for the window, filtered to the question."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in _window_dates(expiry_ns):
        prefix = f"{_TRACE_ARCHIVE_PREFIX}/date={d}/{strategy_id}/traces/"
        for key in _s3_list_keys(S3_TRANSPORT_BUCKET, prefix):
            if key in seen:
                continue
            seen.add(key)
            try:
                raw = gzip.decompress(_s3_download_bytes(S3_TRANSPORT_BUCKET, key))
            except Exception:
                continue
            for line in raw.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("question_idx") != question_idx:
                    continue
                ts = obj.get("ts_ns", 0)
                if ts < window_start or ts > window_end:
                    continue
                rows.append(obj)
    return rows


def _dedup_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup by (ts_ns, question_idx) — one scan per question per slot — keeping
    the first occurrence (live-file rows take precedence over S3), time-ordered."""
    by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for r in rows:
        by_key.setdefault((r.get("ts_ns"), r.get("question_idx")), r)
    return sorted(by_key.values(), key=lambda r: r.get("ts_ns") or 0)


def _config_hash_from_s3_segments(strategy_id: str, *, days: int = 7) -> str | None:
    """Newest sealed segment's config_hash, searching recent date partitions.

    Used when the live on-box trace has just been rotated away (empty tail)."""
    today = datetime.now(UTC).date()
    for back in range(days):
        d = (today - timedelta(days=back)).strftime("%Y-%m-%d")
        prefix = f"{_TRACE_ARCHIVE_PREFIX}/date={d}/{strategy_id}/traces/"
        keys = _s3_list_keys(S3_TRANSPORT_BUCKET, prefix)
        for key in sorted(keys, reverse=True):  # newest seal stamp first
            try:
                raw = gzip.decompress(_s3_download_bytes(S3_TRANSPORT_BUCKET, key))
            except Exception:
                continue
            for line in reversed(raw.decode().splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    ch = json.loads(line).get("config_hash")
                except Exception:
                    continue
                if ch:
                    return str(ch)
    return None


def _ssm_fetch_large_json(
    compute_script: str,
    result_var: str,
    *,
    instance_id: str = DEFAULT_INSTANCE_ID,
    timeout_s: int = 300,
    s3_bucket: str | None = None,
    s3_key: str | None = None,
) -> Any:
    """Run remote python and retrieve an arbitrarily large JSON result via S3.

    ``compute_script`` is remote python that assigns a JSON-serialisable object
    to ``result_var``. The wrapper gzips ``json.dumps(result_var)`` and uploads
    it to ``s3://s3_bucket/s3_key`` on the box (using boto3, available in the
    recorder venv), then we download + decompress + parse locally. This bypasses
    the 24,000-byte SSM inline-output cap and the inline 60s timeout, so a
    full-day decision trace (137k+ rows) round-trips intact.

    Parameters
    ----------
    compute_script:
        Remote python that assigns ``result_var`` (e.g. builds ``rows``).
    result_var:
        Name of the variable holding the JSON-serialisable result.
    instance_id:
        EC2 instance ID.
    timeout_s:
        Max seconds for the remote filter+gzip+upload step (default 300 — a
        157MB trace read is several seconds; give headroom).
    s3_bucket / s3_key:
        Transport object location; defaults to ``S3_TRANSPORT_BUCKET`` and a
        unique ``tmp/recon/<uuid>.json.gz`` key.

    Returns
    -------
    The parsed JSON object (e.g. ``list[dict]``).
    """
    bucket = s3_bucket or S3_TRANSPORT_BUCKET
    key = s3_key or f"{_S3_TRANSPORT_PREFIX}/{uuid.uuid4().hex}.json.gz"
    upload = (
        compute_script
        + "\nimport json as _j, gzip as _g, boto3 as _b\n"
        + f"_b.client('s3').put_object(Bucket={bucket!r}, Key={key!r}, "
        + f"Body=_g.compress(_j.dumps({result_var}).encode()))\n"
        + "print('S3_UPLOAD_OK')\n"
    )
    # Key is generated up-front and deleted in `finally` regardless of where we
    # fail. If the remote step uploads then errors on a later line (or SSM reports
    # Failed post-upload), the object would otherwise orphan in S3 (audit M5).
    # S3 delete is idempotent, so cleaning up a never-created key is harmless.
    try:
        _ssm_python(upload, instance_id=instance_id, timeout_s=timeout_s)
        raw = gzip.decompress(_s3_download_bytes(bucket, key))
    finally:
        _s3_delete(bucket, key)
    return json.loads(raw or b"null")


def _cache_path(
    cache_dir: Path,
    question_idx: int,
    kind: str,
    strategy_id: str = DEFAULT_STRATEGY_ID,
) -> Path:
    """Return a deterministic, strategy-scoped cache file path for a given pull."""
    return cache_dir / f"q{question_idx}_{strategy_id}_{kind}.json"


def pull_live_fills(
    question_idx: int,
    expiry_ns: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> pd.DataFrame:
    """Pull venue-confirmed fills for question_idx windowed to [expiry_ns - 24h, expiry_ns + 60s].

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    strategy_id:
        Slot to filter the unified DB by (e.g. ``v31``, ``v1``).
    cache_dir:
        Directory to cache/read the pulled data. If provided and the cache
        file exists, it is read instead of pulling from EC2.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    DataFrame with cols: ts_ns, symbol, side, price, size, fee, closed_pnl.
    Only source='venue' fills are returned. Empty DataFrame if none found.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "fills", strategy_id)
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else _empty_fills()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json

con = sqlite3.connect("file:{DB_PATH}?mode=ro", uri=True)
rows = con.execute(
    \"\"\"
    SELECT ts_ns, symbol, side, price, size, fee, closed_pnl
    FROM fill
    WHERE question_idx = ? AND strategy_id = ? AND source = 'venue'
      AND ts_ns >= ? AND ts_ns <= ?
    ORDER BY ts_ns
    \"\"\",
    ({question_idx}, {strategy_id!r}, {window_start}, {window_end}),
).fetchall()
con.close()
print(json.dumps([
    dict(ts_ns=r[0], symbol=r[1], side=r[2], price=r[3],
         size=r[4], fee=r[5], closed_pnl=r[6])
    for r in rows
]))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    rows = json.loads(raw.strip() or "[]")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "fills", strategy_id).write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else _empty_fills()


def _empty_fills() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_ns", "symbol", "side", "price", "size", "fee", "closed_pnl"])


def pull_live_trace(
    question_idx: int,
    expiry_ns: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    trace_path: str | None = None,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> pd.DataFrame:
    """Pull decision_trace rows for question_idx windowed by expiry.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    strategy_id:
        Slot to pull the trace for; the JSONL path is built from this.
    trace_path:
        Override path to the decision_trace.jsonl on EC2 (defaults to the
        per-slot path derived from ``strategy_id``).
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    DataFrame with canonical schema columns. Empty if none found.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "trace", strategy_id)
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else pd.DataFrame()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS
    path = trace_path if trace_path is not None else _trace_path(strategy_id)

    # The trace can be 137k+ rows / ~11MB — far past the 24KB SSM inline cap.
    # Filter on the box, then ship the result out via S3 (see _ssm_fetch_large_json).
    compute = f"""
import json

path = {path!r}
question_idx = {question_idx}
window_start = {window_start}
window_end = {window_end}
rows = []
try:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("question_idx") != question_idx:
                continue
            ts = obj.get("ts_ns", 0)
            if ts < window_start or ts > window_end:
                continue
            rows.append(obj)
except FileNotFoundError:
    pass
"""
    live_rows = _ssm_fetch_large_json(compute, "rows", instance_id=instance_id) or []
    # Old rows live ONLY in sealed S3 segments once the box prunes them — union
    # those with the live file and dedup so a reconcile run after rotation still
    # sees the full pre-settlement decision history.
    seg_rows = _trace_segments_from_s3(question_idx, window_start, window_end, strategy_id, expiry_ns)
    rows = _dedup_trace_rows(live_rows + seg_rows)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "trace", strategy_id).write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def pull_settlement(
    question_idx: int,
    expiry_ns: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> dict[str, Any]:
    """Pull the settlement for question_idx from the unified DB.

    Reads the ``settlement`` table (summed across symbols, windowed, scoped to
    ``strategy_id``). If that table has no row for the question, falls back to
    the **HL settlement-as-fill** convention: HL books settlement as a venue
    fill at price ~1.0 (the held leg won) / ~0.0 (it lost). The realized PnL
    and winning side are then derived from the final venue fill(s).

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds (bounds the lookup window).
    strategy_id:
        Slot to filter the unified DB by.
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    Dict with keys: question_idx, realized_pnl, ts_ns, winner_side, source.
    Empty dict if neither the settlement table nor a settlement-as-fill is found.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "settlement", strategy_id)
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json

con = sqlite3.connect("file:{DB_PATH}?mode=ro", uri=True)

# 1) Authoritative: the settlement table (summed across symbols, windowed).
srow = con.execute(
    \"\"\"
    SELECT SUM(realized_pnl), MAX(ts_ns)
    FROM settlement
    WHERE question_idx = ? AND strategy_id = ?
      AND ts_ns >= ? AND ts_ns <= ?
    \"\"\",
    ({question_idx}, {strategy_id!r}, {window_start}, {window_end}),
).fetchone()

if srow is not None and srow[0] is not None:
    con.close()
    print(json.dumps(dict(
        question_idx={question_idx}, realized_pnl=srow[0], ts_ns=srow[1],
        winner_side=None, source="settlement_table",
    )))
else:
    # 2) Fallback: HL books settlement as a venue fill at px~1.0 (win) / ~0.0 (loss).
    frows = con.execute(
        \"\"\"
        SELECT ts_ns, symbol, side, price, closed_pnl
        FROM fill
        WHERE question_idx = ? AND strategy_id = ? AND source = 'venue'
          AND ts_ns >= ? AND ts_ns <= ?
        ORDER BY ts_ns
        \"\"\",
        ({question_idx}, {strategy_id!r}, {window_start}, {window_end}),
    ).fetchall()
    con.close()
    settle = [r for r in frows if r[3] is not None and (r[3] >= 0.99 or r[3] <= 0.01)]
    if not settle:
        print(json.dumps({{}}))
    else:
        last = settle[-1]
        symbol = last[1] or ""
        price = last[3]
        side_idx = None
        if "#" in symbol:
            digits = "".join(ch for ch in symbol.rsplit("#", 1)[1] if ch.isdigit())
            if digits:
                side_idx = int(digits) % 10
        winner_side = None
        if side_idx is not None:
            leg_is_yes = side_idx == 0
            won = price >= 0.5
            winner_side = ("yes" if leg_is_yes else "no") if won else ("no" if leg_is_yes else "yes")
        realized = sum(r[4] for r in settle if r[4] is not None)
        print(json.dumps(dict(
            question_idx={question_idx}, realized_pnl=realized, ts_ns=last[0],
            winner_side=winner_side, settlement_price=price, source="settlement_as_fill",
        )))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    result: dict[str, Any] = json.loads(raw.strip() or "{}")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "settlement", strategy_id).write_text(json.dumps(result))

    return result


def pull_halts_rejects(
    question_idx: int,
    expiry_ns: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> pd.DataFrame:
    """Pull halt/reject events for question_idx.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    strategy_id:
        Slot to filter the unified DB by.
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    DataFrame with cols: ts_ns, kind, reason, payload_json.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "halts_rejects", strategy_id)
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else _empty_halts()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json

KINDS = ("order_rejected", "reconcile_drift", "halt")
con = sqlite3.connect("file:{DB_PATH}?mode=ro", uri=True)
placeholders = ",".join("?" * len(KINDS))
rows = con.execute(
    f\"\"\"
    SELECT ts_ns, kind, reason, payload_json
    FROM events
    WHERE question_idx = ? AND strategy_id = ? AND kind IN ({{placeholders}})
      AND ts_ns >= ? AND ts_ns <= ?
    ORDER BY ts_ns
    \"\"\",
    ({question_idx}, {strategy_id!r}, *KINDS, {window_start}, {window_end}),
).fetchall()
con.close()
print(json.dumps([
    dict(ts_ns=r[0], kind=r[1], reason=r[2], payload_json=r[3])
    for r in rows
]))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    rows = json.loads(raw.strip() or "[]")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "halts_rejects", strategy_id).write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else _empty_halts()


def _empty_halts() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_ns", "kind", "reason", "payload_json"])


def pull_config_hash(
    strategy_id: str = DEFAULT_STRATEGY_ID,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> str | None:
    """Pull the config_hash from the most recent decision_trace row of a slot.

    Parameters
    ----------
    strategy_id:
        Slot whose decision_trace to read.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    Config hash string, or None if the trace is empty or the field is absent.
    """
    # config_hash is identical on every row, so read only the file TAIL — a full
    # readlines() of a ~157MB trace risks the 60s SSM timeout for no reason.
    script = f"""
import json, subprocess

TRACE = {_trace_path(strategy_id)!r}
cfg = None
try:
    tail = subprocess.run(["tail", "-n", "5", TRACE], capture_output=True, text=True).stdout
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            cfg = json.loads(line).get("config_hash")
            break
        except Exception:
            continue
except FileNotFoundError:
    pass
print(json.dumps(cfg))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    value = json.loads(raw.strip() or "null")
    if value is not None:
        return str(value)
    # The live file is often freshly rotated away (empty tail) by the time a
    # reconcile runs; fall back to the newest sealed segment archived in S3.
    return _config_hash_from_s3_segments(strategy_id)


def pull_all(
    question_idx: int,
    expiry_ns: int,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> LiveData:
    """Convenience: pull fills, trace, settlement, halts_rejects, and config_hash.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds.
    strategy_id:
        Slot to scope every pull to.
    cache_dir:
        Directory to cache/read pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    LiveData container with all pulled fields.
    """
    fills = pull_live_fills(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
    trace = pull_live_trace(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
    settlement = pull_settlement(question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id)
    halts_rejects = pull_halts_rejects(
        question_idx, expiry_ns, strategy_id, cache_dir=cache_dir, instance_id=instance_id
    )
    config_hash = pull_config_hash(strategy_id, instance_id=instance_id)
    return LiveData(
        fills=fills,
        trace=trace,
        settlement=settlement,
        halts_rejects=halts_rejects,
        config_hash=config_hash,
    )
