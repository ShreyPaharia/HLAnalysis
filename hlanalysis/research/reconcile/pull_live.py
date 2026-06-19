"""Pull live engine data from EC2 via AWS SSM send-command (no SSH)."""

from __future__ import annotations

import base64
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INSTANCE_ID = "i-0dc4c0abec85a9eda"
_VENV_PYTHON = "/opt/hl-recorder/.venv/bin/python"

_NS_PER_S = 1_000_000_000
_24H_NS = 24 * 3600 * _NS_PER_S
_60S_NS = 60 * _NS_PER_S


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
        Settlement info dict with keys: question_idx, realized_pnl, ts_ns.
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


def _cache_path(
    cache_dir: Path,
    question_idx: int,
    kind: str,
) -> Path:
    """Return a deterministic cache file path for a given pull."""
    return cache_dir / f"q{question_idx}_{kind}.json"


def pull_live_fills(
    question_idx: int,
    expiry_ns: int,
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
        cache_file = _cache_path(cache_dir, question_idx, "fills")
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else _empty_fills()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json, sys

DB_PATH = "/opt/hl-recorder/data/engine/v31/state.db"
try:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        \"\"\"
        SELECT ts_ns, symbol, side, price, size, fee, closed_pnl
        FROM fills
        WHERE question_idx = ? AND source = 'venue'
          AND ts_ns >= ? AND ts_ns <= ?
        ORDER BY ts_ns
        \"\"\",
        ({question_idx}, {window_start}, {window_end}),
    ).fetchall()
    con.close()
    print(json.dumps([
        dict(ts_ns=r[0], symbol=r[1], side=r[2], price=r[3],
             size=r[4], fee=r[5], closed_pnl=r[6])
        for r in rows
    ]))
except Exception as e:
    print(json.dumps([]))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    rows = json.loads(raw.strip() or "[]")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "fills").write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else _empty_fills()


def _empty_fills() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_ns", "symbol", "side", "price", "size", "fee", "closed_pnl"])


def pull_live_trace(
    question_idx: int,
    expiry_ns: int,
    trace_path: str = "/opt/hl-recorder/data/engine/v31/decision_trace.jsonl",
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
    trace_path:
        Path to the decision_trace.jsonl on EC2.
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    DataFrame with canonical schema columns. Empty if none found.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "trace")
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else pd.DataFrame()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import json, sys

path = {repr(trace_path)}
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
print(json.dumps(rows))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    rows = json.loads(raw.strip() or "[]")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "trace").write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def pull_settlement(
    question_idx: int,
    expiry_ns: int,
    cache_dir: Path | None = None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> dict[str, Any]:
    """Pull settlement row for question_idx.

    Parameters
    ----------
    question_idx:
        HL question/market index.
    expiry_ns:
        Market expiry timestamp in nanoseconds (used to bound the lookup window).
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    Dict with keys: question_idx, realized_pnl, ts_ns. Empty dict if none found.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "settlement")
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json

DB_PATH = "/opt/hl-recorder/data/engine/v31/state.db"
try:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        \"\"\"
        SELECT question_idx, realized_pnl, ts_ns
        FROM events
        WHERE question_idx = ? AND kind = 'settlement'
          AND ts_ns >= ? AND ts_ns <= ?
        ORDER BY ts_ns DESC LIMIT 1
        \"\"\",
        ({question_idx}, {window_start}, {window_end}),
    ).fetchone()
    con.close()
    if row:
        print(json.dumps(dict(question_idx=row[0], realized_pnl=row[1], ts_ns=row[2])))
    else:
        print(json.dumps({{}}))
except Exception:
    print(json.dumps({{}}))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    result: dict[str, Any] = json.loads(raw.strip() or "{}")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "settlement").write_text(json.dumps(result))

    return result


def pull_halts_rejects(
    question_idx: int,
    expiry_ns: int,
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
    cache_dir:
        Directory to cache/read the pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    DataFrame with cols: ts_ns, kind, reason, payload_json.
    """
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, question_idx, "halts_rejects")
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
            return pd.DataFrame(rows) if rows else _empty_halts()

    window_start = expiry_ns - _24H_NS
    window_end = expiry_ns + _60S_NS

    script = f"""
import sqlite3, json

DB_PATH = "/opt/hl-recorder/data/engine/v31/state.db"
KINDS = ("order_rejected", "reconcile_drift", "halt")
try:
    con = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(KINDS))
    rows = con.execute(
        f\"\"\"
        SELECT ts_ns, kind, reason, payload_json
        FROM events
        WHERE question_idx = ? AND kind IN ({{placeholders}})
          AND ts_ns >= ? AND ts_ns <= ?
        ORDER BY ts_ns
        \"\"\",
        ({question_idx}, *KINDS, {window_start}, {window_end}),
    ).fetchall()
    con.close()
    print(json.dumps([
        dict(ts_ns=r[0], kind=r[1], reason=r[2], payload_json=r[3])
        for r in rows
    ]))
except Exception:
    print(json.dumps([]))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    rows = json.loads(raw.strip() or "[]")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, question_idx, "halts_rejects").write_text(json.dumps(rows))

    return pd.DataFrame(rows) if rows else _empty_halts()


def _empty_halts() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts_ns", "kind", "reason", "payload_json"])


def pull_config_hash(
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> str | None:
    """Pull the config_hash from the most recent decision_trace row.

    Parameters
    ----------
    instance_id:
        EC2 instance ID.

    Returns
    -------
    Config hash string, or None if the trace is empty or the field is absent.
    """
    script = """
import json

TRACE = "/opt/hl-recorder/data/engine/v31/decision_trace.jsonl"
last_obj = None
try:
    with open(TRACE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last_obj = json.loads(line)
            except Exception:
                continue
except FileNotFoundError:
    pass
print(json.dumps(last_obj.get("config_hash") if last_obj else None))
"""
    raw = _ssm_python(script, instance_id=instance_id)
    value = json.loads(raw.strip() or "null")
    return str(value) if value is not None else None


def pull_all(
    question_idx: int,
    expiry_ns: int,
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
    cache_dir:
        Directory to cache/read pulled data.
    instance_id:
        EC2 instance ID.

    Returns
    -------
    LiveData container with all pulled fields.
    """
    fills = pull_live_fills(question_idx, expiry_ns, cache_dir=cache_dir, instance_id=instance_id)
    trace = pull_live_trace(question_idx, expiry_ns, cache_dir=cache_dir, instance_id=instance_id)
    settlement = pull_settlement(question_idx, expiry_ns, cache_dir=cache_dir, instance_id=instance_id)
    halts_rejects = pull_halts_rejects(question_idx, expiry_ns, cache_dir=cache_dir, instance_id=instance_id)
    config_hash = pull_config_hash(instance_id=instance_id)
    return LiveData(
        fills=fills,
        trace=trace,
        settlement=settlement,
        halts_rejects=halts_rejects,
        config_hash=config_hash,
    )
