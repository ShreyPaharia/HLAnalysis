from __future__ import annotations

from pathlib import Path

from hlanalysis.sim.data.cache import Cache
from hlanalysis.sim.data.schemas import PMTrade


def test_cache_roundtrip_trades(tmp_path: Path):
    c = Cache(root=tmp_path)
    cond = "0xabc"
    trades = [
        PMTrade(ts_ns=1, token_id="t1", side="buy", price=0.5, size=1.0),
        PMTrade(ts_ns=2, token_id="t1", side="sell", price=0.51, size=2.0),
    ]
    c.write_trades(cond, trades)
    got = c.read_trades(cond)
    assert len(got) == 2
    assert got[0].size == 1.0


def test_cache_manifest_records_pull(tmp_path: Path):
    c = Cache(root=tmp_path)
    c.update_manifest(condition_id="0xabc", n_rows=2, last_pull_ts_ns=12345)
    info = c.get_manifest("0xabc")
    assert info["n_rows"] == 2
    assert info["last_pull_ts_ns"] == 12345
