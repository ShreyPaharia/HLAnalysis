# Dump v1 priceBucket fills since 2026-05-31, fill-by-fill + per-coin net.
#
# Read-only: builds an exec client and reads venue user_fills. Places NO orders.
# Run on the live engine box via SSM (SSH is blocked):
#   cd /opt/hl-recorder && set -a && . /etc/hl-engine/env && set +a \
#     && .venv/bin/python /tmp/dump_v1_bucket_fills.py
#
# Used for the 2026-06-08 v1-bucket live-vs-sim divergence analysis
# (docs/research/2026-06-08-v1-bucket-live-vs-sim.md).
import collections
from pathlib import Path
from datetime import datetime, timezone

from hlanalysis.engine.config import load_deploy_config
from hlanalysis.engine.config_builders import build_exec_client
from hlanalysis.engine.state import StateDAL

deploy = load_deploy_config(Path("config/deploy.yaml"))
acct = deploy.accounts["v1"]
client = build_exec_client("v1", acct, paper_mode=False)
dal = StateDAL(Path(deploy.state_db_path_for("v1")))
kmap = dal.coin_klass_map()


def dt(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


START = int(datetime(2026, 5, 31, tzinfo=timezone.utc).timestamp() * 1e9)
fills = [
    f
    for f in client.user_fills(since_ts_ns=0)
    if f.symbol.startswith("#")
    and kmap.get(f.symbol) == "priceBucket"
    and f.ts_ns >= START
]
fills.sort(key=lambda f: f.ts_ns)
print(f"LIVE v1 priceBucket since 05-31: {len(fills)} fills")
print("ts,coin,side,px,sz,ntl,closedPnl,fee")
for f in fills:
    print(
        f"{dt(f.ts_ns):%Y-%m-%d %H:%M:%S},{f.symbol},{f.side},"
        f"{f.price:.4f},{f.size:.2f},{abs(f.price * f.size):.1f},"
        f"{f.closed_pnl:+.3f},{f.fee:.4f}"
    )
by = collections.defaultdict(lambda: [0.0, 0])
for f in fills:
    b = by[(dt(f.ts_ns).strftime("%Y-%m-%d"), f.symbol)]
    b[0] += f.closed_pnl - f.fee
    b[1] += 1
print("--- per (day,coin) net ---")
for k in sorted(by):
    print(f"{k[0]} {k[1]} net {by[k][0]:+.3f} fills {by[k][1]}")
print("TOTAL", round(sum(f.closed_pnl - f.fee for f in fills), 2))
