# Dump RAW HL user_fills for slots v1 + v31 (read-only) + each slot's coin→klass
# map, as machine-parseable CSV on stdout. Places NO orders.
#
# Run on the live engine box via SSM (SSH is blocked):
#   cd /opt/hl-recorder && set -a && . /etc/hl-engine/env && set +a \
#     && .venv/bin/python /tmp/dump_hl_fills_all.py
#
# Output sections (prefixed):
#   DIAG,<slot>,<account_address>,<n_raw_rows>,<err>
#   FILL,<slot>,<ts_ns>,<coin>,<dir>,<side>,<px>,<sz>,<fee>,<closedPnl>,<cloid>
#   KLASS,<slot>,<coin>,<klass>
# Used for docs/research/2026-06-10-hl-all-days-live-vs-sim-shr98.md
import traceback
from pathlib import Path

from hlanalysis.engine.config import load_deploy_config
from hlanalysis.engine.config_builders import build_exec_client
from hlanalysis.engine.state import StateDAL

deploy = load_deploy_config(Path("config/deploy.yaml"))

# Only the 4-day window legs (06-06..06-09) + their #..1 complements. Filtering
# here keeps stdout under SSM's 24KB inline cap (avoids returning all-time fills).
WINDOW_LEGS = {
    "#1590", "#1591", "#1610", "#1611", "#1620", "#1621", "#1630", "#1631",
    "#1640", "#1641", "#1660", "#1661", "#1670", "#1671", "#1680", "#1681",
    "#2200", "#2201", "#2220", "#2221", "#2230", "#2231", "#2240", "#2241",
    "#2250", "#2251", "#2270", "#2271", "#2280", "#2281", "#2290", "#2291",
}

for slot in ("v1", "v31"):
    try:
        acct = deploy.accounts[slot]
        addr = getattr(acct, "account_address", "?")
        client = build_exec_client(slot, acct, paper_mode=False)
        rows = client._fetch_fills_raw(0)
        print(f"DIAG,{slot},{addr},{len(rows or [])},")
        for r in rows or []:
            ts_ns = int(r.get("time", 0)) * 1_000_000
            coin = str(r.get("coin", ""))
            if coin not in WINDOW_LEGS:
                continue
            side = "buy" if r.get("side") == "B" else "sell"
            print(
                f"FILL,{slot},{ts_ns},{coin},{r.get('dir','')},{side},"
                f"{float(r.get('px',0)):.6f},{float(r.get('sz',0)):.4f},"
                f"{float(r.get('fee',0)):.6f},{float(r.get('closedPnl',0) or 0):.6f},"
                f"{r.get('cloid','')}"
            )
        dal = StateDAL(Path(deploy.state_db_path_for(slot)))
        for coin, klass in dal.coin_klass_map().items():
            if coin in WINDOW_LEGS:
                print(f"KLASS,{slot},{coin},{klass}")
    except Exception as e:
        print(f"DIAG,{slot},ERR,0,{type(e).__name__}: {e}")
        traceback.print_exc()
