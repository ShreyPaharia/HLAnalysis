from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import PMMarket, PMTrade


@dataclass(slots=True)
class Cache:
    root: Path

    def __post_init__(self) -> None:
        for sub in ("pm_trades", "pm_prices", "btc_klines"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _trades_path(self, condition_id: str) -> Path:
        return self.root / "pm_trades" / f"{condition_id}.parquet"

    def write_trades(self, condition_id: str, trades: list[PMTrade]) -> None:
        if not trades:
            return
        table = pa.table({
            "ts_ns":    [t.ts_ns for t in trades],
            "token_id": [t.token_id for t in trades],
            "side":     [t.side for t in trades],
            "price":    [t.price for t in trades],
            "size":     [t.size for t in trades],
        })
        pq.write_table(table, self._trades_path(condition_id))

    def read_trades(self, condition_id: str) -> list[PMTrade]:
        path = self._trades_path(condition_id)
        if not path.exists():
            return []
        table = pq.read_table(path)
        d = table.to_pylist()
        return [PMTrade(**row) for row in d]

    @property
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        return json.loads(self._manifest_path.read_text())

    def update_manifest(
        self,
        *,
        condition_id: str,
        n_rows: int,
        last_pull_ts_ns: int,
        market: "PMMarket | None" = None,
    ) -> None:
        m = self._load_manifest()
        entry = {"n_rows": n_rows, "last_pull_ts_ns": last_pull_ts_ns}
        if market is not None:
            entry["market"] = market.model_dump()
        m[condition_id] = entry
        self._manifest_path.write_text(json.dumps(m, indent=2))

    def get_manifest(self, condition_id: str) -> dict:
        return self._load_manifest().get(condition_id, {})

    def get_market(self, condition_id: str) -> "PMMarket | None":
        entry = self.get_manifest(condition_id)
        raw = entry.get("market")
        if not raw:
            return None
        return PMMarket(**raw)

    def manifest_keys(self) -> list[str]:
        return list(self._load_manifest().keys())
