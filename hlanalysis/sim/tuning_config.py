from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TuningConfig:
    v1_grid: dict[str, list[Any]] = field(default_factory=dict)
    v2_grid: dict[str, list[Any]] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)


def load_tuning_yaml(path: Path) -> TuningConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return TuningConfig(
        v1_grid=raw.get("v1_grid", {}),
        v2_grid=raw.get("v2_grid", {}),
        run=raw.get("run", {}),
    )
