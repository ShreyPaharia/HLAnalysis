from __future__ import annotations

import json
import time

from hlanalysis import _fastjson

_FRAME = (
    '{"channel":"l2Book","data":{"coin":"#150","time":1717500000123,'
    '"levels":[[["0.91","12"],["0.90","30"]],[["0.93","8"],["0.94","25"]]]}}'
)


def test_fastjson_is_not_slower(capsys):
    n = 50_000
    t0 = time.perf_counter()
    for _ in range(n):
        json.loads(_FRAME)
    t_json = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        _fastjson.decode(_FRAME)
    t_ms = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\njson.loads: {t_json*1e6/n:.2f} us/frame | "
              f"msgspec: {t_ms*1e6/n:.2f} us/frame | "
              f"speedup: {t_json/t_ms:.2f}x")
    # Lenient guard: msgspec must not be materially slower (CI jitter tolerant).
    assert t_ms <= t_json * 1.5
