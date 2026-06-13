from __future__ import annotations

import json

import pytest

from hlanalysis import _fastjson


@pytest.mark.parametrize(
    "raw",
    [
        '{"channel":"l2Book","data":{"coin":"#150","levels":[[["0.91","12"]],[["0.93","8"]]]}}',
        '[{"asset_id":"123","price":"0.5","size":"100"}]',
        '{"e":"trade","p":"68000.5","q":"0.01","T":1717500000123}',
        '{"a":null,"b":true,"c":1,"d":1.5,"e":"x","f":[1,2,3]}',
    ],
)
def test_decode_matches_json_loads(raw):
    assert _fastjson.decode(raw) == json.loads(raw)


def test_decode_accepts_bytes():
    raw = b'{"k":1}'
    assert _fastjson.decode(raw) == {"k": 1}
