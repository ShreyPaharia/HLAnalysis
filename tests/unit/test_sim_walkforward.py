from __future__ import annotations

from hlanalysis.sim.walkforward import walk_forward_splits


def test_splits_cover_data_with_correct_window():
    items = list(range(100))
    splits = list(walk_forward_splits(items, train=60, test=15, step=15))
    assert len(splits) == 3  # (0..60, 60..75), (15..75, 75..90), (30..90, 90..100 truncated→ ok if last test < 15? we drop)
    train, test = splits[0]
    assert len(train) == 60
    assert len(test) == 15
    assert items[60:75] == test


def test_splits_drop_short_tail():
    items = list(range(100))
    splits = list(walk_forward_splits(items, train=60, test=15, step=15, drop_short_tail=True))
    for tr, te in splits:
        assert len(te) == 15
