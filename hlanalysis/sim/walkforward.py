from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def walk_forward_splits(
    items: Sequence[T],
    *,
    train: int,
    test: int,
    step: int,
    drop_short_tail: bool = False,
) -> Iterator[tuple[list[T], list[T]]]:
    """Yield (train_window, test_window) tuples sliding by `step`. Items must already
    be ordered chronologically by the caller.

    drop_short_tail=True: only yield windows where len(test_window) == test.
    drop_short_tail=False (default): also yield the final partial test window if non-empty.
    """
    n = len(items)
    start = 0
    while start + train <= n:
        tr = list(items[start : start + train])
        te = list(items[start + train : start + train + test])
        if not te:
            break
        if drop_short_tail and len(te) < test:
            break
        yield tr, te
        start += step
