from .base import Strategy
from .types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    Position,
)

# Side-effect imports: each module's tail calls `@register(...)`, so simply
# importing them populates the backtest strategy registry. `hl-bt` and the
# tuning pipeline rely on this so they can resolve strategy ids without
# bespoke imports.
from . import late_resolution  # noqa: F401
from . import model_edge  # noqa: F401

__all__ = [
    "Action",
    "BookState",
    "Decision",
    "Diagnostic",
    "OrderIntent",
    "Position",
    "Strategy",
]
