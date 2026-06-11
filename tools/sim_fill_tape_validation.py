"""Independent sim-fill validation against the recorded HL trade tape (SHR-106).

The standing live-vs-sim comparison (``tools/_compare_ioc.py``) joins sim fills
to the ``user_fills`` CSV / settlement PnL. That tells us whether the *aggregate*
PnL matches, but it cannot tell us whether an individual sim fill was even
*possible* — the sim crosses whatever depth the recorded **order book** displayed,
and the displayed book is not always hittable. The recorded **trade tape** is the
independent ground truth: every on-venue print with price, size, and aggressor
side. If the sim fills 500 shares at 0.90 but the tape shows nobody ever swept
that level in the fill's window, that fill is *phantom liquidity* — depth the book
showed but no taker took.

This module is the reusable tape-side check, sitting alongside ``_compare_ioc.py``
so future fill-model changes are scored against what actually traded, not just the
CSV. The matching / aggregation core is pure (no IO) and unit-tested; the IO layer
loads the recorder's Hive-partitioned ``event=trade`` parquet via the same
conventions as ``hlanalysis.backtest.data.hl_hip4``.

Phantom test, per sim fill:

    hittable = Σ size of real prints in [t − w, t + w] that are at-or-through the
               limit AND (default) on the same aggressor side as the sim fill
    phantom  = sim_fill_size > hittable          (excess = sim_fill_size − hittable)

A fill in the open-burst (no print anywhere near it) has ``hittable == 0`` and is
phantom by definition: the displayed book absorbed our clip but no taker traded.

Notes / honest limitations:
  * Overlapping per-fill windows can count the same real print against two sim
    fills (the doom-loop re-fires densely). That makes the aggregate phantom
    excess *conservative* (a real print excuses both fills), so we under-claim,
    never over-claim, phantom size.
  * A symbol with no recorded prints at all is reported as ``no_tape`` (distinct
    from "had prints, none in window") so a recording gap is never silently
    counted as phantom liquidity.

Run (single arm):
    HLBT_HL_DATA_ROOT=/path/to/data uv run python tools/sim_fill_tape_validation.py \
        --prefix ioc_ --window-seconds 1.0 \
        --report docs/research/<date>-sim-fill-tape-validation.md \
        --out data/sim/runs/_tape_verdicts_ioc.parquet

Run (A/B two fill-model arms against the tape):
    ... --prefix ioc_ --baseline-prefix base_
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Matrix definition — the 16 cells (4 settle-days × {v1,v31} × {binary,bucket}).
# Mirrors tools/_compare_ioc.py so the two checks describe the same matrix.
# --------------------------------------------------------------------------- #
DAYS: dict[str, dict] = {
    "0606": dict(binary=["#1590", "#1591"],
                 bucket=["#1610", "#1611", "#1620", "#1621", "#1630", "#1631"],
                 w=("2026-06-05T07:00:00Z", "2026-06-06T07:00:00Z")),
    "0607": dict(binary=["#1640", "#1641"],
                 bucket=["#1660", "#1661", "#1670", "#1671", "#1680", "#1681"],
                 w=("2026-06-06T07:00:00Z", "2026-06-07T07:00:00Z")),
    "0608": dict(binary=["#2200", "#2201"],
                 bucket=["#2220", "#2221", "#2230", "#2231", "#2240", "#2241"],
                 w=("2026-06-07T07:00:00Z", "2026-06-08T07:00:00Z")),
    "0609": dict(binary=["#2250", "#2251"],
                 bucket=["#2270", "#2271", "#2280", "#2281", "#2290", "#2291"],
                 w=("2026-06-08T07:00:00Z", "2026-06-09T07:00:00Z")),
}

_HL_PREDICTION_PATH = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
_DEFAULT_WINDOW_NS = 1_000_000_000  # ±1.0 s
_PRICE_TOL = 1e-9
_SIZE_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FillRecord:
    """A single sim IOC fill (a non-settle, non-hedge row of fills.parquet)."""
    cloid: str
    ts_ns: int
    side: str  # "buy" | "sell"
    price: float
    size: float
    symbol: str
    question_id: str


@dataclass(frozen=True)
class FillVerdict:
    """The tape verdict for one sim fill.

    Two views of "how much really traded":
      * ``hittable_size`` — Σ size of all in-window at-or-through prints. A *raw*
        window sum: when sim re-fires densely, two fills can both see (and be
        excused by) the same print. Diagnostic only.
      * ``tape_filled`` — volume credited by the greedy ledger, where each real
        share is consumed by at most one sim fill (earliest first). This is what
        drives ``phantom_excess`` so overlapping re-fires can't double-spend the
        same liquidity.
    """
    cloid: str
    symbol: str
    question_id: str
    side: str
    price: float
    size: float
    ts_ns: int
    n_prints: int           # in-window at-or-through prints
    hittable_size: float    # raw windowed sum of their size (double-countable)
    hittable_notional: float
    tape_filled: float      # ledger-credited real volume (≤ size, not double-counted)
    tape_covered: bool      # symbol had ANY recorded print (else recording gap)
    is_phantom: bool        # tape_filled < size
    phantom_excess: float   # size − tape_filled
    phantom_notional: float  # phantom_excess × price


@dataclass(frozen=True)
class CellAgg:
    """Per-cell rollup of fill verdicts."""
    n_fills: int
    n_phantom: int
    n_no_tape: int
    sim_size: float
    sim_notional: float
    hittable_size: float
    phantom_excess_size: float
    phantom_notional: float

    @property
    def phantom_fill_frac(self) -> float:
        return self.n_phantom / self.n_fills if self.n_fills else 0.0

    @property
    def phantom_size_frac(self) -> float:
        return self.phantom_excess_size / self.sim_size if self.sim_size else 0.0

    @property
    def phantom_notional_frac(self) -> float:
        return self.phantom_notional / self.sim_notional if self.sim_notional else 0.0


# --------------------------------------------------------------------------- #
# Pure matching / aggregation core (unit-tested, no IO)
# --------------------------------------------------------------------------- #
def hittable_prints(
    prints: pd.DataFrame,
    *,
    side: str,
    price: float,
    fill_ts_ns: int,
    window_ns: int,
    price_tol: float = _PRICE_TOL,
    match_aggressor: bool = True,
) -> pd.DataFrame:
    """Real prints that compete for the same liquidity as a sim fill.

    Keeps prints within ``[fill_ts_ns ± window_ns]`` that are at-or-through the
    limit — ``price <= limit`` for a BUY (we lift asks at or below our limit),
    ``price >= limit`` for a SELL. When ``match_aggressor`` (default), also
    requires the print's aggressor side to match the sim fill's direction: a sim
    BUY is only "covered" by a real taker who also bought (consumed ask depth).
    """
    if prints.empty:
        return prints
    lo, hi = fill_ts_ns - window_ns, fill_ts_ns + window_ns
    m = (prints["ts_ns"] >= lo) & (prints["ts_ns"] <= hi)
    if side == "buy":
        m &= prints["price"] <= price + price_tol
    else:
        m &= prints["price"] >= price - price_tol
    if match_aggressor:
        m &= prints["side"] == side
    return prints[m]


def classify_fill(
    fill: FillRecord,
    prints: pd.DataFrame,
    *,
    window_ns: int,
    price_tol: float = _PRICE_TOL,
    match_aggressor: bool = True,
) -> FillVerdict:
    """Score ONE sim fill against its symbol's full trade tape, independently.

    The "independent" view: this fill is credited the *entire* windowed at-or-
    through volume (``tape_filled = min(size, hittable_size)``). For a cell with
    densely re-fired fills use :func:`ledger_verdicts` instead, which stops two
    fills from each claiming the same real print. The two agree for an isolated
    fill.

    ``prints`` is the entire recorded tape for ``fill.symbol`` (windowing happens
    here). An empty frame means the symbol was never seen on the tape
    (``tape_covered=False``); a non-empty frame with nothing in the window means
    the level was displayed but untraded at that moment.
    """
    tape_covered = not prints.empty
    hp = hittable_prints(
        prints, side=fill.side, price=fill.price, fill_ts_ns=fill.ts_ns,
        window_ns=window_ns, price_tol=price_tol, match_aggressor=match_aggressor,
    )
    hittable_size = float(hp["size"].sum()) if not hp.empty else 0.0
    hittable_notional = float((hp["size"] * hp["price"]).sum()) if not hp.empty else 0.0
    tape_filled = min(fill.size, hittable_size)
    return _verdict(fill, n_prints=int(len(hp)), hittable_size=hittable_size,
                    hittable_notional=hittable_notional, tape_filled=tape_filled,
                    tape_covered=tape_covered)


def ledger_verdicts(
    fills: Sequence[FillRecord],
    prints: pd.DataFrame,
    *,
    window_ns: int,
    price_tol: float = _PRICE_TOL,
    match_aggressor: bool = True,
) -> list[FillVerdict]:
    """Score every fill of ONE symbol with a greedy volume ledger.

    Fills are processed earliest-first; each consumes real print volume in its
    window (at-or-through, aggressor-matched), and consumed volume is not
    available to a later fill. So when the strategy re-fires across two snapshots
    of the same fleeting level, the first fill is credited the real volume and
    the second is correctly flagged phantom — matching the on-tape truth (e.g.
    #2230: 2×100 sim vs 100 traded → 100 phantom, not 0).
    """
    tape_covered = not prints.empty
    fills_sorted = sorted(fills, key=lambda f: f.ts_ns)
    if prints.empty:
        return [
            _verdict(f, n_prints=0, hittable_size=0.0, hittable_notional=0.0,
                     tape_filled=0.0, tape_covered=False)
            for f in fills_sorted
        ]
    prints = prints.reset_index(drop=True)
    remaining = prints["size"].to_numpy(dtype=float).copy()
    out: list[FillVerdict] = []
    for f in fills_sorted:
        hp = hittable_prints(
            prints, side=f.side, price=f.price, fill_ts_ns=f.ts_ns,
            window_ns=window_ns, price_tol=price_tol, match_aggressor=match_aggressor,
        )
        hittable_size = float(hp["size"].sum()) if not hp.empty else 0.0
        hittable_notional = float((hp["size"] * hp["price"]).sum()) if not hp.empty else 0.0
        need = f.size
        taken = 0.0
        for i in hp.index:  # ascending ts (prints sorted by ts on load)
            if need <= _SIZE_TOL:
                break
            avail = remaining[i]
            if avail <= 0.0:
                continue
            take = min(need, avail)
            remaining[i] -= take
            need -= take
            taken += take
        out.append(_verdict(
            f, n_prints=int(len(hp)), hittable_size=hittable_size,
            hittable_notional=hittable_notional, tape_filled=taken,
            tape_covered=tape_covered,
        ))
    return out


def _verdict(
    fill: FillRecord, *, n_prints: int, hittable_size: float,
    hittable_notional: float, tape_filled: float, tape_covered: bool,
) -> FillVerdict:
    phantom_excess = max(0.0, fill.size - tape_filled)
    return FillVerdict(
        cloid=fill.cloid, symbol=fill.symbol, question_id=fill.question_id,
        side=fill.side, price=fill.price, size=fill.size, ts_ns=fill.ts_ns,
        n_prints=n_prints, hittable_size=hittable_size,
        hittable_notional=hittable_notional, tape_filled=tape_filled,
        tape_covered=tape_covered, is_phantom=phantom_excess > _SIZE_TOL,
        phantom_excess=phantom_excess, phantom_notional=phantom_excess * fill.price,
    )


def aggregate(verdicts: Sequence[FillVerdict]) -> CellAgg:
    """Roll up fill verdicts into a per-cell summary."""
    return CellAgg(
        n_fills=len(verdicts),
        n_phantom=sum(1 for v in verdicts if v.is_phantom),
        n_no_tape=sum(1 for v in verdicts if not v.tape_covered),
        sim_size=sum(v.size for v in verdicts),
        sim_notional=sum(v.size * v.price for v in verdicts),
        hittable_size=sum(v.hittable_size for v in verdicts),
        phantom_excess_size=sum(v.phantom_excess for v in verdicts),
        phantom_notional=sum(v.phantom_notional for v in verdicts),
    )


# --------------------------------------------------------------------------- #
# IO layer
# --------------------------------------------------------------------------- #
def load_sim_fills(run_dir: Path | str) -> list[FillRecord]:
    """Load the real-order fills from a sim run's ``fills.parquet``.

    Excludes the synthetic settlement rows (``cloid == "settle"``) and hedge legs
    (``is_hedge``), which never appear on the HL prediction trade tape.
    """
    fp = Path(run_dir) / "fills.parquet"
    if not fp.exists():
        return []
    df = pd.read_parquet(fp)
    if df.empty:
        return []
    df = df[(df["cloid"] != "settle") & (~df["is_hedge"].astype(bool))]
    return [
        FillRecord(
            cloid=str(r.cloid), ts_ns=int(r.ts_ns), side=str(r.side),
            price=float(r.price), size=float(r.size), symbol=str(r.symbol),
            question_id=str(r.question_id),
        )
        for r in df.itertuples(index=False)
    ]


def make_tape_loader(data_root: Path | str) -> Callable[[str], pd.DataFrame]:
    """Build a memoized ``symbol -> prints DataFrame`` loader over the recorder
    trade partition. Returns an empty frame for symbols with no recorded tape."""
    root = Path(data_root)
    cache: dict[str, pd.DataFrame] = {}

    cols = ["exchange_ts", "price", "size", "side"]

    def load(symbol: str) -> pd.DataFrame:
        if symbol in cache:
            return cache[symbol]
        pat = str(root / _HL_PREDICTION_PATH / "event=trade" / f"symbol={symbol}" / "**" / "*.parquet")
        files = sorted(_glob.glob(pat, recursive=True))
        # Read only the four data columns per file: the recorder's Hive partition
        # column ``hour`` round-trips as int64 in some files and string in others,
        # so a whole-schema concat fails — we never need it here anyway.
        parts = [pq.ParquetFile(f).read(columns=cols).to_pandas() for f in files]
        if not parts:
            df = pd.DataFrame(columns=["ts_ns", "price", "size", "side"])
        else:
            raw = pd.concat(parts, ignore_index=True)
            df = pd.DataFrame({
                "ts_ns": raw["exchange_ts"].astype("int64"),
                "price": raw["price"].astype(float),
                "size": raw["size"].astype(float),
                # Recorder writes "buy"/"sell"/"unknown" aggressor; collapse
                # "unknown" to "buy" to match hl_hip4's TradeEvent narrowing.
                "side": raw["side"].where(raw["side"] == "sell", "buy"),
            }).sort_values("ts_ns").reset_index(drop=True)
        cache[symbol] = df
        return df

    return load


def validate_cell(
    fills: Sequence[FillRecord],
    tape_loader: Callable[[str], pd.DataFrame],
    *,
    window_ns: int,
    price_tol: float = _PRICE_TOL,
    match_aggressor: bool = True,
) -> tuple[CellAgg, list[FillVerdict]]:
    """Validate every fill in a cell against the tape; return rollup + verdicts.

    Fills are grouped per symbol and scored with the greedy :func:`ledger_verdicts`
    so re-fires across the same symbol can't double-spend the same real liquidity.
    """
    from collections import defaultdict

    by_symbol: dict[str, list[FillRecord]] = defaultdict(list)
    for f in fills:
        by_symbol[f.symbol].append(f)
    verdicts: list[FillVerdict] = []
    for symbol, sym_fills in by_symbol.items():
        verdicts.extend(ledger_verdicts(
            sym_fills, tape_loader(symbol), window_ns=window_ns,
            price_tol=price_tol, match_aggressor=match_aggressor,
        ))
    return aggregate(verdicts), verdicts


def run_matrix(
    *,
    runs_dir: Path | str,
    prefix: str,
    tape_loader: Callable[[str], pd.DataFrame],
    window_ns: int,
    price_tol: float = _PRICE_TOL,
    match_aggressor: bool = True,
) -> tuple[dict[tuple[str, str, str], CellAgg | None], list[FillVerdict]]:
    """Validate every cell of the matrix for one run ``prefix``.

    Returns ``{(day, slot, kind): CellAgg | None}`` (``None`` = run dir missing)
    and the flat list of all fill verdicts (each tagged via its symbol → cell).
    """
    runs_dir = Path(runs_dir)
    cells: dict[tuple[str, str, str], CellAgg | None] = {}
    all_verdicts: list[FillVerdict] = []
    for day in DAYS:
        for slot in ("v1", "v31"):
            for kind in ("binary", "bucket"):
                run_dir = runs_dir / f"{prefix}{slot}_{kind}_{day}"
                if not (run_dir / "fills.parquet").exists():
                    cells[(day, slot, kind)] = None
                    continue
                fills = load_sim_fills(run_dir)
                agg, verdicts = validate_cell(
                    fills, tape_loader, window_ns=window_ns,
                    price_tol=price_tol, match_aggressor=match_aggressor,
                )
                cells[(day, slot, kind)] = agg
                all_verdicts.extend(verdicts)
    return cells, all_verdicts


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def format_matrix(cells: dict[tuple[str, str, str], CellAgg | None], *, prefix: str) -> str:
    hdr = (f"{'day':5} {'slot':4} {'kind':7} | {'fills':>5} {'phantom':>7} "
           f"{'no_tape':>7} | {'simSz':>9} {'hitSz':>9} {'phSz':>9} | "
           f"{'ph$':>9} | {'ph%fill':>7} {'ph%sz':>7}")
    lines = [f"=== tape phantom-fill exposure — arm '{prefix}' ===", hdr, "-" * len(hdr)]
    tot = dict(n=0, ph=0, nt=0, sz=0.0, hit=0.0, phsz=0.0, ph_ntl=0.0, ntl=0.0)
    missing = 0
    for day in DAYS:
        for slot in ("v1", "v31"):
            for kind in ("binary", "bucket"):
                agg = cells.get((day, slot, kind))
                if agg is None:
                    missing += 1
                    lines.append(f"{day:5} {slot:4} {kind:7} |  MISSING")
                    continue
                lines.append(
                    f"{day:5} {slot:4} {kind:7} | {agg.n_fills:5d} {agg.n_phantom:7d} "
                    f"{agg.n_no_tape:7d} | {agg.sim_size:9.0f} {agg.hittable_size:9.0f} "
                    f"{agg.phantom_excess_size:9.0f} | {agg.phantom_notional:9.2f} | "
                    f"{agg.phantom_fill_frac*100:6.1f}% {agg.phantom_size_frac*100:6.1f}%"
                )
                tot["n"] += agg.n_fills
                tot["ph"] += agg.n_phantom
                tot["nt"] += agg.n_no_tape
                tot["sz"] += agg.sim_size
                tot["hit"] += agg.hittable_size
                tot["phsz"] += agg.phantom_excess_size
                tot["ph_ntl"] += agg.phantom_notional
                tot["ntl"] += agg.sim_notional
    lines.append("-" * len(hdr))
    ph_fill = tot["ph"] / tot["n"] if tot["n"] else 0.0
    ph_sz = tot["phsz"] / tot["sz"] if tot["sz"] else 0.0
    lines.append(
        f"{'TOTAL':5} {'':4} {'':7} | {tot['n']:5d} {tot['ph']:7d} {tot['nt']:7d} | "
        f"{tot['sz']:9.0f} {tot['hit']:9.0f} {tot['phsz']:9.0f} | {tot['ph_ntl']:9.2f} | "
        f"{ph_fill*100:6.1f}% {ph_sz*100:6.1f}%"
    )
    if missing:
        lines.append(f"\n[{missing} cell(s) missing — run tools/_run_ioc_matrix.sh first]")
    return "\n".join(lines)


def verdicts_to_frame(verdicts: Sequence[FillVerdict]) -> pd.DataFrame:
    return pd.DataFrame([vars(v) for v in verdicts])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    import os

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs-dir", default="data/sim/runs", help="dir of sim run outputs")
    p.add_argument("--prefix", default="ioc_", help="run-dir prefix for the arm to validate")
    p.add_argument("--baseline-prefix", default=None,
                   help="optional second arm to validate side-by-side (e.g. base_)")
    p.add_argument("--data-root", default=os.environ.get("HLBT_HL_DATA_ROOT", "data"),
                   help="recorder data root (env HLBT_HL_DATA_ROOT)")
    p.add_argument("--window-seconds", type=float, default=1.0,
                   help="± match window around each fill in seconds (default 1.0)")
    p.add_argument("--price-tol", type=float, default=_PRICE_TOL,
                   help="price equality tolerance for at-or-through (default 1e-9)")
    p.add_argument("--no-match-aggressor", action="store_true",
                   help="count prints on EITHER aggressor side (looser; default matches side)")
    p.add_argument("--out", default=None, help="write per-fill verdicts parquet here")
    p.add_argument("--report", default=None, help="write a markdown summary here")
    return p


def _run_one(args, prefix: str, tape_loader) -> tuple[str, list[FillVerdict]]:
    cells, verdicts = run_matrix(
        runs_dir=args.runs_dir, prefix=prefix, tape_loader=tape_loader,
        window_ns=int(args.window_seconds * 1e9), price_tol=args.price_tol,
        match_aggressor=not args.no_match_aggressor,
    )
    return format_matrix(cells, prefix=prefix), verdicts


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tape_loader = make_tape_loader(args.data_root)
    prefixes = [args.prefix] + ([args.baseline_prefix] if args.baseline_prefix else [])
    blocks: list[str] = []
    frames: list[pd.DataFrame] = []
    for pfx in prefixes:
        block, verdicts = _run_one(args, pfx, tape_loader)
        blocks.append(block)
        if verdicts:
            fr = verdicts_to_frame(verdicts)
            fr.insert(0, "arm", pfx)
            frames.append(fr)
    text = "\n\n".join(blocks)
    print(text)
    if args.out and frames:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(args.out, index=False)
        print(f"\nwrote per-fill verdicts → {args.out}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("```\n" + text + "\n```\n")
        print(f"wrote report → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
