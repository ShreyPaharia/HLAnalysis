"""SHR-104 — measure displayed-vs-filled per order from the recorded HL trade tape.

ANALYSIS-ONLY. Places no orders, mutates nothing. The decisive first step of the
SHR-103 fill-model epic: our own wallet's prints appear in the recorded HL trade
tape (the recorder writes ``buyer``/``seller`` addresses on every print), so we can
measure — *per marketable IOC order* — what fraction of the **displayed top-of-book
size at our limit** we actually filled live. The depth-limited sim assumes that
fraction is 1.0 (it dumps the whole displayed ladder at the touch); this tool
measures the real number from ground truth.

What it does
------------
1. Reads our maker/taker wallet addresses (per slot) and the leg→kind map from the
   committed ``user_fills`` CSV (the ``DIAG`` / ``KLASS`` sections written by
   ``tools/dump_hl_fills_all.py``). The CSV ``coin`` field is also authoritative for
   *which* leg each slot traded — this matters because HL records every binary trade
   on BOTH the YES and NO (complement) books with complementary prices and distinct
   ``trade_id``s, so naively scanning both legs double-counts. We key extraction on
   the canonical legs that actually appear in the CSV.
2. For each canonical (slot, leg), extracts OUR taker prints from the recorded trade
   stream (``side`` is the aggressor; engine fires IOC so we are always the taker),
   clusters consecutive prints into orders, and joins each order back to the L2 book
   snapshot immediately *before* the order to read the displayed depth at decision
   time. The engine's IOC limit IS the touch (``hl_client.py``), so "displayed size
   at our limit" == displayed size at the best level.
3. Reports the displayed-vs-filled ratio distribution (p10 / median / p90), split by
   slot, kind (binary vs bucket), and book width.
4. Emits a MARKET clip-size distribution that EXCLUDES every trade we participated in
   (on either side), for the downstream SHR-105 calibration — avoids circularity.

Run (read-only) against the recorded corpus::

    python tools/hl_own_fills_displayed_vs_filled.py \
        --data-root /path/to/data \
        --fills-csv docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv \
        --out-orders docs/research/2026-06-11-hl-own-fills-displayed-vs-filled.csv \
        --out-market docs/research/2026-06-11-hl-market-clips-ex-own.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

# Float tolerance for matching a price level against a limit. HL prices are
# recorded to ~5 dp; 1e-9 cleanly separates distinct levels without dropping the
# limit level itself to rounding.
PRICE_EPS = 1e-9

# Order-clustering gap. Consecutive prints within this gap (and on the same side)
# are one marketable IOC; intra-order sub-fills share an exchange_ts (gap 0) while
# distinct re-fired orders are >= ~0.73 s apart (the measured live re-fire floor,
# docs/research/2026-06-11-ioc-refire-floor-hl-fill-model.md). 0.5 s sits cleanly
# in that empty band.
DEFAULT_ORDER_GAP_NS = 500_000_000

# Book-width buckets (best_ask - best_bid). Binary legs run ~0.001-0.009 (tight);
# bucket legs run ~0.13-0.34 (wide); "mid" catches the transition.
WIDTH_TIGHT_MAX = 0.02
WIDTH_WIDE_MIN = 0.10

Side = Literal["buy", "sell"]


# ---------------------------------------------------------------------------
# CSV (addresses + leg->kind + cross-check fills) — pure parsing
# ---------------------------------------------------------------------------


def normalize_kind(klass: str) -> str:
    """``priceBinary`` -> ``binary``; ``priceBucket`` -> ``bucket``; else passthrough."""
    k = (klass or "").strip()
    if k == "priceBinary":
        return "binary"
    if k == "priceBucket":
        return "bucket"
    return k


@dataclass(frozen=True)
class CsvFill:
    slot: str
    ts_ns: int
    coin: str
    dir: str  # Buy / Sell / Settlement
    side: str  # buy / sell
    px: float
    sz: float
    cloid: str


@dataclass
class FillsMeta:
    # lowercased wallet address -> slot
    addr_to_slot: dict[str, str] = field(default_factory=dict)
    # leg symbol (e.g. "#1670") -> kind ("binary" / "bucket")
    leg_to_kind: dict[str, str] = field(default_factory=dict)
    fills: list[CsvFill] = field(default_factory=list)

    def slot_addr(self) -> dict[str, str]:
        """slot -> lowercased address."""
        return {slot: addr for addr, slot in self.addr_to_slot.items()}

    def traded_legs(self) -> set[tuple[str, str]]:
        """(slot, leg) pairs that the CSV records a non-settlement fill on — the
        canonical legs (avoids the YES/NO mirror double-count)."""
        return {
            (f.slot, f.coin) for f in self.fills if f.dir.lower() != "settlement"
        }


def parse_fills_csv(text: str) -> FillsMeta:
    """Parse the ``dump_hl_fills_all.py`` CSV (DIAG / FILL / KLASS sections)."""
    meta = FillsMeta()
    for raw in text.splitlines():
        row = raw.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split(",")]
        tag = parts[0]
        if tag == "DIAG" and len(parts) >= 3:
            slot, addr = parts[1], parts[2]
            if addr and addr.upper() != "ERR" and addr.startswith("0x"):
                meta.addr_to_slot[addr.lower()] = slot
        elif tag == "KLASS" and len(parts) >= 4:
            _slot, coin, klass = parts[1], parts[2], parts[3]
            meta.leg_to_kind[coin] = normalize_kind(klass)
        elif tag == "FILL" and len(parts) >= 11:
            try:
                meta.fills.append(
                    CsvFill(
                        slot=parts[1],
                        ts_ns=int(parts[2]),
                        coin=parts[3],
                        dir=parts[4],
                        side=parts[5],
                        px=float(parts[6]),
                        sz=float(parts[7]),
                        cloid=parts[10],
                    )
                )
            except (ValueError, IndexError):
                continue
    return meta


# ---------------------------------------------------------------------------
# Trade-tape classification + order clustering — pure
# ---------------------------------------------------------------------------


def is_our_taker(side: str, buyer: str, seller: str, addr: str) -> bool:
    """True iff ``addr`` is the *aggressor* of this print.

    ``side`` is the aggressor side (recorder convention). A buy-aggressor print
    has the taker as ``buyer``; a sell-aggressor print has the taker as ``seller``.
    The engine fires IOC-at-touch, so our fills are always taker — a maker print
    of ours (addr on the passive side) would be a data anomaly worth flagging.
    """
    a = addr.lower()
    b = (buyer or "").lower()
    s = (seller or "").lower()
    if side == "buy":
        return b == a
    if side == "sell":
        return s == a
    return False


@dataclass(frozen=True)
class Print:
    ts_ns: int
    side: Side
    px: float
    sz: float


def cluster_orders(prints: Sequence[Print], max_gap_ns: int) -> list[list[Print]]:
    """Group time-sorted prints into orders: a new order starts when the gap to the
    previous print exceeds ``max_gap_ns`` or the side flips."""
    clusters: list[list[Print]] = []
    cur: list[Print] = []
    for p in prints:
        if not cur:
            cur = [p]
            continue
        prev = cur[-1]
        if p.side == prev.side and (p.ts_ns - prev.ts_ns) <= max_gap_ns:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    if cur:
        clusters.append(cur)
    return clusters


# ---------------------------------------------------------------------------
# Book join + displayed-depth metrics — pure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookSnap:
    ts_ns: int
    bid_px: tuple[float, ...]
    bid_sz: tuple[float, ...]
    ask_px: tuple[float, ...]
    ask_sz: tuple[float, ...]


def book_before_index(snap_ts: Sequence[int], target_ns: int) -> int:
    """Index of the last snapshot with ``ts < target_ns`` (strictly before), or -1.

    ``snap_ts`` must be ascending. Strictly-before so the snapshot reflects the
    resting depth the order saw, not the post-trade book.
    """
    i = bisect.bisect_left(snap_ts, target_ns)
    return i - 1


@dataclass(frozen=True)
class DepthMetrics:
    best_bid: float
    best_ask: float
    width: float
    disp_top: float  # size at the touch (best level we hit)
    disp_at_limit: float  # total resting depth at-or-better than our limit


def displayed_metrics(book: BookSnap, side: Side, limit_px: float) -> DepthMetrics | None:
    """Displayed depth the order saw. ``side`` is our aggressor side.

    A buy lifts asks (best level = lowest ask); a sell hits bids (best = highest
    bid). ``disp_top`` is the touch size; ``disp_at_limit`` sums every level
    marketable at our limit. Returns None if the relevant side is empty.
    """
    best_bid = book.bid_px[0] if book.bid_px else float("nan")
    best_ask = book.ask_px[0] if book.ask_px else float("nan")
    width = (best_ask - best_bid) if (book.bid_px and book.ask_px) else float("nan")
    if side == "buy":
        if not book.ask_px:
            return None
        disp_top = book.ask_sz[0]
        disp_at_limit = sum(
            sz for px, sz in zip(book.ask_px, book.ask_sz) if px <= limit_px + PRICE_EPS
        )
    else:
        if not book.bid_px:
            return None
        disp_top = book.bid_sz[0]
        disp_at_limit = sum(
            sz for px, sz in zip(book.bid_px, book.bid_sz) if px >= limit_px - PRICE_EPS
        )
    return DepthMetrics(best_bid, best_ask, width, disp_top, disp_at_limit)


def width_bucket(width: float) -> str:
    if width != width:  # NaN
        return "unknown"
    if width < WIDTH_TIGHT_MAX:
        return "tight"
    if width >= WIDTH_WIDE_MIN:
        return "wide"
    return "mid"


@dataclass(frozen=True)
class OrderMetric:
    slot: str
    leg: str
    kind: str
    side: Side
    ts_ns: int
    n_prints: int
    filled: float
    limit_px: float
    decision_ts_ns: int
    best_bid: float
    best_ask: float
    width: float
    width_bucket: str
    disp_top: float
    disp_at_limit: float
    ratio_top: float | None  # filled / disp_top  (SHR-103 headline)
    ratio_at_limit: float | None  # filled / disp_at_limit


def order_metric(
    order: Sequence[Print],
    book: BookSnap,
    *,
    slot: str,
    leg: str,
    kind: str,
) -> OrderMetric | None:
    """Compute displayed-vs-filled for one clustered order against its decision book."""
    side: Side = order[0].side
    filled = sum(p.sz for p in order)
    # The IOC limit is unknowable from fills alone; the worst realized price is a
    # lower bound on it and the level the order demonstrably reached.
    limit_px = max(p.px for p in order) if side == "buy" else min(p.px for p in order)
    dm = displayed_metrics(book, side, limit_px)
    if dm is None:
        return None
    ratio_top = filled / dm.disp_top if dm.disp_top > 0 else None
    ratio_at_limit = filled / dm.disp_at_limit if dm.disp_at_limit > 0 else None
    return OrderMetric(
        slot=slot,
        leg=leg,
        kind=kind,
        side=side,
        ts_ns=order[0].ts_ns,
        n_prints=len(order),
        filled=filled,
        limit_px=limit_px,
        decision_ts_ns=book.ts_ns,
        best_bid=dm.best_bid,
        best_ask=dm.best_ask,
        width=dm.width,
        width_bucket=width_bucket(dm.width),
        disp_top=dm.disp_top,
        disp_at_limit=dm.disp_at_limit,
        ratio_top=ratio_top,
        ratio_at_limit=ratio_at_limit,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dist:
    n: int
    p10: float
    median: float
    p90: float

    @property
    def empty(self) -> bool:
        return self.n == 0


def _pct(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]) over an ascending list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(values: Iterable[float]) -> Dist:
    vals = sorted(v for v in values if v is not None and v == v)
    if not vals:
        return Dist(0, float("nan"), float("nan"), float("nan"))
    return Dist(
        n=len(vals),
        p10=_pct(vals, 0.10),
        median=statistics.median(vals),
        p90=_pct(vals, 0.90),
    )


# ---------------------------------------------------------------------------
# Data access (duckdb) — thin, untested I/O over the pure core above
# ---------------------------------------------------------------------------

_HL_PRED = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"


def _leg_glob(data_root: Path, event: str, leg: str) -> str:
    return str(data_root / _HL_PRED / f"event={event}" / f"symbol={leg}" / "**" / "*.parquet")


def _read_our_prints(con, data_root: Path, leg: str, addr: str) -> list[Print]:
    g = _leg_glob(data_root, "trade", leg)
    a = addr.lower()
    rows = con.sql(
        f"""
        SELECT exchange_ts, side, price, size
        FROM read_parquet('{g}', hive_partitioning=1)
        WHERE (side='buy'  AND lower(buyer)='{a}')
           OR (side='sell' AND lower(seller)='{a}')
        ORDER BY exchange_ts
        """
    ).fetchall()
    return [Print(int(ts), ("buy" if s == "buy" else "sell"), float(px), float(sz)) for ts, s, px, sz in rows]


def _read_books(con, data_root: Path, leg: str) -> list[BookSnap]:
    g = _leg_glob(data_root, "book_snapshot", leg)
    rows = con.sql(
        f"""
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{g}', hive_partitioning=1)
        ORDER BY exchange_ts
        """
    ).fetchall()
    out: list[BookSnap] = []
    for ts, bpx, bsz, apx, asz in rows:
        out.append(
            BookSnap(
                int(ts),
                tuple(bpx or []),
                tuple(bsz or []),
                tuple(apx or []),
                tuple(asz or []),
            )
        )
    return out


def _read_market_clips(con, data_root: Path, leg: str, our_addrs: set[str]) -> list[float]:
    """Per-print taker clip sizes EXCLUDING any trade we participated in (either
    side), to avoid circularity in the downstream calibration."""
    g = _leg_glob(data_root, "trade", leg)
    if our_addrs:
        excl = " OR ".join(
            f"lower(buyer)='{a}' OR lower(seller)='{a}'" for a in sorted(our_addrs)
        )
        where = f"WHERE NOT ({excl})"
    else:
        where = ""
    rows = con.sql(
        f"SELECT size FROM read_parquet('{g}', hive_partitioning=1) {where}"
    ).fetchall()
    return [float(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class Extraction:
    orders: list[OrderMetric]
    market_clips: dict[str, list[float]]  # leg -> per-print sizes (ex-own)
    own_clips: dict[tuple[str, str], list[float]]  # (slot,leg) -> our per-print sizes
    tape_counts: dict[tuple[str, str], tuple[int, int]]  # (slot,leg) -> (n_prints, n_orders)


def extract(
    data_root: Path,
    meta: FillsMeta,
    *,
    order_gap_ns: int = DEFAULT_ORDER_GAP_NS,
) -> Extraction:
    import duckdb

    slot_addr = meta.slot_addr()
    our_addrs = set(meta.addr_to_slot.keys())
    con = duckdb.connect()
    orders: list[OrderMetric] = []
    tape_counts: dict[tuple[str, str], tuple[int, int]] = {}
    market_clips: dict[str, list[float]] = {}
    own_clips: dict[tuple[str, str], list[float]] = {}

    legs_for_market: set[str] = set()
    for slot, leg in sorted(meta.traded_legs()):
        addr = slot_addr.get(slot)
        if addr is None:
            continue
        kind = meta.leg_to_kind.get(leg, "unknown")
        prints = _read_our_prints(con, data_root, leg, addr)
        books = _read_books(con, data_root, leg)
        book_ts = [b.ts_ns for b in books]
        clusters = cluster_orders(prints, order_gap_ns)
        tape_counts[(slot, leg)] = (len(prints), len(clusters))
        own_clips[(slot, leg)] = [p.sz for p in prints]
        for cl in clusters:
            idx = book_before_index(book_ts, cl[0].ts_ns)
            if idx < 0:
                continue
            om = order_metric(cl, books[idx], slot=slot, leg=leg, kind=kind)
            if om is not None:
                orders.append(om)
        legs_for_market.add(leg)

    for leg in sorted(legs_for_market):
        market_clips[leg] = _read_market_clips(con, data_root, leg, our_addrs)

    con.close()
    return Extraction(
        orders=orders,
        market_clips=market_clips,
        own_clips=own_clips,
        tape_counts=tape_counts,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(d: Dist) -> str:
    if d.empty:
        return f"{'n=0':>28}"
    return f"n={d.n:<4} p10={d.p10:6.3f} med={d.median:6.3f} p90={d.p90:6.3f}"


def render_report(ext: Extraction, meta: FillsMeta) -> str:
    out = io.StringIO()
    orders = ext.orders
    w = out.write

    w("# SHR-104 displayed-vs-filled (our HL fills, recorded trade tape)\n\n")
    w(f"Orders measured: {len(orders)}  (clustered IOCs joined to a pre-trade book)\n\n")

    # --- Headline ratio_top by slot x kind ---
    w("## filled / displayed-top-of-book (ratio_top) — SHR-103 headline\n\n")
    w(f"{'group':22} {'dist'}\n")
    w(f"{'ALL':22} {_fmt(summarize(o.ratio_top for o in orders))}\n")
    for slot in sorted({o.slot for o in orders}):
        for kind in sorted({o.kind for o in orders if o.slot == slot}):
            sub = [o.ratio_top for o in orders if o.slot == slot and o.kind == kind]
            w(f"{slot + '/' + kind:22} {_fmt(summarize(sub))}\n")
    w("\n")

    # --- by width bucket ---
    w("## ratio_top by book width\n\n")
    for wb in ("tight", "mid", "wide", "unknown"):
        sub = [o.ratio_top for o in orders if o.width_bucket == wb]
        if sub:
            w(f"{wb:22} {_fmt(summarize(sub))}\n")
    w("\n")

    # --- ratio_at_limit (full marketable ladder) ---
    w("## filled / displayed-at-limit (ratio_at_limit) — full marketable ladder\n\n")
    w(f"{'ALL':22} {_fmt(summarize(o.ratio_at_limit for o in orders))}\n")
    for kind in sorted({o.kind for o in orders}):
        sub = [o.ratio_at_limit for o in orders if o.kind == kind]
        w(f"{kind:22} {_fmt(summarize(sub))}\n")
    w("\n")

    # --- cross-check vs CSV ---
    w("## tape vs user_fills CSV cross-check (per slot/leg)\n\n")
    csv_counts: dict[tuple[str, str], tuple[int, int, float]] = {}
    for f in meta.fills:
        if f.dir.lower() == "settlement":
            continue
        k = (f.slot, f.coin)
        n, nc, sz = csv_counts.get(k, (0, 0, 0.0))
        csv_counts[k] = (n + 1, nc, sz + f.sz)
    cloids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in meta.fills:
        if f.dir.lower() != "settlement":
            cloids[(f.slot, f.coin)].add(f.cloid)
    w(f"{'slot/leg':16} {'kind':8} {'tape_prints':>11} {'tape_orders':>11} {'csv_fills':>9} {'csv_cloids':>10}\n")
    for k in sorted(ext.tape_counts):
        tp, to = ext.tape_counts[k]
        cf = csv_counts.get(k, (0, 0, 0.0))[0]
        cc = len(cloids.get(k, set()))
        kind = meta.leg_to_kind.get(k[1], "?")
        w(f"{k[0] + '/' + k[1]:16} {kind:8} {tp:11} {to:11} {cf:9} {cc:10}\n")
    w("\n")

    # --- our own per-print clip sizes (the real per-IOC clip, vs sim single dump) ---
    w("## OUR per-print clip sizes (one marketable IOC catches this much at once)\n\n")
    own_all: list[float] = []
    own_by_kind: dict[str, list[float]] = defaultdict(list)
    for (slot, leg), clips in sorted(ext.own_clips.items()):
        own_all.extend(clips)
        own_by_kind[meta.leg_to_kind.get(leg, "?")].extend(clips)
    w(f"{'group':16} {'n':>5} {'p10':>7} {'med':>7} {'p90':>7} {'max':>7}\n")
    for label, clips in (
        ("ALL", own_all),
        ("binary", own_by_kind.get("binary", [])),
        ("bucket", own_by_kind.get("bucket", [])),
    ):
        d = summarize(clips)
        mx = max(clips) if clips else float("nan")
        w(f"{label:16} {d.n:5} {d.p10:7.1f} {d.median:7.1f} {d.p90:7.1f} {mx:7.1f}\n")
    w("\n")

    # --- market clip distribution (ex-own) ---
    w("## MARKET clip-size distribution (per print, EXCLUDING our trades) — for SHR-105\n\n")
    all_clips: list[float] = []
    binary_clips: list[float] = []
    bucket_clips: list[float] = []
    w(f"{'leg':10} {'kind':8} {'n':>6} {'p10':>7} {'med':>7} {'p90':>7} {'max':>7}\n")
    for leg in sorted(ext.market_clips):
        clips = ext.market_clips[leg]
        all_clips.extend(clips)
        kind = meta.leg_to_kind.get(leg, "?")
        if kind == "binary":
            binary_clips.extend(clips)
        elif kind == "bucket":
            bucket_clips.extend(clips)
        d = summarize(clips)
        mx = max(clips) if clips else float("nan")
        w(f"{leg:10} {kind:8} {d.n:6} {d.p10:7.1f} {d.median:7.1f} {d.p90:7.1f} {mx:7.1f}\n")
    for label, clips in (("ALL", all_clips), ("binary", binary_clips), ("bucket", bucket_clips)):
        d = summarize(clips)
        mx = max(clips) if clips else float("nan")
        w(f"{label:10} {'':8} {d.n:6} {d.p10:7.1f} {d.median:7.1f} {d.p90:7.1f} {mx:7.1f}\n")

    return out.getvalue()


def write_orders_csv(path: Path, orders: Sequence[OrderMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "slot", "leg", "kind", "side", "ts_ns", "n_prints", "filled",
                "limit_px", "decision_ts_ns", "best_bid", "best_ask", "width",
                "width_bucket", "disp_top", "disp_at_limit", "ratio_top",
                "ratio_at_limit",
            ]
        )
        for o in orders:
            wr.writerow(
                [
                    o.slot, o.leg, o.kind, o.side, o.ts_ns, o.n_prints,
                    f"{o.filled:.4f}", f"{o.limit_px:.6f}", o.decision_ts_ns,
                    f"{o.best_bid:.6f}", f"{o.best_ask:.6f}", f"{o.width:.6f}",
                    o.width_bucket, f"{o.disp_top:.4f}", f"{o.disp_at_limit:.4f}",
                    "" if o.ratio_top is None else f"{o.ratio_top:.6f}",
                    "" if o.ratio_at_limit is None else f"{o.ratio_at_limit:.6f}",
                ]
            )


_SUMMARY_QS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]


def _clip_summary_row(group: str, kind: str, clips: list[float]) -> list:
    vals = sorted(clips)
    n = len(vals)
    qs = [_pct(vals, q) for q in _SUMMARY_QS]
    mx = vals[-1] if vals else float("nan")
    mean = (sum(vals) / n) if n else float("nan")
    return [group, kind, n, *[f"{q:.2f}" for q in qs], f"{mx:.2f}", f"{mean:.2f}"]


def write_market_summary_csv(
    path: Path, market_clips: dict[str, list[float]], meta: FillsMeta
) -> None:
    """Compact per-leg / per-kind clip-size percentile summary — the calibration
    input handed to SHR-105. The full per-print distribution is large and fully
    reproducible from the corpus (use --out-market-raw); this summary is what gets
    committed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["group", "kind", "n", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean"]
    by_kind: dict[str, list[float]] = defaultdict(list)
    all_clips: list[float] = []
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for leg in sorted(market_clips):
            kind = meta.leg_to_kind.get(leg, "?")
            wr.writerow(_clip_summary_row(leg, kind, market_clips[leg]))
            by_kind[kind].extend(market_clips[leg])
            all_clips.extend(market_clips[leg])
        for kind in sorted(by_kind):
            wr.writerow(_clip_summary_row(f"ALL_{kind}", kind, by_kind[kind]))
        wr.writerow(_clip_summary_row("ALL", "", all_clips))


def write_market_raw_csv(path: Path, market_clips: dict[str, list[float]], meta: FillsMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["leg", "kind", "size"])
        for leg in sorted(market_clips):
            kind = meta.leg_to_kind.get(leg, "?")
            for sz in market_clips[leg]:
                wr.writerow([leg, kind, f"{sz:.4f}"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path, help="recorded corpus root")
    ap.add_argument("--fills-csv", required=True, type=Path, help="dump_hl_fills_all.py CSV")
    ap.add_argument("--order-gap-ns", type=int, default=DEFAULT_ORDER_GAP_NS)
    ap.add_argument("--out-orders", type=Path, default=None)
    ap.add_argument(
        "--out-market", type=Path, default=None,
        help="compact per-leg/per-kind clip percentile summary (committed artifact)",
    )
    ap.add_argument(
        "--out-market-raw", type=Path, default=None,
        help="full per-print market clip dump (large; reproducible, not committed)",
    )
    args = ap.parse_args(argv)

    meta = parse_fills_csv(args.fills_csv.read_text())
    if not meta.addr_to_slot:
        raise SystemExit("no wallet addresses parsed from --fills-csv (DIAG rows)")
    ext = extract(args.data_root, meta, order_gap_ns=args.order_gap_ns)
    print(render_report(ext, meta))
    if args.out_orders:
        write_orders_csv(args.out_orders, ext.orders)
        print(f"wrote {len(ext.orders)} orders -> {args.out_orders}")
    if args.out_market:
        write_market_summary_csv(args.out_market, ext.market_clips, meta)
        print(f"wrote market clip summary -> {args.out_market}")
    if args.out_market_raw:
        write_market_raw_csv(args.out_market_raw, ext.market_clips, meta)
        print(f"wrote market clip raw -> {args.out_market_raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
