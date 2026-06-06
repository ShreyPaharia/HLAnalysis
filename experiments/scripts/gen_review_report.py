#!/usr/bin/env python3
"""Generate the codebase-review HTML overview from the multi-agent workflow output.

Loads the synthesized findings JSON, merges a handful of unique hedge/risk
findings the re-run risk reviewer surfaced, and emits a self-contained,
filterable HTML report ordered by priority and bucketed into groups.
"""
import json
import html
import datetime

SRC = ('/private/tmp/claude-501/'
       '-Users-shreypaharia-Documents-Projects-Trading-HLAnalysis--worktrees-'
       'review-the-codebase-using-various-subagents-for-is-wHYMLU/'
       '4cc96c81-4e41-4bd2-b4b8-139283e2f0a5/tasks/w1vtl6ny2.output')
OUT = ('/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/.worktrees/'
       'review-the-codebase-using-various-subagents-for-is-wHYMLU/'
       'summeries/2026-05-31-codebase-review.html')

raw = open(SRC).read()
raw = raw[raw.find('{'):]
data = json.loads(raw)
synth = data['result']['synth']
groups = synth['groups']

# --- Unique hedge/risk findings recovered from the re-run risk reviewer ---
# (The original risk-and-hedging reviewer failed to return structured output;
#  these are items NOT already covered by the synthesis Hedge & risk group.)
extra_risk = [
    {
        "title": "Global inventory cap double-counts in-flight orders and ignores cross-venue exposure",
        "severity": "medium", "priority": 25.5,
        "locations": ["hlanalysis/engine/risk.py:79-84", "hlanalysis/engine/scanner.py:185-191"],
        "problem": "max_total_inventory = live_orders_notional + new_notional + sum(|qty|*avg_entry). live_orders can include pending orders already (partially) reflected as positions once filled, and positions are read per-slot/per-venue only. Aggregate exposure is bounded per-account-slot but NOT across slots/venues, so a supposedly-hedged multi-account book has no true portfolio-level exposure bound and the in-flight term can over- or under-count by fill timing.",
        "recommendation": "Define inventory from confirmed positions plus genuinely-unacked order notional (exclude orders already booked as positions); add a cross-slot/global exposure aggregate if accounts share capital or net risk.",
        "effort": "M", "confidence": "medium",
    },
    {
        "title": "Concurrent-positions / inventory caps evaluated against a stale pre-fill snapshot within a scan tick",
        "severity": "low", "priority": 25.6,
        "locations": ["hlanalysis/engine/risk.py:86-90", "hlanalysis/engine/scanner.py:185-191"],
        "problem": "positions and live_orders_total_notional are snapshotted once per scan tick and reused across every intent emitted that tick, with no provisional increment between intents. Multiple new entries in a single tick are all checked against the same pre-fill snapshot, so the engine can approve several entries that jointly exceed the concurrent-positions or global-inventory cap.",
        "recommendation": "Accumulate provisional notional/position-count across intents within a single scan tick (or re-snapshot after each placement) so caps reflect same-tick approvals already routed.",
        "effort": "S", "confidence": "medium",
    },
    {
        "title": "Reconcile loop keeps mutating positions/orders even when the slot is restart-blocked",
        "severity": "medium", "priority": 25.7,
        "locations": ["hlanalysis/engine/runtime.py:470-474", "hlanalysis/engine/runtime.py:762-822"],
        "problem": "slot.blocked (from the restart-drift gate) only suppresses the scan loop; _reconcile_loop still adopts/overwrites positions, cancels orphan orders, and publishes settlement Exits while the operator is supposedly holding the slot for inspection. Automatic DB/venue mutation races the operator and can erase the very drift evidence the block was meant to preserve.",
        "recommendation": "When slot.blocked, run reconcile in detect-only mode (emit drift events, no DB mutation, no cancels) until the operator clears the block.",
        "effort": "S", "confidence": "medium",
    },
    {
        "title": "Stale-reconcile pre-trade gate fails open when no reconcile has ever succeeded",
        "severity": "medium", "priority": 25.8,
        "locations": ["hlanalysis/engine/risk.py:119-124", "hlanalysis/engine/runtime.py:430-431"],
        "problem": "check_pre_trade only enforces the stale-reconcile guard when last_reconcile_ns > 0, but last_reconcile_ns is optimistically seeded to startup time regardless of whether a reconcile actually succeeded. If the startup reconcile errored (or the periodic loop keeps crashing), the engine still trades for the first intervals on the false assumption that startup == a clean reconcile.",
        "recommendation": "Only set last_reconcile_ns after a reconcile completes successfully (including the startup drift run); treat last_reconcile_ns==0 as stale (veto) rather than skip-the-check.",
        "effort": "S", "confidence": "medium",
    },
    {
        "title": "Hedge ledger marks itself fully hedged the instant the IOC intent is emitted, before any fill",
        "severity": "high", "priority": 12.5,
        "locations": ["hlanalysis/strategy/delta_hedged.py:111-121", "hlanalysis/strategy/delta_hedged.py:100"],
        "problem": "_HedgeState.hedge_qty_btc is set to the full target_delta the moment the hedge IOC intent is emitted, before any fill confirmation. If the hedge IOC partial-fills or fails, the ledger believes it is fully hedged and won't re-fire next tick (gap~=0), leaving silent naked exposure. The strategy never sees realized hedge fills, so there is no feedback loop to correct an under-filled hedge — the same fire-and-forget bug as the primary leg, applied to the hedge.",
        "recommendation": "Update hedge_qty_btc from confirmed hedge fills via a fill-feedback channel, not from the intended target; treat any hedge IOC shortfall as residual exposure that must re-fire next tick.",
        "effort": "M", "confidence": "high",
    },
]

# Insert the extra risk items into the Hedge & risk group.
for g in groups:
    if g['group'].startswith('Hedge'):
        g['items'].extend(extra_risk)
        g['items'].sort(key=lambda it: it['priority'])
        break

# Re-rank priority to clean integers across all items by current priority order.
all_items = []
for gi, g in enumerate(groups):
    for it in g['items']:
        all_items.append((it['priority'], gi, it))
all_items.sort(key=lambda x: x[0])
for new_rank, (_, _, it) in enumerate(all_items, start=1):
    it['rank'] = new_rank

# --- Stats ---
sev_order = ['critical', 'high', 'medium', 'low']
sev_counts = {s: 0 for s in sev_order}
eff_counts = {'S': 0, 'M': 0, 'L': 0}
for _, _, it in all_items:
    sev_counts[it['severity']] += 1
    eff_counts[it.get('effort', 'M')] += 1
total = len(all_items)

SEV_COLOR = {
    'critical': '#e5484d', 'high': '#f76808', 'medium': '#ffb224', 'low': '#3e9b4f',
}
SEV_BG = {
    'critical': 'rgba(229,72,77,.12)', 'high': 'rgba(247,104,8,.12)',
    'medium': 'rgba(255,178,36,.12)', 'low': 'rgba(62,155,79,.12)',
}
EFF_LABEL = {'S': 'S · <2h', 'M': 'M · ~½ day', 'L': 'L · multi-day'}


def esc(s):
    return html.escape(str(s))


def loc_html(locs):
    return ''.join(
        f'<code class="loc">{esc(l)}</code>' for l in locs
    )


# Top-10 quick table
top = [it for _, _, it in all_items[:10]]


def render_item(it):
    sev = it['severity']
    return f'''
    <div class="card" data-sev="{sev}" data-eff="{it.get('effort','M')}">
      <div class="card-head">
        <span class="rank">#{it['rank']}</span>
        <span class="badge" style="background:{SEV_BG[sev]};color:{SEV_COLOR[sev]};border-color:{SEV_COLOR[sev]}">{sev.upper()}</span>
        <span class="eff" title="effort">{EFF_LABEL[it.get('effort','M')]}</span>
        <span class="conf" title="reviewer confidence">conf: {esc(it.get('confidence','?'))}</span>
        <h3>{esc(it['title'])}</h3>
      </div>
      <div class="locs">{loc_html(it['locations'])}</div>
      <p class="prob"><strong>Problem.</strong> {esc(it['problem'])}</p>
      <p class="rec"><strong>Fix.</strong> {esc(it['recommendation'])}</p>
    </div>'''


group_html = []
for gi, g in enumerate(groups):
    items = sorted(g['items'], key=lambda it: it['rank'])
    gsev = {s: 0 for s in sev_order}
    for it in items:
        gsev[it['severity']] += 1
    chips = ' '.join(
        f'<span class="mini" style="color:{SEV_COLOR[s]}">{gsev[s]} {s}</span>'
        for s in sev_order if gsev[s]
    )
    cards = ''.join(render_item(it) for it in items)
    group_html.append(f'''
    <section class="group" data-group="{gi}">
      <div class="group-head">
        <h2>{esc(g['group'])}</h2>
        <div class="group-chips">{chips}</div>
      </div>
      <p class="rationale">{esc(g['rationale'])}</p>
      {cards}
    </section>''')

top_rows = ''.join(
    f'''<tr data-sev="{it['severity']}">
      <td class="r">#{it['rank']}</td>
      <td><span class="dot" style="background:{SEV_COLOR[it['severity']]}"></span>{it['severity']}</td>
      <td>{esc(it['title'])}</td>
      <td>{esc(it.get('effort','M'))}</td>
      <td class="loc-cell">{esc(it['locations'][0]) if it['locations'] else ''}</td>
    </tr>''' for it in top
)

sev_stat = ''.join(
    f'''<div class="stat" style="border-color:{SEV_COLOR[s]}">
      <div class="stat-n" style="color:{SEV_COLOR[s]}">{sev_counts[s]}</div>
      <div class="stat-l">{s}</div></div>''' for s in sev_order
)

now = datetime.date(2026, 5, 31).strftime('%Y-%m-%d')
summary = esc(synth['summary'])

doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HLAnalysis — Codebase Review &amp; Action Plan</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c232d; --border:#2a313c;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 22px 80px; }}
  header.top {{ border-bottom:1px solid var(--border); padding-bottom:20px; margin-bottom:24px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.4px; }}
  .sub {{ color:var(--muted); font-size:13.5px; }}
  .summary {{
    background:linear-gradient(180deg,var(--panel2),var(--panel));
    border:1px solid var(--border); border-left:3px solid var(--accent);
    border-radius:10px; padding:18px 20px; margin:22px 0; font-size:14.5px;
  }}
  .summary strong {{ color:#fff; }}
  .stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:18px 0 8px; }}
  .stat {{
    flex:1; min-width:120px; background:var(--panel); border:1px solid var(--border);
    border-top:3px solid; border-radius:8px; padding:12px 14px; text-align:center;
  }}
  .stat-n {{ font-size:30px; font-weight:700; line-height:1; }}
  .stat-l {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; margin-top:4px; }}
  .stat.total {{ border-top-color:var(--accent); }}
  .stat.total .stat-n {{ color:var(--accent); }}
  .meta-line {{ color:var(--muted); font-size:12.5px; margin-top:6px; }}
  h2 {{ font-size:19px; margin:0; }}
  .controls {{
    position:sticky; top:0; z-index:5; background:rgba(13,17,23,.92);
    backdrop-filter:blur(6px); padding:12px 0; margin:18px 0 6px;
    border-bottom:1px solid var(--border); display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  }}
  .controls .lbl {{ color:var(--muted); font-size:12.5px; margin-right:2px; }}
  .fbtn {{
    background:var(--panel); color:var(--text); border:1px solid var(--border);
    border-radius:999px; padding:5px 13px; font-size:12.5px; cursor:pointer; transition:.12s;
  }}
  .fbtn:hover {{ border-color:var(--accent); }}
  .fbtn.active {{ background:var(--accent); color:#06101f; border-color:var(--accent); font-weight:600; }}
  table.top {{ width:100%; border-collapse:collapse; margin:8px 0 10px; font-size:13.5px; }}
  table.top th {{ text-align:left; color:var(--muted); font-weight:500; font-size:11.5px;
    text-transform:uppercase; letter-spacing:.5px; padding:6px 8px; border-bottom:1px solid var(--border); }}
  table.top td {{ padding:7px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
  table.top td.r {{ color:var(--muted); white-space:nowrap; }}
  table.top td.loc-cell {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--muted); }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
  details.topwrap {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:6px 16px; margin:10px 0 26px; }}
  details.topwrap summary {{ cursor:pointer; font-weight:600; padding:8px 0; font-size:14px; }}
  section.group {{ margin:30px 0; }}
  .group-head {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
    border-bottom:1px solid var(--border); padding-bottom:8px; margin-bottom:6px; }}
  .group-chips {{ display:flex; gap:10px; }}
  .mini {{ font-size:12px; font-weight:600; }}
  .rationale {{ color:var(--muted); font-size:13.5px; margin:6px 0 16px; }}
  .card {{
    background:var(--panel); border:1px solid var(--border); border-radius:10px;
    padding:15px 17px; margin:11px 0; transition:.12s;
  }}
  .card:hover {{ border-color:#3d4654; }}
  .card-head {{ display:flex; align-items:center; gap:9px; flex-wrap:wrap; }}
  .card-head h3 {{ font-size:15.5px; margin:0; flex-basis:100%; order:5; line-height:1.4; }}
  .rank {{ color:var(--muted); font-weight:700; font-size:13px; font-variant-numeric:tabular-nums; }}
  .badge {{ font-size:10.5px; font-weight:700; letter-spacing:.5px; padding:2px 8px;
    border-radius:5px; border:1px solid; }}
  .eff, .conf {{ font-size:11.5px; color:var(--muted); background:var(--panel2);
    border:1px solid var(--border); border-radius:5px; padding:2px 7px; }}
  .locs {{ margin:10px 0 9px; display:flex; flex-wrap:wrap; gap:6px; }}
  code.loc {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--accent);
    background:rgba(88,166,255,.08); border:1px solid rgba(88,166,255,.18);
    border-radius:5px; padding:1px 7px; }}
  .prob {{ margin:8px 0; font-size:13.8px; }}
  .rec {{ margin:8px 0 2px; font-size:13.8px; color:#cfe7d4; }}
  .prob strong {{ color:#ffb4b4; }}
  .rec strong {{ color:#8ee0a0; }}
  .hidden {{ display:none !important; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:40px; border-top:1px solid var(--border); padding-top:16px; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>HLAnalysis — Codebase Review &amp; Prioritized Action Plan</h1>
    <div class="sub">Live HL HIP-4 / BTC + Polymarket market-making &amp; hedging engine · multi-agent audit · {now}</div>
  </header>

  <div class="summary"><strong>Executive summary.</strong> {summary}</div>

  <div class="stats">
    <div class="stat total"><div class="stat-n">{total}</div><div class="stat-l">findings</div></div>
    {sev_stat}
  </div>
  <div class="meta-line">Effort split — S (&lt;2h): {eff_counts['S']} · M (~½ day): {eff_counts['M']} · L (multi-day): {eff_counts['L']} &nbsp;|&nbsp; 9 specialist reviewers across engine safety, hedge/risk, strategy math, sim performance, sim-vs-live parity, adapter robustness, duplication, testing, data/ops.</div>

  <details class="topwrap" open>
    <summary>Top 10 — fix before any further tuning or capital increase</summary>
    <table class="top">
      <thead><tr><th>#</th><th>Sev</th><th>Finding</th><th>Eff</th><th>First location</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>
  </details>

  <div class="controls">
    <span class="lbl">Severity:</span>
    <button class="fbtn active" data-f="sev" data-v="all">all</button>
    <button class="fbtn" data-f="sev" data-v="critical">critical</button>
    <button class="fbtn" data-f="sev" data-v="high">high</button>
    <button class="fbtn" data-f="sev" data-v="medium">medium</button>
    <button class="fbtn" data-f="sev" data-v="low">low</button>
    <span class="lbl" style="margin-left:14px">Effort:</span>
    <button class="fbtn active" data-f="eff" data-v="all">all</button>
    <button class="fbtn" data-f="eff" data-v="S">S</button>
    <button class="fbtn" data-f="eff" data-v="M">M</button>
    <button class="fbtn" data-f="eff" data-v="L">L</button>
  </div>

  {''.join(group_html)}

  <footer>
    Generated by a multi-agent review workflow (9 parallel specialist reviewers + synthesis).
    Severity / priority / effort / confidence reflect reviewer judgment over a static read of the
    code as of {now}; verify each finding against current <code>main</code> before acting.
    Locations are <code>file:line</code> at review time and may shift.
  </footer>
</div>

<script>
  const state = {{ sev:'all', eff:'all' }};
  function apply() {{
    document.querySelectorAll('.card').forEach(c => {{
      const okS = state.sev==='all' || c.dataset.sev===state.sev;
      const okE = state.eff==='all' || c.dataset.eff===state.eff;
      c.classList.toggle('hidden', !(okS && okE));
    }});
    document.querySelectorAll('section.group').forEach(g => {{
      const any = g.querySelectorAll('.card:not(.hidden)').length;
      g.classList.toggle('hidden', any===0);
    }});
    document.querySelectorAll('table.top tbody tr').forEach(r => {{
      const okS = state.sev==='all' || r.dataset.sev===state.sev;
      r.classList.toggle('hidden', !okS);
    }});
  }}
  document.querySelectorAll('.fbtn').forEach(b => b.addEventListener('click', () => {{
    const f=b.dataset.f, v=b.dataset.v;
    state[f]=v;
    document.querySelectorAll('.fbtn[data-f="'+f+'"]').forEach(x=>x.classList.toggle('active', x===b));
    apply();
  }}));
</script>
</body>
</html>'''

open(OUT, 'w').write(doc)
print('wrote', OUT)
print('total items:', total, '| critical:', sev_counts['critical'], 'high:', sev_counts['high'])
