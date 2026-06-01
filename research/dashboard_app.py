#!/usr/bin/env python3
"""Live monitoring dashboard for the paper book — balance, positions (live P&L),
equity curve (accumulates over time), and rebalance markers.

Host-served on 0.0.0.0:8090 (reads the mounted docker-data/state.db + live HL prices),
so it's reachable in a browser without the container's loopback restriction.

  .venv/bin/python research/dashboard_app.py          # serve at http://localhost:8090
  .venv/bin/python research/dashboard_app.py --tick    # append one equity snapshot (cron)
"""
import sys, os, json, time, csv, subprocess, urllib.request
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
sys.path.insert(0, "shared_tools")
import ccxt


def _load_env(path=".env"):
    """launchd gives the dashboard a minimal env; pull HYPERLIQUID_* from .env so
    we can read the live account. Only the (public) account ADDRESS is needed for
    the read-only /info query -- the secret key is never used here."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("HYPERLIQUID_") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


_load_env()
LEDGER = "research/results/live_ledger.csv"
EVENTS = "research/results/rebalance_events.csv"
BASELINE = "research/results/baseline.json"
PAPER = "research/results/paper_live.csv"
# (config key, legend label, distinct color) — order = legend order; 3w is the winner
PAPER_META = [
    ("biweekly_3w", "biweekly / 3w  (HL winner)", "#3fb950"),
    ("weekly_1w",   "weekly / 1w",                "#f85149"),
    ("weekly_2w",   "weekly / 2w",                "#d29922"),
    ("biweekly_4w", "biweekly / 4w",              "#a371f7"),
]
CONTAINER = "go-trader-paper"
DOCKER = os.environ.get("DOCKER_BIN", "/usr/local/bin/docker")   # launchd has minimal PATH
TESTNET = os.environ.get("HYPERLIQUID_TESTNET", "") == "1"
HL_INFO = ("https://api.hyperliquid-testnet.xyz" if TESTNET else "https://api.hyperliquid.xyz") + "/info"
AUM = 10000.0                          # synthetic base (paper fallback only)
PORT = 8090
_sc = {"t": 0.0, "d": None}
_pc = {"t": 0.0, "px": {}}
_ac = {"t": 0.0, "d": None}
_hl = None


def hl_prices():
    """LIVE HL tickers (fresh each refresh, cached 15s) for marking."""
    global _hl
    if time.time() - _pc["t"] < 15 and _pc["px"]:
        return _pc["px"]
    try:
        _hl = _hl or ccxt.hyperliquid({"enableRateLimit": True})
        t = _hl.fetch_tickers()
        _pc.update(t=time.time(), px={s.split("/")[0]: (t[s].get("last") or t[s].get("close") or t[s].get("mark"))
                                      for s in t if s.endswith("/USDC:USDC")})
    except Exception:
        pass
    return _pc["px"]


def status():
    """Authoritative in-memory book via go-trader /status (positions + live marks)."""
    if time.time() - _sc["t"] < 25 and _sc["d"]:
        return _sc["d"]
    try:
        r = subprocess.run([DOCKER, "exec", CONTAINER, "curl", "-fsS", "http://localhost:8099/status"],
                           capture_output=True, text=True, timeout=20)
        _sc.update(t=time.time(), d=json.loads(r.stdout))
    except Exception:
        pass
    return _sc["d"] or {}


def live_account():
    """Authoritative book + balance from the ACTUAL HL account (read-only /info).
    Returns (account_value, positions) or None if no address / query fails."""
    addr = os.environ.get("HYPERLIQUID_ACCOUNT_ADDRESS")
    if not addr:
        return None
    if time.time() - _ac["t"] < 25 and _ac["d"]:
        return _ac["d"]
    try:
        body = json.dumps({"type": "clearinghouseState", "user": addr}).encode()
        req = urllib.request.Request(HL_INFO, data=body, headers={"Content-Type": "application/json"})
        st = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception:
        return _ac["d"]
    val = float((st.get("marginSummary") or {}).get("accountValue") or 0)
    rows = []
    for p in st.get("assetPositions", []):
        pp = p["position"]; szi = float(pp.get("szi", 0) or 0)
        if szi == 0:
            continue
        entry = float(pp.get("entryPx") or 0); upnl = float(pp.get("unrealizedPnl") or 0)
        side = "long" if szi > 0 else "short"; sgn = 1 if szi > 0 else -1
        rows.append(dict(coin=pp["coin"], side=side, qty=abs(szi), entry=entry,
                         cur=entry + (upnl / (sgn * abs(szi)) if szi else 0),
                         pnl=upnl, notional=abs(szi) * entry))
    _ac.update(t=time.time(), d=(val, rows))
    return val, rows


def _baseline(val):
    """First-seen account value, persisted -> total P&L = current - baseline."""
    if os.path.exists(BASELINE):
        try:
            return float(json.load(open(BASELINE))["v"])
        except Exception:
            pass
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    json.dump({"v": val}, open(BASELINE, "w"))
    return val


def mark():
    acct = live_account() if TESTNET else None
    if acct is not None:                       # live/testnet: account is ground truth
        val, rows = acct
        gross = sum(r["notional"] for r in rows)
        pnl = val - _baseline(val)             # total P&L since first snapshot (incl. realized+funding)
        return rows, val, pnl, gross
    d = status(); spx = d.get("prices", {}) or {}; lpx = hl_prices()
    strat = d.get("strategies", [])
    if isinstance(strat, dict):
        strat = list(strat.values())
    out = []; pnl = 0.0; gross = 0.0
    for x in strat:
        for sym, p in (x.get("positions") or {}).items():
            side = p.get("side"); qty = p.get("quantity", 0); entry = p.get("avg_cost", 0)
            cur = lpx.get(sym) or spx.get(sym) or entry   # live HL ticker, fallback /status
            sgn = 1 if side == "long" else -1
            pp = sgn * qty * (cur - entry); notional = qty * entry
            pnl += pp; gross += notional
            out.append(dict(coin=sym, side=side, qty=qty, entry=entry, cur=cur, pnl=pp, notional=notional))
    return out, AUM + pnl, pnl, gross


def snapshot():
    rows, equity, pnl, gross = mark()
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    last = 0.0
    if os.path.exists(LEDGER):
        try:
            last = datetime.fromisoformat(open(LEDGER).readlines()[-1].split(",")[0]).timestamp()
        except Exception:
            pass
    if rows and time.time() - last >= 120:        # one point / 2 min max
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a") as f:
            if new:
                f.write("time,equity,pnl,gross,n\n")
            f.write(f"{datetime.now(timezone.utc).isoformat()},{equity:.2f},{pnl:.2f},{gross:.0f},{len(rows)}\n")
    return rows, equity, pnl, gross


def _read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def paper_curves():
    """Per-config paper equity from paper_live.csv -> (labels, Chart.js datasets, standings).
    Equity shown as % return; each config a distinct color, the 3w winner thickest."""
    rows = _read(PAPER)
    if not rows:
        return None
    times = sorted({r["time"] for r in rows})
    labels = [t[:16].replace("T", " ") for t in times]
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["time"]] = float(r["equity"])
    datasets, standings = [], []
    for cfg, label, color in PAPER_META:
        s = by.get(cfg)
        if not s:
            continue
        data = [round((s[t] - 1) * 100, 3) if t in s else None for t in times]
        datasets.append({"label": label, "data": data, "borderColor": color,
                         "backgroundColor": "rgba(0,0,0,0)", "fill": False, "tension": 0.2,
                         "borderWidth": 3 if cfg == "biweekly_3w" else 1.8,
                         "pointRadius": 0, "spanGaps": True})
        last = data[-1] if data and data[-1] is not None else 0.0
        standings.append((label, color, last))
    return labels, datasets, standings


def render():
    rows, equity, pnl, gross = snapshot()
    led = _read(LEDGER); evs = _read(EVENTS)
    times = [r["time"][:16].replace("T", " ") for r in led]
    eq = [float(r["equity"]) for r in led]
    ev_ts = [e["time"] for e in evs]
    # mark ledger points that coincide (within 6min) with a rebalance
    radius, pcolor = [], []
    for r in led:
        try:
            t = datetime.fromisoformat(r["time"]).timestamp()
        except Exception:
            t = 0
        hit = any(abs(t - datetime.fromisoformat(e).timestamp()) < 360 for e in ev_ts)
        radius.append(5 if hit else 0); pcolor.append("#f0883e" if hit else "rgba(0,0,0,0)")
    color = "pos" if pnl >= 0 else "neg"
    prows = "".join(
        f"<tr class={p['side']}><td>{p['coin']}</td><td>{p['side']}</td><td>${p['notional']:,.0f}</td>"
        f"<td>{p['entry']:.4g}</td><td>{p['cur']:.4g}</td>"
        f"<td class={'pos' if p['pnl']>=0 else 'neg'}>${p['pnl']:+,.0f}</td></tr>"
        for p in sorted(rows, key=lambda x: (x["side"], -x["notional"])))

    # paper-shadow race (the 4 configs running in parallel on live HL prices)
    pc = paper_curves()
    paper_block = paper_script = ""
    if pc:
        plabels, pdata, pstand = pc
        chips = "".join(
            f"<span style='color:{c};font-weight:600;margin-right:16px'>● {lab.split('  ')[0]} "
            f"<b>{v:+.2f}%</b></span>" for lab, c, v in sorted(pstand, key=lambda x: -x[2]))
        paper_block = (
            "<h2>Paper-shadow race — 4 configs on live HL prices (paper, no orders)</h2>"
            f"<div style='margin:0 0 8px;font-size:13px'>{chips}</div>"
            "<canvas id=paper height=80></canvas>")
        paper_script = (
            "new Chart(document.getElementById('paper'),{type:'line',"
            f"data:{{labels:{json.dumps(plabels)},datasets:{json.dumps(pdata)}}},"
            "options:{interaction:{mode:'index',intersect:false},"
            "plugins:{legend:{display:true,position:'top',labels:{color:'#c9d1d9',boxWidth:12,usePointStyle:true}}},"
            "scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{color:'#21262d'}},"
            "y:{ticks:{color:'#8b949e',callback:function(v){return v+'%'}},grid:{color:'#21262d'}}}}});")
    return f"""<!doctype html><html><head><meta charset=utf-8><meta http-equiv=refresh content=60>
<title>XS-Momentum Paper</title><script src=https://cdn.jsdelivr.net/npm/chart.js></script>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:26px}}
h1{{font-weight:600;margin:0 0 18px}} .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:22px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 22px;min-width:130px}}
.card .v{{font-size:26px;font-weight:600}} .card .l{{color:#8b949e;font-size:12px;margin-bottom:4px}}
.pos{{color:#3fb950}} .neg{{color:#f85149}} h2{{color:#8b949e;font-weight:500;font-size:14px;margin:22px 0 8px}}
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}}
th,td{{padding:7px 14px;text-align:left;border-bottom:1px solid #21262d;font-size:13px}} th{{color:#8b949e;font-weight:500}}
tr.short td:nth-child(2){{color:#f85149}} tr.long td:nth-child(2){{color:#3fb950}}
canvas{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}}</style></head><body>
<h1>Cross-Sectional Momentum — Paper Book</h1>
<div class=cards>
<div class=card><div class=l>Balance</div><div class="v {color}">${equity:,.0f}</div></div>
<div class=card><div class=l>P&amp;L</div><div class="v {color}">${pnl:+,.0f}</div></div>
<div class=card><div class=l>Gross</div><div class=v>${gross:,.0f}</div></div>
<div class=card><div class=l>Positions</div><div class=v>{len(rows)}</div></div>
<div class=card><div class=l>Rebalances</div><div class=v>{len(evs)}</div></div>
</div>
<h2>Equity curve — orange dots = rebalances</h2><canvas id=eq height=78></canvas>
<h2>Positions (live mark)</h2>
<table><tr><th>coin</th><th>side</th><th>notional</th><th>entry</th><th>current</th><th>P&amp;L</th></tr>{prows}</table>
{paper_block}
<p style="color:#8b949e;font-size:12px;margin-top:14px">auto-refreshes 60s · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · {'live HL testnet account' if TESTNET else f'base AUM $' + format(AUM, ',.0f')}</p>
<script>new Chart(document.getElementById('eq'),{{type:'line',
data:{{labels:{json.dumps(times)},datasets:[{{data:{json.dumps(eq)},borderColor:'#58a6ff',
backgroundColor:'rgba(88,166,255,.08)',fill:true,tension:.2,borderWidth:2,
pointRadius:{json.dumps(radius)},pointBackgroundColor:{json.dumps(pcolor)},pointBorderColor:{json.dumps(pcolor)}}}]}},
options:{{plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#8b949e',maxTicksLimit:8}},grid:{{color:'#21262d'}}}},
y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}}}}}});
{paper_script}</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            body = render().encode()
        except Exception as e:
            body = f"<pre>error: {e}</pre>".encode()
        self.send_response(200); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if "--tick" in sys.argv:
        r, e, p, g = snapshot(); print(f"snapshot: balance ${e:,.0f}  pnl ${p:+,.0f}  ({len(r)} pos)")
    else:
        print(f"dashboard -> http://localhost:{PORT}")
        HTTPServer(("0.0.0.0", PORT), H).serve_forever()
