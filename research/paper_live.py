"""Live forward PAPER shadow — run several cadence/lookback configs in parallel on
LIVE Hyperliquid prices (no testnet, no orders, no account), to watch how they
behave going forward before committing real capital.

Each config holds a simulated inverse-vol, vol-targeted, weak-bear-gated L/S book
(identical construction to the live select_engine) and rebalances on its OWN
cadence. Each tick marks every book to the latest HL price and appends per-config
paper equity to research/results/paper_live.csv. Turnover cost is charged at
rebalance (8.5bp one-way); funding is omitted live (small net on an L/S book —
add at reconciliation). State persists in paper_live_state.json so it resumes.

  paper_live.py            # one tick: rebalance any due config, mark all, log
  paper_live.py --status   # print the current standings, no state change

Run on a 4h cron (aligned to candle close). Equity is a normalized index (1.0 = start).
"""
import sys, os, json, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from regime import compute_regime_composite

TF = "4h"; BARS_DAY = 6; WEEK = 7 * BARS_DAY
QFRAC = 0.20; NLIQ = 40; NAME_CAP = 0.25; REGIME_PERIOD = 90
TARGET_VOL = 0.15; LEV_CAP = (0.25, 3.0); VT_WARMUP = 8; COST = 0.00085
DOWN = "trending_down_choppy"
STATE = "research/results/paper_live_state.json"
LEDGER = "research/results/paper_live.csv"

# (name, lookback_weeks, cadence_days) — the configs we're racing
CONFIGS = [
    ("weekly_1w",   1, 7),      # the just-deployed pick (BinanceUS artifact)
    ("weekly_2w",   2, 7),
    ("biweekly_3w", 3, 14),     # the HL-validated winner (+101%/2.3y, Sharpe 1.08)
    ("biweekly_4w", 4, 14),     # the original
]


def _cap(iv):
    w = iv / iv.sum()
    for _ in range(20):
        over = w > NAME_CAP + 1e-9
        if not over.any():
            break
        exc = (w[over] - NAME_CAP).sum(); w[over] = NAME_CAP
        un = ~over
        if not un.any():
            break
        w[un] += exc * (w[un] / w[un].sum())
    return w


def fetch():
    """Live HL: universe (crypto ∩ top-NLIQ), close-price matrix, current price, regime."""
    hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets(); tk = hl.fetch_tickers()
    def vol(c):
        t = tk.get(f"{c}/USDC:USDC"); return (t.get("quoteVolume") or 0) if t else 0
    uni = json.load(open("research/universe.json"))
    universe = sorted([s.split("/")[0] for s in uni if vol(s.split("/")[0]) > 0], key=lambda c: -vol(c))[:NLIQ]
    need = max(lb for _, lb, _ in CONFIGS) * WEEK + 80
    closes = {}
    for c in universe:
        try:
            o = hl.fetch_ohlcv(f"{c}/USDC:USDC", TF, limit=need)
            s = pd.Series([r[4] for r in o], index=pd.to_datetime([r[0] for r in o], unit="ms"))
            closes[c] = s
        except Exception:
            continue
    P = pd.DataFrame(closes).sort_index().ffill()
    btc = hl.fetch_ohlcv("BTC/USDC:USDC", TF, limit=need)
    bdf = pd.DataFrame(btc, columns=["t", "open", "high", "low", "close", "volume"])
    bdf.index = pd.to_datetime(bdf["t"], unit="ms")
    regime = compute_regime_composite(bdf, period=REGIME_PERIOD)["regime"].iloc[-1]
    cur = {c: float(P[c].iloc[-1]) for c in P.columns}
    return P, cur, regime


def target_book(P, lb_bars, lev):
    """Signed-weight book (each leg sums to lev): top/bottom QFRAC by lb momentum,
    inverse-vol with per-name cap. Returns {coin: signed_weight}."""
    last = P.iloc[-1]; prev = P.iloc[-1 - lb_bars]
    mom = (last / prev - 1)
    vol = P.pct_change().iloc[-lb_bars:].std()
    elig = [c for c in P.columns if not (pd.isna(mom[c]) or pd.isna(vol[c]) or vol[c] <= 0)]
    if len(elig) < 8:
        return {}
    ranked = sorted(elig, key=lambda c: mom[c]); k = max(1, int(len(elig) * QFRAC))
    book = {}
    for leg, sgn in [(ranked[-k:], 1.0), (ranked[:k], -1.0)]:
        w = _cap(np.array([1.0 / vol[c] for c in leg]))
        for c, wi in zip(leg, w):
            book[c] = sgn * lev * float(wi)
    return book


def _vt_lev(hist):
    if len(hist) < VT_WARMUP:
        return 1.0
    rv = np.std(hist[-VT_WARMUP:], ddof=1)
    if rv <= 0:
        return 1.0
    return float(np.clip((TARGET_VOL / np.sqrt(365.0 / 14)) / rv, *LEV_CAP))


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def tick():
    P, cur, regime = fetch()
    gated = regime == DOWN
    state = load_state()
    now = datetime.now(timezone.utc); now_s = now.isoformat()
    rows = []
    for name, lb_w, cad_d in CONFIGS:
        lb = lb_w * WEEK
        s = state.get(name) or {"locked_eq": 1.0, "book": {}, "vt_hist": [], "last_rebal": None}
        # mark current book to live prices
        mtm = sum(w * (cur[c] / e - 1) for c, (w, e) in s["book"].items() if c in cur) if s["book"] else 0.0
        eq = s["locked_eq"] * (1 + mtm)
        last = s["last_rebal"] and datetime.fromisoformat(s["last_rebal"])
        due = last is None or (now - last).total_seconds() >= cad_d * 86400
        action = "mark"
        if due:
            # realize the closing book, charge turnover vs the new target, re-form
            lev = _vt_lev(s["vt_hist"] + [mtm])
            tgt = {} if gated else target_book(P, lb, lev)
            old = {c: w for c, (w, e) in s["book"].items()}
            turn = sum(abs(tgt.get(c, 0.0) - old.get(c, 0.0)) for c in set(tgt) | set(old))
            realized = mtm - turn * COST
            s["locked_eq"] *= (1 + realized)
            s["vt_hist"] = (s["vt_hist"] + [mtm])[-40:]
            s["book"] = {c: (w, cur[c]) for c, w in tgt.items() if c in cur}
            s["last_rebal"] = now_s
            eq = s["locked_eq"]
            action = "FLAT(gate)" if gated else f"rebalance lev={lev:.2f}"
        state[name] = s
        gross = sum(abs(w) for w, e in s["book"].values())
        nlong = sum(1 for w, e in s["book"].values() if w > 0)
        rows.append((name, eq, len(s["book"]), nlong, gross, action))
    json.dump(state, open(STATE, "w"), indent=2)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a") as f:
        if new:
            f.write("time,config,equity,n,n_long,gross,regime\n")
        for name, eq, n, nl, g, act in rows:
            f.write(f"{now_s},{name},{eq:.6f},{n},{nl},{g:.2f},{regime}\n")
    print(f"=== paper_live tick {now:%Y-%m-%d %H:%M} UTC | BTC regime {regime} ===")
    print(f"  {'config':14s} {'paper equity':>13s} {'ret%':>7s} {'pos':>4s} {'gross':>6s}  action")
    for name, eq, n, nl, g, act in rows:
        print(f"  {name:14s} {eq:>13.4f} {(eq-1)*100:>+6.2f}% {n:>4d} {g:>6.2f}  {act}")


def status():
    if not os.path.exists(LEDGER):
        print("no paper_live ledger yet"); return
    df = pd.read_csv(LEDGER)
    print(f"=== paper_live standings ({df['time'].nunique()} ticks, since {df['time'].min()[:16]}) ===")
    print(f"  {'config':14s} {'equity':>9s} {'ret%':>7s}")
    for name, _, _ in CONFIGS:
        sub = df[df.config == name]
        if len(sub):
            e = sub.equity.iloc[-1]
            print(f"  {name:14s} {e:>9.4f} {(e-1)*100:>+6.2f}%")


if __name__ == "__main__":
    status() if "--status" in sys.argv else tick()
