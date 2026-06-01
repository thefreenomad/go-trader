"""Selection engine (Step 4, piece 1) — the deployable cross-sectional brain.

`compute_target(aum, lev)` returns the target portfolio for this rebalance,
encoding the locked, validated config. Run directly to print today's book;
imported by orchestrator.py to drive go-trader manual positions.

Locked config (from research):
  universe = top-N HL-liquid perps     lookback = 3 weeks (momentum)
  legs     = top/bottom 20%            weighting = inverse-vol, per-name cap 25%
  sizing   = vol-target 15% (live: from running track record; cold start lev=1)
  gate     = weak-bear: if BTC composite == trending_down_choppy -> FLAT the book
  cadence  = biweekly  (hl_validate.py: on the REAL HL universe + real funding,
             biweekly/3w is the only robust config — +101%/2.3y, Sharpe 1.08,
             maxDD -16%; weekly/1w was a BinanceUS-top-25 artifact (~0 on HL).
             monitor stays unscheduled — it hurt the 3w book. code in orchestrator.)
Signals are fetched LIVE from Hyperliquid (the venue we trade) -- same prices we
execute against, no proxy. Universe stays crypto-only (BinanceUS-derived list
intersected with HL; excludes HL's tokenized equities).
"""
import sys, os, json, warnings, urllib.request
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from regime import compute_regime_composite


def _hl_ohlcv(hl, coin, limit):
    o = hl.fetch_ohlcv(f"{coin}/USDC:USDC", TF, limit=limit)
    df = pd.DataFrame(o, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="ms")
    return df


def _testnet_universe():
    """Coins LISTED on HL testnet with a live price -> {coin: {szd, px}}."""
    body = json.dumps({"type": "metaAndAssetCtxs"}).encode()
    req = urllib.request.Request("https://api.hyperliquid-testnet.xyz/info", data=body,
                                 headers={"Content-Type": "application/json"})
    meta, ctxs = json.loads(urllib.request.urlopen(req, timeout=15).read())
    out = {}
    for i, u in enumerate(meta["universe"]):
        ctx = ctxs[i] if i < len(ctxs) else {}
        px = ctx.get("markPx")
        if px and float(px) > 0:
            out[u["name"]] = {"szd": int(u["szDecimals"]), "px": float(px),
                              "vol": float(ctx.get("dayNtlVlm", 0) or 0)}
    return out

TF = "4h"; BARS_DAY = 6; LB = 3 * 7 * BARS_DAY   # 3-week momentum, biweekly rebalance (hl_validate.py)
QFRAC = 0.20; NLIQ = 40; NAME_CAP = 0.25; REGIME_PERIOD = 90


def _cap_weights(inv_vols):
    """Inverse-vol weights, iteratively enforcing the per-name cap."""
    w = np.array(inv_vols, float); w = w / w.sum()
    for _ in range(20):
        over = w > NAME_CAP + 1e-9
        if not over.any():
            break
        excess = (w[over] - NAME_CAP).sum()
        w[over] = NAME_CAP
        under = ~over
        if not under.any():
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w


def _signals(aum, nliq=NLIQ):
    """Shared universe + momentum + regime computation. compute_target (biweekly
    entry) and compute_monitor (4h exit check) BOTH build off this, so the entry
    ranking and the exit ranking can never drift apart."""
    hl = ccxt.hyperliquid({"enableRateLimit": True}); hl.load_markets(); htk = hl.fetch_tickers()
    def hl_vol(c): return (htk.get(f"{c}/USDC:USDC") or {}).get("quoteVolume")
    syms = json.load(open("research/universe.json"))
    universe = [s.split("/")[0] for s in syms if hl_vol(s.split("/")[0]) is not None]
    universe = sorted(universe, key=lambda c: -(hl_vol(c) or 0))[:nliq]

    # On testnet: keep only coins LISTED on testnet and coarse enough to size
    # (drops e.g. BCH/UNI not on testnet, ZEC szDecimals=0 whose min size >> a slot).
    if os.environ.get("HYPERLIQUID_TESTNET", "") == "1":
        tn = _testnet_universe()
        universe = [c for c in universe
                    if c in tn and (10.0 ** -tn[c]["szd"]) * tn[c]["px"] <= aum * 0.10]

    # signal prices fetched LIVE from Hyperliquid (same venue we execute on)
    need = LB + 60; closes = {}
    for c in universe:
        try:
            closes[c] = _hl_ohlcv(hl, c, need)["close"]
        except Exception:
            continue
    P = pd.DataFrame(closes).sort_index().ffill()
    universe = list(P.columns)
    mom = P.iloc[-1] / P.iloc[-1 - LB] - 1
    vol = P.pct_change().iloc[-LB:].std()
    elig = [c for c in universe if not (pd.isna(mom[c]) or pd.isna(vol[c]) or vol[c] <= 0)]

    btc_df = _hl_ohlcv(hl, "BTC", need)
    regime = compute_regime_composite(btc_df, period=REGIME_PERIOD)["regime"].iloc[-1]
    return dict(P=P, mom=mom, vol=vol, universe=universe, elig=elig,
                asof=str(P.index[-1]), regime=regime, gated=(regime == "trending_down_choppy"))


def compute_target(aum, lev=1.0, nliq=NLIQ):
    s = _signals(aum, nliq)
    P, mom, vol, elig, gated = s["P"], s["mom"], s["vol"], s["elig"], s["gated"]
    target = []
    if not gated and len(elig) >= 8:
        ranked = sorted(elig, key=lambda c: mom[c]); k = max(1, int(len(elig) * QFRAC))
        for leg, side in [(ranked[-k:], "long"), (ranked[:k], "short")]:
            w = _cap_weights([1.0 / vol[c] for c in leg])
            for c, wi in zip(leg, w):
                target.append(dict(coin=c, side=side, weight=float(wi),
                                   notional=round(aum * lev * wi, 2), mom=float(mom[c]),
                                   price=round(float(P.iloc[-1][c]), 6)))
    return dict(asof=s["asof"], regime=s["regime"], gated=gated, universe_coins=s["universe"],
                universe=len(s["universe"]), eligible=len(elig), target=target)


def compute_monitor(aum, nliq=NLIQ):
    """4h intra-cycle exit signals (close-only). Returns the regime gate plus each
    eligible coin's momentum percentile rank in [0,1] (1 = strongest). The monitor
    closes a held long when its rank falls below EXIT_PCT and a held short when its
    rank rises above 1-EXIT_PCT (validated combo: regime4h + exit-decay)."""
    s = _signals(aum, nliq)
    elig, mom = s["elig"], s["mom"]
    pct = mom[elig].rank(pct=True) if elig else mom[:0]
    return dict(asof=s["asof"], regime=s["regime"], gated=s["gated"],
                eligible=len(elig), mom_pct={c: float(pct[c]) for c in elig})


def main():
    aum = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
    r = compute_target(aum)
    print(f"=== SELECTION ENGINE  (as of {r['asof']}, AUM ${aum:,.0f}) ===")
    print(f"universe {r['universe']} | eligible {r['eligible']} | BTC regime: {r['regime']}"
          f"{'  -> WEAK BEAR: FLAT' if r['gated'] else ''}\n")
    if not r["target"]:
        print("TARGET: FLAT"); return
    g = sum(t["notional"] for t in r["target"])
    nl = sum(t["side"] == "long" for t in r["target"])
    print(f"TARGET ({nl} long / {len(r['target'])-nl} short, gross ${g:,.0f}):")
    print(f"  {'coin':6s} {'side':6s} {'mom4w':>8s} {'weight':>7s} {'notional':>12s}")
    for t in sorted(r["target"], key=lambda t: (t["side"], -t["weight"])):
        print(f"  {t['coin']:6s} {t['side']:6s} {t['mom']*100:>7.1f}% {t['weight']*100:>6.1f}% ${t['notional']:>11,.0f}")


if __name__ == "__main__":
    main()
