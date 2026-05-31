"""Build an expanded ~30-50 coin universe (liquid BinanceUS USDT spot, ex-stables)
and fetch 4h history since 2023. Reports coverage + volatility for the high-level
picture. SURVIVORSHIP-BIASED (today's survivors only) — note in any conclusion.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd, ccxt
from data_fetcher import fetch_full_history

START = "2023-01-01"; TF = "4h"; N_CANDIDATES = 60; MIN_BARS = 180 * 4  # ~120d of 4h
EXCLUDE = {"USDT","USDC","DAI","USDG","TUSD","BUSD","FDUSD","PYUSD","USDP","GUSD",
           "PAXG","XAUT","WBTC","WETH","WBETH","EUR","GBP","AEUR"}

ex = ccxt.binanceus({"enableRateLimit": True}); ex.load_markets()
tk = ex.fetch_tickers()
cands = []
for sym, d in tk.items():
    if not sym.endswith("/USDT") or not d.get("quoteVolume"):
        continue
    base = sym.split("/")[0]
    if base in EXCLUDE or base.endswith(("UP","DOWN","BULL","BEAR")):
        continue
    cands.append((base, sym, d["quoteVolume"]))
cands.sort(key=lambda x: -x[2])
cands = cands[:N_CANDIDATES]
print(f"Top {len(cands)} liquid non-stable USDT candidates on binanceus\n")

kept = []
for base, sym, vol in cands:
    try:
        df = fetch_full_history(sym, TF, START, "binanceus", store=True)
        df = df[df.index >= pd.Timestamp(START)]
    except Exception as e:
        print(f"  {base:7s} FETCH FAIL {str(e)[:50]}"); continue
    if len(df) < MIN_BARS:
        print(f"  {base:7s} skip ({len(df)} bars, too short)"); continue
    r = df["close"].pct_change().dropna()
    annvol = r.std() * np.sqrt(365 * 6) * 100
    kept.append(dict(coin=base, symbol=sym, bars=len(df),
                     start=str(df.index[0].date()), annvol=round(annvol, 0)))

kept.sort(key=lambda x: x["annvol"])
pd.DataFrame(kept).to_csv("research/results/universe.csv", index=False)
with open("research/universe.json", "w") as f:
    json.dump([k["symbol"] for k in kept], f, indent=0)

print(f"\n=== UNIVERSE: {len(kept)} coins with >= {MIN_BARS} bars ===")
print(f"  {'coin':7s} {'bars':>6s} {'start':>11s} {'ann.vol%':>9s}")
for k in kept:
    print(f"  {k['coin']:7s} {k['bars']:>6d} {k['start']:>11s} {k['annvol']:>9.0f}")

full = [k for k in kept if k["start"] <= "2023-01-02"]
print(f"\n  coins with FULL history (from 2023-01): {len(full)}/{len(kept)}")
print(f"  vol tiers (terciles): low <= {np.quantile([k['annvol'] for k in kept],1/3):.0f}%, "
      f"high > {np.quantile([k['annvol'] for k in kept],2/3):.0f}%")
