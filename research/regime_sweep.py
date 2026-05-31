"""Regime-gate comparison: same long-only strategies as alpha_sweep, but now
with the ADX regime gate ON. Tests whether `allowed_regimes` adds value.

Three modes per (strategy, asset, tf):
  none      — no regime gate (baseline, matches alpha_sweep)
  up        — allowed_regimes=['trending_up']         (only long confirmed uptrends)
  trend     — allowed_regimes=['trending_up','trending_down'] (block only 'ranging')

Default ADX params (period 14, threshold 20) = the bot's default regime block.
Net of fee+slippage. Trailing 12 months.
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest")
sys.path.insert(0, "shared_tools")

import numpy as np
import pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry, run_single_backtest

SINCE = "2025-05-31"
CAPITAL = 10000.0
PLATFORM = "binanceus"
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
          "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
TIMEFRAMES = ["1d", "4h"]
MIN_TRADES = 4
MODES = {
    "none": None,
    "up": ["trending_up"],
    "trend": ["trending_up", "trending_down"],
}

reg = load_registry("spot")
STRATS = [s for s in reg.list_strategies() if s != "hold"]


def ppy(tf):
    return {"1d": 365, "4h": 365 * 6}.get(tf, 365)


def benchmark(df, tf):
    c = df["close"]
    ret = c.iloc[-1] / c.iloc[0] - 1
    r = c.pct_change().dropna()
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy(tf)) if r.std() > 0 else 0.0
    return ret * 100, sharpe


rows = []
bench = {}
for tf in TIMEFRAMES:
    for sym in ASSETS:
        df = load_cached_data(sym, tf, start_date=SINCE)
        df = df[df.index >= pd.Timestamp(SINCE)]
        if df.empty or len(df) < 60:
            continue
        bret, bsharpe = benchmark(df, tf)
        bench[(sym, tf)] = (bret, bsharpe)
        for name in STRATS:
            rec = {"strategy": name, "asset": sym.split("/")[0], "tf": tf,
                   "bench_ret": bret, "bench_sharpe": bsharpe}
            ok = True
            for mode, allowed in MODES.items():
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        res = run_single_backtest(
                            name, sym, tf, SINCE, CAPITAL,
                            registry="spot", platform=PLATFORM,
                            regime_enabled=(allowed is not None),
                            allowed_regimes=allowed)
                except Exception:
                    ok = False
                    break
                if not res:
                    ok = False
                    break
                rec[f"ret_{mode}"] = res["total_return_pct"]
                rec[f"shp_{mode}"] = res["sharpe_ratio"]
                rec[f"trd_{mode}"] = res["total_trades"]
            if ok:
                rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("research/results/regime_sweep_results.csv", index=False)


def beats(mode):
    return ((df[f"ret_{mode}"] > df["bench_ret"]) &
            (df[f"shp_{mode}"] > df["bench_sharpe"]) &
            (df[f"trd_{mode}"] >= MIN_TRADES))


def positive(mode):
    return (df[f"ret_{mode}"] > 0) & (df[f"trd_{mode}"] >= 10) & (df[f"shp_{mode}"] > 0)


print(f"\n=== REGIME-GATE COMPARISON ({len(df)} configs, trailing 12mo, long-only) ===\n")
print(f"{'mode':6s} {'avg_ret':>8s} {'avg_shp':>8s} {'avg_trd':>8s} "
      f"{'beat_BH':>8s} {'profitable(>0,>=10trd)':>22s}")
for mode in MODES:
    print(f"{mode:6s} {df[f'ret_{mode}'].mean():>7.1f}% {df[f'shp_{mode}'].mean():>8.2f} "
          f"{df[f'trd_{mode}'].mean():>8.1f} {beats(mode).sum():>6d}   "
          f"{positive(mode).sum():>10d} / {len(df)}")

print("\n=== Per-mode: did gating raise risk-adjusted return vs no-gate? ===")
print(f"  avg Sharpe  none={df['shp_none'].mean():.3f}  "
      f"up={df['shp_up'].mean():.3f}  trend={df['shp_trend'].mean():.3f}")
print(f"  median trades  none={df['trd_none'].median():.0f}  "
      f"up={df['trd_up'].median():.0f}  trend={df['trd_trend'].median():.0f}")
# Per-config: did 'up' gate improve Sharpe vs none?
df["d_shp_up"] = df["shp_up"] - df["shp_none"]
df["d_shp_trend"] = df["shp_trend"] - df["shp_none"]
print(f"\n  configs where 'up' gate IMPROVED Sharpe: {(df.d_shp_up>0).sum()}/{len(df)} "
      f"({100*(df.d_shp_up>0).mean():.0f}%);  mean ΔSharpe {df.d_shp_up.mean():+.3f}")
print(f"  configs where 'trend' gate IMPROVED Sharpe: {(df.d_shp_trend>0).sum()}/{len(df)} "
      f"({100*(df.d_shp_trend>0).mean():.0f}%);  mean ΔSharpe {df.d_shp_trend.mean():+.3f}")

print("\n=== PROFITABLE under 'up' gate (positive return, >=10 trades, +Sharpe) ===")
pu = df[positive("up")].sort_values("shp_up", ascending=False)
if len(pu):
    print(pu[["strategy", "asset", "tf", "ret_up", "shp_up", "trd_up",
              "ret_none", "shp_none", "trd_none"]].to_string(index=False))
else:
    print("  (none)")

print("\n=== Strategies most helped by the 'up' gate (avg ΔSharpe vs no-gate) ===")
agg = (df.groupby("strategy")
       .agg(d_shp_up=("d_shp_up", "mean"),
            shp_none=("shp_none", "mean"),
            shp_up=("shp_up", "mean"),
            trd_none=("trd_none", "mean"),
            trd_up=("trd_up", "mean"))
       .sort_values("d_shp_up", ascending=False))
with pd.option_context("display.width", 200, "display.max_rows", 40,
                       "display.float_format", lambda x: f"{x:.2f}"):
    print(agg.to_string())
