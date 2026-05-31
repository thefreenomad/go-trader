"""Bidirectional (long+SHORT) sweep, gated on regime — the closest backtestable
approximation of how the bot runs perps in a bear.

Each strategy's signal is turned into a target-position state machine:
  signal +1 -> hold long,  signal -1 -> hold short,  signal 0 -> hold last side.
On a flip, close fully (close_fraction=1) and open the opposite side same bar.
Injecting open_action/close_fraction columns activates the backtester's
short-capable open/close path (backtester.py:858). 1x, UNLEVERED (the backtester
carries no leverage context); no perp funding modeled. HL perps fees (0.035% taker,
0.05% slippage).

Regime modes:
  none  — bidirectional, no gate
  trend — allowed=['trending_up','trending_down']  (long up / short down, flat ranging)
  down  — allowed=['trending_down']                (only trade downtrends -> mostly shorts)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest")
sys.path.insert(0, "shared_tools")

import numpy as np
import pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry
from backtester import Backtester

SINCE = "2025-05-31"
CAPITAL = 10000.0
PLATFORM = "hyperliquid"          # perps fee model for the short scenario
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
          "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
TIMEFRAMES = ["1d", "4h"]
MIN_TRADES = 4
MODES = {"none": None,
         "trend": ["trending_up", "trending_down"],
         "down": ["trending_down"]}

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


def make_bidir_columns(df):
    """signal -> open_action / close_fraction for a long+short state machine."""
    sig = df["signal"].fillna(0).astype(int)
    desired = sig.replace(0, np.nan).ffill().fillna(0).astype(int)  # hold last non-zero
    prev = desired.shift(1).fillna(0).astype(int)
    open_action = pd.Series("none", index=df.index, dtype=object)
    open_action[(desired == 1) & (prev != 1)] = "long"
    open_action[(desired == -1) & (prev != -1)] = "short"
    close_fraction = pd.Series(0.0, index=df.index)
    close_fraction[(desired != prev) & (prev != 0)] = 1.0   # close old side on flip
    out = df.copy()
    out["open_action"] = open_action
    out["close_fraction"] = close_fraction
    return out


rows = []
bench = {}
for tf in TIMEFRAMES:
    for sym in ASSETS:
        df0 = load_cached_data(sym, tf, start_date=SINCE)
        df0 = df0[df0.index >= pd.Timestamp(SINCE)]
        if df0.empty or len(df0) < 60:
            continue
        bret, bsharpe = benchmark(df0, tf)
        bench[(sym, tf)] = (bret, bsharpe)
        for name in STRATS:
            try:
                sigdf = reg.apply_strategy(name, df0, reg.STRATEGY_REGISTRY[name]["default_params"])
                coldf = make_bidir_columns(sigdf)
            except Exception:
                continue
            rec = {"strategy": name, "asset": sym.split("/")[0], "tf": tf,
                   "bench_ret": bret, "bench_sharpe": bsharpe}
            ok = True
            for mode, allowed in MODES.items():
                try:
                    bt = Backtester(
                        initial_capital=CAPITAL, platform=PLATFORM,
                        open_strategy={"name": name, "params": {}},
                        strategy_type="perps",
                        regime_enabled=(allowed is not None),
                        allowed_regimes=allowed)
                    res = bt.run(coldf.copy(), strategy_name=name, symbol=sym, timeframe=tf)
                except Exception:
                    ok = False
                    break
                rec[f"ret_{mode}"] = res["total_return_pct"]
                rec[f"shp_{mode}"] = res["sharpe_ratio"]
                rec[f"trd_{mode}"] = res["total_trades"]
            if ok:
                rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("research/results/bidir_sweep_results.csv", index=False)


def positive(mode):
    return (df[f"ret_{mode}"] > 0) & (df[f"trd_{mode}"] >= 10) & (df[f"shp_{mode}"] > 0)


print(f"\n=== BIDIRECTIONAL (long+short) + REGIME GATE ({len(df)} configs, "
      f"trailing 12mo, 1x unlevered, HL fees) ===")
print(f"Benchmark: every asset down, mean B&H {df.groupby('asset')['bench_ret'].first().mean():+.1f}%\n")
print(f"{'mode':6s} {'avg_ret':>8s} {'avg_shp':>8s} {'avg_trd':>8s} {'profitable(>0,>=10trd)':>22s}")
for mode in MODES:
    print(f"{mode:6s} {df[f'ret_{mode}'].mean():>7.1f}% {df[f'shp_{mode}'].mean():>8.2f} "
          f"{df[f'trd_{mode}'].mean():>8.1f}   {positive(mode).sum():>10d} / {len(df)}")

for mode in ["trend", "down"]:
    print(f"\n=== PROFITABLE under '{mode}' gate (bidirectional; >0 return, >=10 trades, +Sharpe) ===")
    p = df[positive(mode)].sort_values(f"shp_{mode}", ascending=False)
    cols = ["strategy", "asset", "tf", f"ret_{mode}", f"shp_{mode}", f"trd_{mode}"]
    if len(p):
        print(p[cols].to_string(index=False))
    else:
        print("  (none)")

print("\n=== Best strategies by avg Sharpe across assets (mode='down', short-the-bear) ===")
agg = (df.groupby("strategy")
       .agg(avg_ret_down=("ret_down", "mean"), avg_shp_down=("shp_down", "mean"),
            avg_trd_down=("trd_down", "mean"),
            avg_shp_trend=("shp_trend", "mean"))
       .sort_values("avg_shp_down", ascending=False))
with pd.option_context("display.width", 200, "display.max_rows", 40,
                       "display.float_format", lambda x: f"{x:.2f}"):
    print(agg.to_string())
