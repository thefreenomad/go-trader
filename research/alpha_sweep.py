"""Ad-hoc alpha sweep: every spot strategy x every asset x timeframe over the
trailing ~12 months, scored against that asset's buy-and-hold benchmark.

Default params only (no curve-fitting) -> the whole window is effectively
out-of-sample for the strategy logic. Net of the backtester's fee+slippage model.
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest")
sys.path.insert(0, "shared_tools")

import numpy as np
import pandas as pd
from data_fetcher import fetch_full_history, load_cached_data
from run_backtest import load_registry, run_single_backtest

SINCE = "2025-05-31"            # trailing ~12 months (today ~2026-05-31)
CAPITAL = 10000.0
PLATFORM = "binanceus"
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
          "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
TIMEFRAMES = ["1d", "4h"]
MIN_TRADES = 4                 # ignore configs with too few trades to mean anything

reg = load_registry("spot")
STRATS = [s for s in reg.list_strategies() if s != "hold"]


def periods_per_year(tf):
    return {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}.get(tf, 365)


def benchmark(df, tf):
    c = df["close"]
    ret = c.iloc[-1] / c.iloc[0] - 1
    peak = c.cummax()
    mdd = (c / peak - 1).min()
    r = c.pct_change().dropna()
    ppy = periods_per_year(tf)
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    return ret * 100, sharpe, mdd * 100


rows = []
bench = {}
for tf in TIMEFRAMES:
    for sym in ASSETS:
        try:
            df = load_cached_data(sym, tf, start_date=SINCE)
            if df.empty or len(df) < 60:
                df = fetch_full_history(sym, tf, SINCE, PLATFORM, store=True)
            df = df[df.index >= pd.Timestamp(SINCE)]
        except Exception as e:
            print(f"[skip] {sym} {tf}: fetch failed: {e}", file=sys.stderr)
            continue
        if df.empty or len(df) < 60:
            print(f"[skip] {sym} {tf}: only {len(df)} bars", file=sys.stderr)
            continue
        bret, bsharpe, bmdd = benchmark(df, tf)
        bench[(sym, tf)] = (bret, bsharpe, bmdd, len(df))
        for name in STRATS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = run_single_backtest(
                        name, sym, tf, SINCE, CAPITAL,
                        registry="spot", platform=PLATFORM)
            except Exception as e:
                continue
            if not res:
                continue
            rows.append({
                "strategy": name, "asset": sym.split("/")[0], "tf": tf,
                "ret": res["total_return_pct"], "sharpe": res["sharpe_ratio"],
                "mdd": res["max_drawdown_pct"], "trades": res["total_trades"],
                "win": res["win_rate"],
                "bench_ret": bret, "bench_sharpe": bsharpe, "bench_mdd": bmdd,
            })

df = pd.DataFrame(rows)
df["alpha_ret"] = df["ret"] - df["bench_ret"]            # excess return vs B&H
df["alpha_sharpe"] = df["sharpe"] - df["bench_sharpe"]    # risk-adj edge vs B&H
# "real alpha" = beats B&H on BOTH return and Sharpe with enough trades
df["beats_bh"] = (df["alpha_ret"] > 0) & (df["alpha_sharpe"] > 0) & (df["trades"] >= MIN_TRADES)
df.to_csv("research/results/alpha_sweep_results.csv", index=False)

print("\n=== BUY & HOLD BENCHMARKS (trailing 12mo, net) ===")
for (sym, tf), (r, s, m, n) in sorted(bench.items()):
    print(f"  {sym:10s} {tf:3s}  ret {r:+7.1f}%  sharpe {s:5.2f}  maxDD {m:6.1f}%  ({n} bars)")

print(f"\n=== SWEEP: {len(df)} configs ({df['strategy'].nunique()} strategies x "
      f"{df['asset'].nunique()} assets x {df['tf'].nunique()} tf) ===")
print(f"Min trades filter: {MIN_TRADES}")

winners = df[df["beats_bh"]].sort_values("alpha_sharpe", ascending=False)
print(f"\n=== CONFIGS BEATING BUY&HOLD on BOTH return AND Sharpe: {len(winners)} / {len(df)} "
      f"({100*len(winners)/max(len(df),1):.1f}%) ===")
cols = ["strategy", "asset", "tf", "ret", "bench_ret", "alpha_ret",
        "sharpe", "bench_sharpe", "alpha_sharpe", "mdd", "trades", "win"]
with pd.option_context("display.width", 200, "display.max_rows", 60,
                       "display.float_format", lambda x: f"{x:.2f}"):
    print(winners[cols].to_string(index=False))

print("\n=== TOP 15 BY RISK-ADJUSTED ALPHA (alpha_sharpe), any trade count ===")
top = df.sort_values("alpha_sharpe", ascending=False).head(15)
with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.2f}"):
    print(top[cols].to_string(index=False))

print("\n=== PER-STRATEGY AVG ACROSS ALL ASSETS/TF (consistency check) ===")
agg = (df.groupby("strategy")
       .agg(avg_alpha_ret=("alpha_ret", "mean"),
            avg_alpha_sharpe=("alpha_sharpe", "mean"),
            avg_sharpe=("sharpe", "mean"),
            pct_beating_bh=("beats_bh", "mean"),
            avg_trades=("trades", "mean"),
            n=("strategy", "count"))
       .sort_values("avg_alpha_sharpe", ascending=False))
with pd.option_context("display.width", 200, "display.max_rows", 40,
                       "display.float_format", lambda x: f"{x:.2f}"):
    print(agg.to_string())
