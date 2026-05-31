"""Walk-forward parameter optimization — does re-tuning params weekly add real
out-of-sample value, or just overfit?

For each (asset, strategy, direction): every 7d, pick the param combo with the
best trailing-30d Sharpe, realize the next 7d with it, chain the blocks. Compare:
  WFO     — weekly re-optimized params
  default — fixed default params
  random  — random param combo each week (control)
Also report the in-sample (trailing-best) Sharpe vs its realized out-of-sample
Sharpe — the overfit gap.

Vectorized, fee-aware, 1x, direction-isolated, no funding. 4h, 2023->now.
"""
import sys, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "backtest"); sys.path.insert(0, "shared_tools")
import numpy as np, pandas as pd
from data_fetcher import load_cached_data
from run_backtest import load_registry
from optimizer import DEFAULT_PARAM_RANGES, generate_param_grid

START = "2023-01-01"; TF = "4h"; BARS_DAY = 6
TRAIN = 30 * BARS_DAY; STEP = 7 * BARS_DAY
COST = 0.00085; PPY = 365 * BARS_DAY; WEEKS_YR = 365 / 7
MIN_ACTIVE = 8; MAX_COMBOS = 60
ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT","LINK/USDT"]
DIRECTIONS = ["long","short"]
reg = load_registry("spot")
SPOT = set(reg.list_strategies())
TUNABLE = [s for s in DEFAULT_PARAM_RANGES if s in SPOT and
           len(generate_param_grid(DEFAULT_PARAM_RANGES[s])) <= MAX_COMBOS]


def cfg_net(close_pct, sig, direction):
    s = sig.fillna(0).astype(int)
    want = 1 if direction == "long" else -1
    held = (s.replace(0, np.nan).ffill().fillna(0).astype(int) == want)
    pos = (held.astype(float) * want).shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(abs(pos.iloc[0]))
    return (pos * close_pct - turn * COST).values, pos.abs().values


def block_chain(pick_fn, fwd, rebals, STEP):
    """Chain next-block returns using pick_fn(i)->combo index. Returns weekly series."""
    wk = []
    for k, i in enumerate(range(len(rebals))):
        c = pick_fn(i)
        wk.append(fwd[i, c] if c is not None else 0.0)
    return np.array(wk)


def wk_stats(wk):
    eq = np.cumprod(1 + wk)
    tot = (eq[-1] - 1) * 100
    sd = wk.std(ddof=1)
    shp = (wk.mean() / sd) * np.sqrt(WEEKS_YR) if sd > 0 else 0.0
    return tot, shp


rng = np.random.default_rng(7)
rows = []
for name in TUNABLE:
    grid = generate_param_grid(DEFAULT_PARAM_RANGES[name])
    default_p = reg.STRATEGY_REGISTRY[name]["default_params"]
    for sym in ASSETS:
        a = sym.split("/")[0]
        df = load_cached_data(sym, TF, start_date=START)
        df = df[df.index >= pd.Timestamp(START)]
        if len(df) < TRAIN + STEP:
            continue
        cpct = df["close"].pct_change().fillna(0.0)
        # signal per combo (signal is direction-independent)
        sigs = []
        for p in grid:
            try:
                sigs.append(reg.apply_strategy(name, df, p)["signal"])
            except Exception:
                sigs.append(pd.Series(0, index=df.index))
        try:
            sig_def = reg.apply_strategy(name, df, default_p)["signal"]
        except Exception:
            sig_def = pd.Series(0, index=df.index)
        nbars = len(df)
        rebals = list(range(TRAIN, nbars - STEP, STEP))
        if not rebals:
            continue
        for direction in DIRECTIONS:
            nets = np.column_stack([cfg_net(cpct, s, direction)[0] for s in sigs])  # bars x combos
            acts = np.column_stack([cfg_net(cpct, s, direction)[1] for s in sigs])
            ndef = cfg_net(cpct, sig_def, direction)[0]
            NR = len(rebals); G = nets.shape[1]
            tr_shp = np.full((NR, G), np.nan); fwd = np.zeros((NR, G)); elig = np.zeros((NR, G), bool)
            fwd_def = np.zeros(NR)
            for i, b in enumerate(rebals):
                tw = nets[b - TRAIN:b, :]; ac = acts[b - TRAIN:b, :].sum(axis=0)
                sd = tw.std(axis=0, ddof=1); mean = tw.mean(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tr_shp[i] = np.where(sd > 0, mean / sd, np.nan) * np.sqrt(PPY)
                elig[i] = ac >= MIN_ACTIVE
                fwd[i] = np.prod(1 + nets[b:b + STEP, :], axis=0) - 1
                fwd_def[i] = np.prod(1 + ndef[b:b + STEP]) - 1

            def pick_best(i):
                e = np.where(elig[i] & ~np.isnan(tr_shp[i]))[0]
                return e[np.argmax(tr_shp[i][e])] if e.size else None

            def pick_rand(i):
                e = np.where(elig[i])[0]
                return int(rng.choice(e)) if e.size else None

            wk_wfo = block_chain(pick_best, fwd, rebals, STEP)
            wk_rnd = block_chain(pick_rand, fwd, rebals, STEP)
            tot_wfo, shp_wfo = wk_stats(wk_wfo)
            tot_rnd, shp_rnd = wk_stats(wk_rnd)
            tot_def, shp_def = wk_stats(fwd_def)
            # in-sample (trailing-best) vs its realized OOS, per block
            is_shp = np.array([tr_shp[i][pick_best(i)] if pick_best(i) is not None else np.nan
                               for i in range(NR)])
            rows.append(dict(strategy=name, asset=a, direction=direction, combos=G,
                             shp_def=shp_def, shp_wfo=shp_wfo, shp_rnd=shp_rnd,
                             tot_def=tot_def, tot_wfo=tot_wfo, tot_rnd=tot_rnd,
                             is_shp=np.nanmean(is_shp)))

R = pd.DataFrame(rows)
R.to_csv("research/results/wfo_results.csv", index=False)

print(f"Tunable spot strategies tested: {len(TUNABLE)} (<= {MAX_COMBOS} combos each)")
print(f"Configs (strategy x asset x direction): {len(R)}\n")

print("=== DOES WALK-FORWARD PARAM OPTIMIZATION ADD OOS VALUE? (weekly Sharpe) ===")
print(f"  {'comparison':32s} {'mean Sharpe':>12s}")
print(f"  {'default params (fixed)':32s} {R.shp_def.mean():>12.3f}")
print(f"  {'WFO (weekly re-optimized)':32s} {R.shp_wfo.mean():>12.3f}")
print(f"  {'random param each week (control)':32s} {R.shp_rnd.mean():>12.3f}")
print(f"\n  WFO trailing-best IN-SAMPLE Sharpe (ann): {R.is_shp.mean():.2f}")
print(f"  -> realized OOS Sharpe:                    {R.shp_wfo.mean():.2f}   "
      f"(overfit tax {R.is_shp.mean()-R.shp_wfo.mean()*np.sqrt(1):.2f}+ in raw units)")
print(f"\n  WFO beats default (OOS Sharpe): {(R.shp_wfo>R.shp_def).mean()*100:.0f}% of configs")
print(f"  WFO beats random  (OOS Sharpe): {(R.shp_wfo>R.shp_rnd).mean()*100:.0f}% of configs")
print(f"  default beats random:           {(R.shp_def>R.shp_rnd).mean()*100:.0f}% of configs")

print("\n=== PER-STRATEGY (mean OOS weekly Sharpe across assets/directions) ===")
agg = (R.groupby("strategy")
       .agg(combos=("combos","first"), n=("asset","count"),
            shp_def=("shp_def","mean"), shp_wfo=("shp_wfo","mean"),
            shp_rnd=("shp_rnd","mean"))
       .assign(wfo_minus_def=lambda d: d.shp_wfo - d.shp_def)
       .sort_values("wfo_minus_def", ascending=False))
with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.3f}"):
    print(agg.to_string())
