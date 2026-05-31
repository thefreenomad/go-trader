# research/ — strategy evaluation & autonomous selection

Backtest research toward a **process-driven, autonomously-rebalanced** trading system:
on a rolling cadence (daily/weekly), re-evaluate a universe of assets, find the best
params, filter what to trade, and update the live configs — `f(regime, volatility,
asset, direction, params) → deploy`.

This directory is **our additions on top of the forked `go-trader`** (upstream:
`richkuo/go-trader`). Scripts run from the **repo root** with the project venv, e.g.:

```bash
PYTHONPATH=. .venv/bin/python research/rolling_wf.py
```

Data: public OHLCV via ccxt (BinanceUS), cached in `shared_tools/trading_bot.db`
(gitignored). All backtests are **1× unlevered, no perp funding modeled** unless noted.

## Scripts (in the order they were built)

| script | question it answers | headline finding |
|---|---|---|
| `alpha_sweep.py` | Do the strategies beat buy&hold on 8 assets / last 12mo (long-only, no regime)? | No. ~3% of configs made money; "winners" don't generalize across assets. Window was a bear, so most "beats B&H" = lost less by sitting in cash. |
| `regime_sweep.py` | Does the ADX regime gate add value (long-only)? | Yes — `trending_up`-only gate ~halved avg loss and lifted Sharpe. It's a real loss-filter, not an alpha engine. |
| `bidir_sweep.py` | Does shorting (gated on regime) help in a bear? | Looks great (avg Sharpe −0.5→~0) but it's **short-beta**: only ~26% of profitable short configs beat a *naive always-short* benchmark. |
| `decision_matrix.py` | Static `f(regime, vol_tier, direction)→strategy` with train/test split | **Overfit demo.** Long book: train Sharpe 1.34 → test −0.61 (1/9 cells survive). The split bisected the cycle (bull train / bear test), conflating edge with beta. |
| `decision_matrix_v2.py` | Same, but **beta-relative scoring + cross-asset holdout** | Removes both confounds. (Superseded by the rolling approach before full analysis.) |
| `rolling_wf.py` | **Rolling walk-forward** (30d train / 7d test / 7d step) — the live operating loop | **Faint but real edge:** trailing-return selection at K≥5 beats *random* weekly selection by +3–4σ. But best variant (+54% / 3.3yr) loses badly to BTC hold (+236%) with −70% drawdowns. |
| `wfo.py` | Does weekly **walk-forward parameter optimization** add OOS value? | **No — it overfits.** WFO weekly Sharpe (−0.073) beats fixed defaults (−0.087) but *loses to random param selection* (−0.058). Trailing-best params look great in-sample, deliver ~zero OOS. Short-window param tuning is a curve-fit trap; use fixed defaults. |
| `universe.py` | Build expanded liquid universe (~58 coins, 4h, 2023→now) | Vol spreads cleanly (BTC 46% → small-caps 200%+); 38 full-history + 20 late-listers. Survivorship-biased. |
| `xsmom.py` | Cross-sectional momentum (v1, 37 coins, equal-weight) | **First real edge.** Beats random at every lookback (12w: +49% / Sharpe 0.52 vs random −17%); the reversal mirror is strongly negative, confirming directional signal. Market-neutral, modest Sharpe, still −57% DD (needs risk layer). "Pick the best coins" IS the edge — ranking across assets, not single-asset strategies. |
| `xsmom_v2.py` | **XS-momentum + breadth fix + risk layer** | **Best result.** 57 coins point-in-time; 8w lookback, inverse-vol + vol-targeting → +101% / **Sharpe 0.86** / −32% DD / **BTC-corr 0.18**. Vol-target halved drawdowns while raising Sharpe; beats random (−23%) by a chasm. Market-neutral, leverable, near-uncorrelated to BTC. Short-side realism (funding/borrow) still to address. |
| `xsmom_robust.py` | **Robustness lock** — sub-period (bull/bear), cost, quantile, skip, blend, rebalance-freq | **Passes the acid test.** Locked config **8w / inverse-vol / vol-target / BIWEEKLY / q20 / no-skip**: Sharpe **1.72 bull, 1.00 bear**, −14% DD, cost-robust to 40bps. Sub-period split CAUGHT two bull-overfit "improvements" (multi-horizon blend and 12w lookback go negative in the bear). Intermediate lookback required. |
| `xsmom_deploy.py` | **Deployability** — liquidity restriction, direction mode, funding | **Honest reframe.** All-weather property (bear Sharpe 1.0, −14% DD) needs the FULL universe + short leg; restrict to liquid coins or go long-only → bull Sharpe ~3 but **bear ~0/negative**. Funding is material (Sharpe 1.59→~1.0 at 20-30%/yr on the long leg). It is a strong **bull-market** momentum edge, not a clean market-neutral machine. → pair with the regime gate (run in favorable regime, flatten in bear). |

Results CSVs are in `results/`.

## Where the evidence leaves us

- Individual **single-asset** strategies have **no standalone alpha**; they capture market
  beta inefficiently. Weekly **param optimization overfits** (worse than random).
- The **edge is cross-sectional**: ranking the *universe* and trading the best/worst coins.
  Cross-sectional momentum (`xsmom_v2.py`) with inverse-vol weighting + vol-targeting reaches
  **Sharpe ~0.86, −32% DD, ~0.18 BTC-correlation**, market-neutral, beating a random control
  decisively. "Pick the best coins" *is* the edge — exactly the project thesis.
- The **regime gate** remains a useful loss-filter for the single-asset/directional path.
- This is now a real, harvestable, near-uncorrelated factor — investable as a diversifying,
  leverable sleeve. Robustness LOCKED (`xsmom_robust.py`): 8w / inverse-vol / vol-target / biweekly is all-weather
  (Sharpe 1.72 bull, 1.00 bear, −14% DD, cost-robust); blend/long lookbacks are bull-overfit and were rejected.
  Remaining work is **deployability**: short-side realism (funding/borrow) and the autonomous pipeline.

## Where we're lacking (roadmap)

1. **Edge source** — only time-series selection tested. **Cross-sectional momentum**
   (rank the universe, long strongest/short weakest) is the documented crypto factor and
   is literally "pick the best coins" — untested, highest-leverage next.
2. **Universe & data** — only 8 majors. Need a **liquidity-screened, survivorship-aware**
   pipeline for 100s of coins (survivorship bias is fatal otherwise).
3. **Cost realism** — no funding, flat slippage. At scale, costs decide the outcome.
4. **Risk layer** — no vol-targeting / correlation-aware sizing / drawdown control
   (hence −70% DD).
5. **Param optimization** — TESTED (`wfo.py`): weekly walk-forward param tuning **overfits**
   (loses to random params OOS). **Dropped** from the pipeline — use fixed defaults, or only
   long-window/robust optimization. Not a short-cadence step.
6. **Anti-overfit gate** — a large-universe scan will surface spurious winners every cycle;
   deploy only configs that beat their null/random baseline out-of-sample. This gate is the
   backbone that keeps the autonomous system from becoming an automated curve-fit.

## Target architecture (cron-driven, autonomous)

```
cron → 1. refresh universe OHLCV (100s coins)
       2. eligibility screen (liquidity, listing age, survivorship)
       3. per-asset regime + volatility classification
       4. walk-forward param optimization per (asset, strategy)
       5. rank (cross-sectional + time-series) → select what to trade
       6. risk/size/allocate (vol-target, correlation-aware)
       7. significance gate (beat null baseline OOS, else drop)
       8. write scheduler/config.json → SIGHUP reload
       9. paper-shadow + live-vs-backtest reconciliation
```

Build/validate in that order; **automation is the last step, not the first.**
