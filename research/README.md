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

Results CSVs are in `results/`.

## Where the evidence leaves us

- The individual strategies have **no standalone alpha**; they capture market beta inefficiently.
- The **regime gate is the one genuinely value-adding feature** (loss reduction).
- There **is** a small, statistically-real short-term *persistence* edge (recent winners
  keep winning ~1 week out: beats random by ~4σ), best harvested **diversified** (K=10 > K=1).
- It is **too weak to beat holding BTC** and carries severe drawdowns. As a standalone
  money-maker it fails; its only plausible value is as a risk-managed, uncorrelated sleeve.
- Every honest benchmark (cash, naive-short, random, BTC-hold) shrinks the edge to
  "real but not enough."

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
