# Lessons learned

Distilled, transferable principles from the research (the *why*, not the per-script
numbers — those live in `README.md`). Append as we clarify more.

## Methodology / anti-overfitting

1. **Benchmark choice makes or breaks a backtest.** "Positive return" means nothing on
   its own. Always compare against the *right* null: cash, buy-and-hold, naive-directional
   (e.g. naive-always-short in a bear), and **random selection**. Almost every apparent
   edge shrank to "real but not enough" once benchmarked properly.
2. **A single time-split is fooled by macro regime.** Our bull-train / bear-test split
   bisected the cycle, so the long book "overfit" (train Sharpe 1.34 → test −0.61) was
   partly just longs-tested-in-a-bear, and the short book "validated" was just
   short-beta-in-a-bear. Use **rolling-window Sharpe distributions** and **cross-asset
   holdouts**, not one chronological split, when the halves are opposite regimes.
3. **The random control is the cleanest test of skill.** If "deploy the winners" doesn't
   beat picking at random, the selection is noise. The rolling walk-forward only earned
   belief because it beat random by ~4σ.
4. **Every degree of freedom must EARN its existence.** Regime states, params,
   strategy-per-cell — each one is a chance to fit noise. The decision-matrix
   (regime×vol×direction → strategy) is the cautionary tale: gorgeous in-sample, −0.61
   out-of-sample. Keep state-count and knob-count small; prune what doesn't discriminate.
5. **The sub-period (bull/bear) acid test catches bull-overfit "improvements."** The
   multi-horizon blend and 12w lookback had higher *headline* Sharpe but went **negative
   in the bear** — rejected. Headline Sharpe will sell you a bull-loaded config; always
   split.
6. **Survivorship bias is fatal at universe scale.** Point-in-time eligibility (coins enter
   when listed) + keeping mid-period delistings (trade while live, drop when gone) reduces
   it. Testing only today's survivors inflates returns.
7. **Watch for disguised market-timing.** "clean-only" gating looked great on 4w but was
   secretly a market gate (sat out 50/87 cycles) and failed on 8w. If a rule's strength is
   really "it sits out sometimes," name that honestly and test the timing directly.
8. **Fix your accounting bugs before believing magnitudes.** A stats-from-gross-returns bug
   made costs/funding look free until caught. Re-derive when numbers look too clean.

## Strategy & edge

9. **Single-asset TA strategies have no standalone alpha** — they capture market beta
   inefficiently. The whole 28-strategy library, swept across assets/regimes, never beat
   the right benchmark.
10. **Weekly parameter optimization OVERFITS — worse than random params out-of-sample.**
    Trailing-best params look great in-sample, deliver ~zero next week. Use fixed defaults;
    don't re-optimize on short windows.
11. **The edge is cross-sectional SELECTION (ranking the universe), not single-asset
    strategies or parameter precision.** "Pick the best coins" is literally the edge.
12. **Cross-sectional momentum is a real crypto factor** (long winners / short losers).
    Confirmed by the **reversal mirror being strongly negative** — if the ranking were
    noise, reversal would be flat; instead it's the exact inverse.
13. **The risk layer adds real value, not just safety.** Inverse-vol leg weighting *raised*
    return and Sharpe; vol-targeting *halved* drawdowns while raising Sharpe.
14. **Deployability reframes the edge.** The all-weather/low-DD property depended on the
    *least* deployable parts (illiquid small-cap shorts). The deployable (liquid, long-tilt)
    version is a strong **bull-market** momentum edge, not a clean market-neutral machine.
15. **Funding is a material headwind for momentum** (it longs the highest-funding winners).
    Sharpe 1.6 → ~1.0 at realistic 20–30%/yr — and that model is optimistic (long leg only).
16. **Crypto momentum wants recent data — skip-recent (classic 12-1) HURTS.** Unlike
    equities, including the most recent window is better (continuation, not reversal).
17. **Match rebalance cadence to signal speed — don't fix one in isolation.** Early on,
    biweekly beat weekly *at a 4w lookback* (less whipsaw). But a full lookback×cadence
    grid (`param_sweep.py`/`deploy_compare.py`) showed the real structure: short momentum
    decays fast and needs a fast rebalance. **weekly/1w** (Sharpe ~1.4 bare, 1.24 @25bp,
    −18% maxDD) beats biweekly/4w net of cost and turnover; biweekly/1w collapses (0.49,
    stale ranking). Cadence and lookback are one joint choice, not two.

## Regime

18. **The repo's composite 7-state classifier is degenerate on 4h with default thresholds**
    (~98% trending, ranging <2%). Effectively a 4-state classifier until retuned.
19. **Regime conditioning helps modestly (validates the strong/weak-bear taxonomy).**
    XS-momentum dies in the **choppy/weak downtrend** (the "momentum crash" regime) and
    thrives in clean trends (incl. clean *down*trends). Sitting out the weak-bear looked like
    1.08→1.80 *in-sample*; **causally + net of funding it's a modest 0.72→0.87 (+0.9σ vs
    random) — see #26.** Real but small.
20. **For a RELATIVE (cross-sectional) strategy, gate at the PORTFOLIO/SELECTION level, not
    per-asset.** Per-asset absolute-regime gating fights the relative signal — it discards
    valid cross-sectional winners (strong-vs-peers coins still in downtrends) and *reduces*
    returns (1.08 → 0.65–0.85). This is the **opposite** of go-trader's native per-asset
    `allowed_regimes`. Two questions, two signals: *rank* = "trade this coin?"; *universe
    breadth* = "is the factor working now?".
21. **Breadth gate: in-sample mirage — DID NOT survive causal validation (see #25).** It
    looked dominant in-sample (high breadth → Sharpe 3+, mid → −1.8) but that used full-sample
    thresholds; causally it scored −0.1σ vs random and was *worse* than no gate. Left here as
    a cautionary record, not a recommendation.

## Architecture

22. **Two layers, regime classifier shared.** Our **selection layer** (rank → breadth-gate →
    size → write `config.json`) sits on top of go-trader as the **execution + risk layer**.
    Use the repo for what it's strong at; don't rebuild it.
23. **Automation is the LAST step.** The pipeline *harvests* an edge; it doesn't create one.
    Prove a robust edge first (cross-sectional momentum), then build the cron pipeline.
24. **Security:** the repo is clean — no data exfiltration in code, agent instructions, or
    docs. The only thing to manage is the opt-in **auto-update** path (supply-chain trust);
    keep it off and review diffs.

## Validation (Step 3 — the hard lessons)

25. **In-sample threshold selection creates mirages; causal + random-null validation is
    non-negotiable.** The breadth gate looked spectacular (in-sample Sharpe 3.2 in
    high-breadth) but FAILED causally — with rolling thresholds and benchmarked against
    sitting out *random* periods of the same rate, it scored −0.1σ vs random and was *worse*
    than no gate (0.32 vs 0.72). Its strength was full-sample tercile thresholds. Always run
    the same-sit-out-rate random null before believing a gate.
26. **Fitted-threshold gates overfit; fixed-classifier gates with economic backing can hold.**
    The composite weak-bear gate (sit out `trending_down_choppy`) modestly SURVIVED (+0.9σ vs
    random, lifted 0.72→0.87) — because its labels come from a fixed classifier (no fitted
    threshold); only the *state choice* was hindsight, and it has economic backing (momentum
    crash). The gate that looked *best* in-sample (breadth) died; the one that looked merely
    *good* (weak-bear) survived.
27. **The validated edge is real; magnitude uncertain.** Step 3 first put it at ~0.85 Sharpe,
    but that was deflated by a funding-model error (#28). Corrected (HL-native, signed funding,
    real slippage) it's stronger but inflated by a clean universe (#30). Honest estimate:
    ~1.0–1.8 market-neutral, good HL capacity. Real, beats random, reversal-confirmed.

## Deployability (Hyperliquid-native — corrections)

28. **Funding on a perp L/S book is SIGNED (longs pay, shorts RECEIVE).** Step 3's
    validate.py modeled funding one-sided on the long leg only → over-penalized by ~the full
    rate and deflated the Sharpe (the "modest 0.85" was partly a funding-model error). Correct
    signed treatment → ~0 net for a balanced book; the only real cost is the *adverse
    winner-vs-loser funding spread* (modest). Real HL funding confirms ~0 net effect.
29. **Execution-venue liquidity is decisive for capacity — model the venue you'll trade.**
    BinanceUS volumes ($2M BTC) falsely showed the edge dying at $50k AUM; real Hyperliquid
    volumes ($~1B BTC) show capacity to ~$20M with small slippage. Same strategy, opposite
    conclusion, purely from the liquidity data source.
30. **Beware universe-driven Sharpe inflation.** A clean large-cap-only HL universe (top-25 by
    HL volume) gave in-sample Sharpe ~2.78 vs ~0.9 on the broader 57-coin universe — likely
    sample-specific (those majors trended cleanly this cycle). Combined with the usual ~2x
    in-sample optimism, the realistic deployable edge is ~1.0–1.8, not 2.78. Don't anchor on
    the rosiest universe.
31. **Negative controls verify the engine, not just the edge.** reversal (flip the ranking) →
    Sharpe −3.16 (perfect inverse); random rank → ~0. Run these whenever a result looks too
    good — they distinguish real signal from a look-ahead bug.

## Squeeze / robustness (trade-data refinements)

32. **The signal is what it is — refinements were marginal-to-negative.** skip-oversold short
    helped the bull but cratered the bear (wash); per-name cap did nothing at 5 coins/leg;
    3-week holds were clearly worse. Don't over-engineer a real but modest edge.
33. **Expanding the universe is the real robustness lever (confirmed).** Top-25 -> top-40 HL
    dropped FET's P&L share 34% -> 20% AND dropped Sharpe 2.32 -> 1.45. The lower number is the
    HONEST one — the headline was partly single-name concentration. More breadth = a smaller,
    more trustworthy, more robust edge. Per-name caps only bite once the universe is large.
34. **Realistic deployable edge: ~1.5-1.9 Sharpe market-neutral** (expanded HL universe, net of
    HL costs, beats random +4σ). Robust, not spectacular.

## Paper-shadow (Step 5)

35. **The deployable config gives a CONSISTENT ~1.1 Sharpe (recent ≈ full).** Paper-shadowing
    the exact deployable engine (expanded HL universe + caps + weak-bear gate) net of real HL
    costs+funding: full-period 1.12, recent-9-months 1.11. The expanded universe + caps that
    deflated the inflated 2.78 produced an honest number that is NOT bull-loaded — it holds in
    the recent window. ~1.1 market-neutral is the realistic deployment expectation.

## Pre-deploy smoke test (expanded HL universe, past month)

36. **Hyperliquid's broad universe includes tokenized EQUITIES (XYZ-MU/AMD/INTC/PLTR/HOOD...).**
    "Expand the HL universe" naively pulls in stock perps. Mixed crypto+equity book gave a
    flattering +20%/Sharpe 4.3 over the past month -- but that was a semiconductor run, NOT the
    crypto edge. Crypto-only over the same month: +6%/Sharpe 1.60 (in our validated ~1.1-1.8
    range). Deploy crypto-only; tokenized-equity cross-sectional momentum is a SEPARATE future
    sleeve to validate independently -- don't rank DOGE against Intel without validation.
37. **Liquid crypto on HL is only ~30 perps (>$3M/day).** Crypto-only breadth is inherently
    limited on HL; the big universe expansion HL offers is equities. Breadth vs asset-class
    purity is a real tradeoff to decide deliberately.

## High-level params + the deployed system (Step 6-8)

38. **Optimize the DEPLOYED system, not the bare book — components that each help can fight
    when combined.** The 4h exit-decay monitor *lifts* a slow biweekly/4w book (0.88→1.14)
    but *craters* a fast weekly/1w one (1.40→0.71, 248 whipsaw closes/yr): a 1w rank
    recomputed every 4h is pure noise. A fast rebalance and an intra-cycle exit-monitor are
    two ways to buy the same responsiveness — stacking them double-trades. `deploy_compare.py`
    flipped the bare-book ranking; always backtest the *thing you'll actually run*.
39. **Commit to one config; don't adapt — realistic selection underperforms committing.**
    Walk-forward that re-picks the best lookback by trailing Sharpe (no hindsight) scored
    0.30 mean OOS Sharpe vs **1.02 for always-weekly/1w** and 0.83 for always-bi/4w —
    trailing-Sharpe selection is too noisy and whipsaws (mirror of #10, now at the
    config level). Pick the best *fixed* config and hold it.
40. **weekly/1w validated OOS, but it's higher-octane.** Positive in all 4 sequential
    sub-periods and best fixed OOS total (123%), so not luck — but more variable than
    bi/4w (5/9 vs 7/9 OOS folds positive) and weakest in the deep-bear stretch (2025-Q2),
    where the slower book is more defensive. Expect bigger swings; watch it in bear regimes.
