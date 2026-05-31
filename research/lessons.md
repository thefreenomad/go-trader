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
17. **Less-frequent rebalancing (biweekly) beat weekly** by removing whipsaw — but
    re-validate sub-period, because biweekly can be bull-loaded.

## Regime

18. **The repo's composite 7-state classifier is degenerate on 4h with default thresholds**
    (~98% trending, ranging <2%). Effectively a 4-state classifier until retuned.
19. **Regime conditioning works and validates the strong/weak-bear taxonomy.** XS-momentum
    dies in the **choppy/weak downtrend** (the "momentum crash" regime) and thrives in clean
    trends (incl. clean *down*trends). Sitting out the weak-bear lifted 4w Sharpe 1.08→1.80
    and neutralized the bear.
20. **For a RELATIVE (cross-sectional) strategy, gate at the PORTFOLIO/SELECTION level, not
    per-asset.** Per-asset absolute-regime gating fights the relative signal — it discards
    valid cross-sectional winners (strong-vs-peers coins still in downtrends) and *reduces*
    returns (1.08 → 0.65–0.85). This is the **opposite** of go-trader's native per-asset
    `allowed_regimes`. Two questions, two signals: *rank* = "trade this coin?"; *universe
    breadth* = "is the factor working now?".
21. **Breadth/dispersion beats BTC price-trend as the gate.** Universe breadth (% of coins
    with positive momentum): high → Sharpe 3+, mid (transition) → −1.8, low → ~0. The
    cross-sectional factor needs broad dispersion/leadership; it dies in the mushy middle.
    (Set breadth thresholds **causally** for deployment — rolling quantiles, not in-sample.)

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
27. **The validated edge is MODEST.** Net of funding, the deployable strategy is ~0.85 Sharpe
    market-neutral (not the 1.8–3.2 the in-sample suggested). Real, beats random,
    reversal-confirmed — but modest. In-sample numbers ran ~2x optimistic; the truth is ~0.8.
