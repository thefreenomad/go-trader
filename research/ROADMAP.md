# Roadmap — regime-conditioned cross-sectional momentum

Marrying our validated edge (cross-sectional momentum) with the repo's existing
regime engine (`shared_tools/regime.py` composite 7-state classifier + live
`allowed_regimes` gating). Architecture: **selection layer** (our cron job ranks
+ regime-gates + sizes → writes `scheduler/config.json`) on top of go-trader as the
**execution + risk layer**. The composite classifier is the shared vocabulary.

## Task order

- [x] **1. Wire repo classifier into the harness.** DONE (regime_xs.py). Finding: composite ranging states are vestigial on 4h (default thresholds) -> effectively 4 trending states; retune thresholds to recover ranging. Use `compute_regime_composite` /
      `ensure_regime_columns` (`shared_tools/regime.py`) to label universe bars + a
      market aggregate with the real 7 states. Sanity-check label distribution
      (not 95% one state); pick sensible regime period/window. *(in progress)*
- [x] **2. Conditional-performance measurement.** DONE. Market regime discriminates strongly: XS-momentum dies in trending_down_choppy (weak bear, Sharpe -2.0), works in clean trends. Gating off weak-bear lifts 4w config 1.08->1.80, bear -0.78->~0. Per-asset leg conditioning mild; market gate is the lever.
      (a) Market-level: XS-momentum forward performance in each of the 7 states →
          which are "trade" vs "sit out"; validate `ranging_quiet → flat`.
      (b) Per-asset: condition legs on each coin's own regime; beats raw ranking?
      (c) Prune states that don't discriminate; judge by rolling-window Sharpe
          distribution (not just 2 macro halves).
- [x] **>>> CHECKPOINT: PASSED.** Regime gate yields sensible, economically-motivated
      improvement (validates strong/weak-bear taxonomy). Continue.
- [ ] **3. Build & re-validate** the regime-conditioned deployable strategy from only
      what step 2 proved. REGIME GATE = BREADTH/dispersion from the universe (NOT BTC alone):
      breadth high -> Sharpe 3+, mid -> -1.8, low -> ~0. Gate to avoid the mushy middle; set
      breadth thresholds CAUSALLY (rolling quantiles). Also: deployable 4w XS-mom. Full gauntlet: bull/bear + rolling dist + vs
      always-on + vs random + net of funding. Keep degrees of freedom small.
- [ ] **4. Config-writer pipeline.** Cron: rank → regime-gate → significance gate →
      write `scheduler/config.json` (per-coin entries). Apply the BREADTH gate at the
      SELECTION layer (deploy book or not); set per-coin `allowed_regimes` LOOSE — per-asset
      regime gating HURTS a relative cross-sectional signal (regime_perasset.py). SIGHUP.
- [ ] **5. Paper-shadow + significance gate** — reconcile predicted vs realized for
      weeks before any live capital.
- [ ] **6. Refine** (iterate on the above once forward evidence accrues).

## Disciplines (carry through all steps)
- Each regime state must EARN its existence (prune non-discriminating states).
- Only deploy VALIDATED edges (don't fill empty regime cells with unproven strategies).
- Keep degrees of freedom small (the decision-matrix overfit lesson).
- RETUNE composite thresholds so ranging states populate on 4h (currently vestigial).
- Validate with rolling-window distribution + vs random/null baseline, net of cost+funding.
