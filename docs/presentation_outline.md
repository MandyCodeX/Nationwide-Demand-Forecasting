# 10-minute presentation outline (8–9 slides)
1. **Problem (1 min)** — stockouts vs waste; why forecasts alone are not enough → decisions.
2. **Data & hierarchy (1 min)** — M5, item/store/state, subset used, intermittency (EDA chart).
3. **Approach / architecture (1 min)** — `docs/architecture.png`.
4. **Features & leakage control (1 min)** — lags ≥ 28 so one direct model forecasts 28 days; test shows no leakage.
5. **Models & accuracy (2 min)** — baselines vs LightGBM vs TFT; WAPE/RMSSE table; honest takeaway.
6. **Reconciliation & intervals (1 min)** — coherence before/after; coverage vs target.
7. **Inventory optimisation (1.5 min)** — formulation, policies A–D, cost / fill-rate trade-off.
8. **Live demo (1 min)** — Dash: pick a store, move the budget slider, show order list.
9. **Challenges, results, next steps (0.5 min)** — what failed, what you'd do with more time.
