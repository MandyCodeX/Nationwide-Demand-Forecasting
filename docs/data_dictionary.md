# Data dictionary

## Input (M5 / Walmart format — same layout for the synthetic stand-in)
| File | Key columns |
|---|---|
| `calendar.csv` | `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1/2, event_type_1/2, snap_CA/TX/WI` |
| `sell_prices.csv` | `store_id, item_id, wm_yr_wk, sell_price` (weekly price; missing = product not yet on sale) |
| `sales_train_evaluation.csv` | `id, item_id, dept_id, cat_id, store_id, state_id, d_1 … d_N` (daily units) |

Hierarchy: **state → store → category → department → item**; one series = one item in one store (`id`).

## Working tables (`data/processed/`)
| File | Content |
|---|---|
| `long.parquet` | tidy long table for the chosen subset: one row per series-day with price, calendar, SNAP |
| `features.parquet` | `long` + engineered features |
| `feature_list.json` | model feature names |

## Features (all sales-based features are shifted ≥ 28 days → no leakage)
| Feature | Meaning |
|---|---|
| `lag_28/35/42/49/56` | sales exactly 28…56 days earlier |
| `dow_mean_4` | mean of the same weekday over the last 4 weeks (shifted) |
| `rmean_7/14/28/56`, `rstd_28` | rolling mean / std of sales (series shifted by 28) |
| `rzero_28` | share of zero-sales days in the last 28 (shifted) → intermittency |
| `sell_price, price_rel, price_chg_7, price_vs_mean` | price level, vs item-store max, weekly change, vs 56-day mean |
| `promo_proxy` | 1 if price ≥ 5% below its 56-day mean (M5 has no explicit promo flag) |
| `dow, dom, month, weekofyear, is_weekend` | calendar |
| `is_event, event_type_1, days_to_event, days_since_event` | holidays / events (distance capped at 14) |
| `snap` | SNAP-benefit day for the store's state |
| `item_id, dept_id, cat_id, store_id, state_id` | categorical identifiers |

## Outputs (`data/outputs/`) — the contract with the Dash app
| File | Content |
|---|---|
| `pred_baselines.parquet` / `pred_point.parquet` | test-window point forecasts per series-day (`pred_naive, pred_snaive, pred_ma28, pred_prophet, pred_lgbm, pred_reconciled`) |
| `pred_tft.parquet` | TFT quantile forecasts (`tft_q05 … tft_q95`, `pred_tft`) — written by notebook 06 |
| `forecast_quantiles.parquet` | `q05 … q95` quantile forecasts, calibrated `lo80_cal/hi80_cal/lo90_cal/hi90_cal` |
| `orders.parquet` | per week/series: price, forecast, orders under each policy, actual demand |
| `metrics_*.csv`, `cost_comparison.csv`, `cost_by_week.csv` | accuracy, interval coverage, reconciliation, inventory results |
| `data_mode.json` | `real` or `synthetic` |

## Metrics
- **WAPE** = Σ|y − ŷ| / Σ|y|.
- **RMSSE** = √( mean((y − ŷ)²) / mean((yₜ − yₜ₋₁)²_train) ), averaged over series (M5 definition, unweighted).
- **Coverage** = share of actuals inside the interval; compare with the nominal level.

## Optimizer cost assumptions (`src/config.py`; M5 has no cost data)
unit cost = 70% of price · waste/holding per cycle = 25% (FOODS) / 5% (others) of unit cost ·
stockout = lost margin + 0.5 × price goodwill. Change them and re-run notebook 09 or move the app sliders.
