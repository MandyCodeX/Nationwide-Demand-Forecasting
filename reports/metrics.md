# Results summary

> **Data mode: SYNTHETIC.** These numbers validate the pipeline on an M5-shaped stand-in dataset. Re-run all notebooks on the real M5 files and replace this file before presenting.

## Point forecast accuracy (test = last 28 days)
| model   |   n_series |   WAPE |   RMSSE | scope         |
|:--------|-----------:|-------:|--------:|:--------------|
| lgbm    |        600 | 0.6699 |  0.8091 | all series    |
| ma28    |        600 | 0.7037 |  0.8428 | all series    |
| naive   |        600 | 0.8764 |  1.0720 | all series    |
| snaive  |        600 | 0.9299 |  1.1651 | all series    |
| lgbm    |         40 | 0.4834 |  0.8345 | top 40 series |
| ma28    |         40 | 0.5171 |  0.8927 | top 40 series |
| prophet |         40 | 0.5264 |  0.8915 | top 40 series |
| naive   |         40 | 0.6603 |  1.1249 | top 40 series |
| snaive  |         40 | 0.7213 |  1.2373 | top 40 series |

Best model on all series: **lgbm** (WAPE 0.670, RMSSE 0.809).

## Prediction intervals (coverage vs target)
| interval       |   target |   coverage |   avg_width |
|:---------------|---------:|-----------:|------------:|
| 80% raw        |   0.8000 |     0.8564 |      3.6641 |
| 80% calibrated |   0.8000 |     0.8564 |      3.6641 |
| 90% raw        |   0.9000 |     0.9387 |      4.5969 |
| 90% calibrated |   0.9000 |     0.9387 |      4.5969 |

## Hierarchical reconciliation (WAPE by level)
| level      |   MinT reconciled |   base (unreconciled) |   bottom-up |
|:-----------|------------------:|----------------------:|------------:|
| total      |            0.0518 |                0.0745 |      0.0419 |
| state      |            0.0647 |                0.0777 |      0.0546 |
| store      |            0.0865 |                0.0963 |      0.0789 |
| item-store |            0.6801 |                0.6699 |      0.6699 |

## Inventory policies replayed on 4 test weeks
| policy                  |   holding_cost |   stockout_cost |   total_cost |   purchase_spend |   units_short |   units_left |   fill_rate |   cost_vs_B_% |
|:------------------------|---------------:|----------------:|-------------:|-----------------:|--------------:|-------------:|------------:|--------------:|
| A last-week sales       |      1871.9500 |      16102.4700 |   17974.4200 |       73426.7000 |     5678.0000 |    5783.0000 |      0.8100 |       32.3600 |
| B forecast mean         |      1191.2500 |      12388.7500 |   13579.9900 |       71171.3500 |     4400.0000 |    3437.0000 |      0.8500 |        0.0000 |
| C optimised             |      3220.5000 |       2828.4700 |    6048.9800 |      102616.4500 |      939.0000 |   13109.0000 |      0.9700 |      -55.4600 |
| D optimised, B's budget |      1079.7600 |      12825.5000 |   13905.2600 |       71159.8000 |     4558.0000 |    3598.0000 |      0.8500 |        2.4000 |

Optimised policy C vs forecast-mean policy B: total cost 6,049 vs 13,580
(-55%), fill rate 97.0% vs 85.0%, at purchase spend
102,616 vs 71,171.
