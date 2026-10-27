#!/usr/bin/env bash
# Run the whole pipeline in order, then the tests. Usage: bash run_all.sh
set -e
cd notebooks
for nb in 01_data_understanding 02_eda 03_feature_engineering 04_baselines_naive_prophet 05_lightgbm_forecast \
          07_hierarchical_reconciliation 08_prediction_intervals 09_inventory_optimization_ortools; do
  echo "=== $nb"; jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $nb.ipynb
done
# Notebook 06 (TFT) needs torch/pytorch-forecasting and is slow on CPU -> run it separately (GPU recommended):
#   TFT_SERIES=300 TFT_EPOCHS=20 jupyter nbconvert --to notebook --execute --inplace 06_tft_pytorch_forecasting.ipynb
[ -f ../data/outputs/pred_tft.parquet ] || echo "(skipping TFT: run notebook 06 to include it)"
jupyter nbconvert --to notebook --execute --inplace 10_model_comparison.ipynb
cd .. && python -m pytest tests -q
