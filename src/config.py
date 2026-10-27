"""Shared constants so notebooks, tests and the Dash app agree with each other."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "outputs"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

HORIZON = 28                      # M5 evaluation horizon (days)
CYCLE_DAYS = 7                    # replenishment cycle used by the optimizer
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Which slice of M5 we work on (the full dataset is ~30k series; see README)
SUBSET = dict(states=["CA", "TX", "WI"], stores_per_state=2, n_items=100, last_days=730)

# Cost assumptions for the optimizer (M5 has prices only, so these are explicit assumptions)
COST_RATIO = 0.70                 # unit cost = 70% of selling price
# Holding/waste cost per leftover unit per weekly cycle, as a share of unit cost. Perishable FOODS lose
# much more (spoilage/markdown) than durable categories. These are assumptions - M5 has no cost data.
HOLD_RATE_BY_CAT = {"FOODS": 0.25, "HOUSEHOLD": 0.05, "HOBBIES": 0.05}
HOLD_RATE = 0.10                  # default when the category is unknown
STOCKOUT_GOODWILL = 0.50          # lost-sale penalty = lost margin + 0.5 * price (goodwill)

SEED = 42
