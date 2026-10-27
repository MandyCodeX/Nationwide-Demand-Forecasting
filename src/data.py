"""Load M5 (real or synthetic, same file layout) into one tidy long DataFrame for a chosen subset."""
import numpy as np
import pandas as pd
from pathlib import Path
from .config import RAW_DIR, SUBSET

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def raw_files_present(raw_dir=RAW_DIR) -> bool:
    raw_dir = Path(raw_dir)
    return all((raw_dir / f).exists() for f in
               ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"])


def load_m5(raw_dir=RAW_DIR, subset=None, sales_file="sales_train_evaluation.csv") -> pd.DataFrame:
    """Return long frame: id, item/dept/cat/store/state, date, sales, sell_price, calendar flags."""
    subset = {**SUBSET, **(subset or {})}
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / sales_file)
    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw_dir / "sell_prices.csv")

    # --- choose subset: N stores per state, top-N items by units sold in those stores
    stores = (sales[["store_id", "state_id"]].drop_duplicates().sort_values("store_id")
              .groupby("state_id").head(subset["stores_per_state"]))
    stores = stores[stores.state_id.isin(subset["states"])]
    sales = sales[sales.store_id.isin(stores.store_id)]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_cols = day_cols[-subset["last_days"]:]                       # keep most recent days
    tot = sales.groupby("item_id")[day_cols].sum().sum(axis=1).sort_values(ascending=False)
    sales = sales[sales.item_id.isin(tot.index[: subset["n_items"]])]

    # --- wide -> long
    long = sales[ID_COLS + day_cols].melt(id_vars=ID_COLS, var_name="d", value_name="sales")
    long = long.merge(cal, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # price missing => product not on sale yet; drop those pre-launch rows (standard M5 practice)
    long = long.dropna(subset=["sell_price"]).copy()
    long["snap"] = np.select([long.state_id == s for s in ["CA", "TX", "WI"]],
                             [long.snap_CA, long.snap_TX, long.snap_WI], default=0)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    long["sales"] = long["sales"].astype("float32")
    return long.sort_values(["id", "date"]).reset_index(drop=True)


def ensure_raw_data(raw_dir=RAW_DIR):
    """Use real M5 if present, otherwise generate the synthetic stand-in. Returns 'real'/'synthetic'."""
    marker = Path(raw_dir) / ".synthetic"
    if raw_files_present(raw_dir):
        return "synthetic" if marker.exists() else "real"
    from .synthetic_m5 import generate
    generate(raw_dir)
    marker.write_text("synthetic stand-in; delete this folder's csv files and drop in the real M5 files")
    return "synthetic"
