import numpy as np
import pandas as pd
from src.features import add_features, time_split
from src.config import HORIZON


def _toy(n_days=120):
    dates = pd.date_range("2020-01-01", periods=n_days)
    rows = []
    for i, store in enumerate(["CA_1", "TX_1"]):
        rows.append(pd.DataFrame({
            "id": f"X_{store}", "item_id": "X", "dept_id": "D", "cat_id": "C", "store_id": store,
            "state_id": store[:2], "date": dates, "sales": np.arange(n_days, dtype=float) + 100 * i,
            "sell_price": 2.0, "event_name_1": None, "event_type_1": None, "snap": 0,
            "wm_yr_wk": 1, "weekday": "x", "wday": 1, "month": dates.month, "year": 2020, "d": "d"}))
    return pd.concat(rows, ignore_index=True)


def test_lags_use_only_old_data():
    f = add_features(_toy())
    row = f[(f.id == "X_CA_1")].iloc[60]                    # day index 60, sales == 60
    assert row["lag_28"] == 60 - HORIZON                      # value from 28 days earlier
    assert row["rmean_7"] == np.mean(np.arange(60 - HORIZON - 6, 60 - HORIZON + 1))


def test_no_lag_leakage_inside_horizon():
    f = add_features(_toy())
    assert not any(c.startswith("lag_") and int(c.split("_")[1]) < HORIZON for c in f.columns)


def test_lags_do_not_cross_series():
    f = add_features(_toy())
    first = f[f.id == "X_TX_1"].iloc[0]
    assert np.isnan(first["lag_28"])


def test_time_split_windows():
    f = add_features(_toy())
    tr, va, te = time_split(f)
    assert te.date.nunique() == HORIZON and va.date.nunique() == HORIZON
    assert tr.date.max() < va.date.min() < te.date.min()
