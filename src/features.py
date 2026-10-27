"""Feature engineering for a *direct* multi-horizon global model.

Key idea: every lag / rolling feature is shifted by at least HORIZON (28) days. That means a row on
day T only uses sales known on or before T-28, so one model can forecast the whole 28-day window
without recursive prediction, and there is no target leakage. Prices, calendar and SNAP are known in
advance (as in M5), so they are allowed unshifted.
"""
import numpy as np
import pandas as pd
from .config import HORIZON

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]


def add_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id", sort=False)["sales"]

    # ---- lags (>= horizon) and same-weekday history
    for lag in [horizon, horizon + 7, horizon + 14, horizon + 21, horizon + 28]:
        df[f"lag_{lag}"] = g.shift(lag)
    df["dow_mean_4"] = df[[f"lag_{horizon + 7 * k}" for k in range(4)]].mean(axis=1)

    # ---- rolling stats on a series shifted by `horizon`
    shifted = g.shift(horizon)
    sh_g = shifted.groupby(df["id"], sort=False)
    for w in [7, 14, 28, 56]:
        df[f"rmean_{w}"] = sh_g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    df["rstd_28"] = sh_g.transform(lambda s: s.rolling(28, min_periods=2).std())
    df["rzero_28"] = sh_g.transform(lambda s: s.eq(0).rolling(28, min_periods=1).mean())  # intermittency

    # ---- price features (price is known ahead of time)
    pg = df.groupby("id", sort=False)["sell_price"]
    df["price_max"] = pg.transform("max")
    df["price_rel"] = df["sell_price"] / df["price_max"]
    df["price_chg_7"] = df["sell_price"] / pg.shift(7) - 1
    df["price_mean_56"] = pg.transform(lambda s: s.rolling(56, min_periods=1).mean())
    df["price_vs_mean"] = df["sell_price"] / df["price_mean_56"]
    # M5 has no explicit promo flag -> proxy: price at least 5% below its own recent average
    df["promo_proxy"] = (df["price_vs_mean"] < 0.95).astype("int8")

    # ---- calendar / holidays
    d = df["date"].dt
    df["dow"], df["dom"], df["month"] = d.dayofweek, d.day, d.month
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_event"] = df["event_name_1"].notna().astype("int8")
    df = _add_event_distance(df)

    # ---- categoricals as pandas category (LightGBM handles natively)
    for c in CAT_COLS:
        df[c] = df[c].fillna("none").astype("category")
    return df


def _add_event_distance(df: pd.DataFrame, cap: int = 14) -> pd.DataFrame:
    days = pd.DataFrame({"date": np.sort(df["date"].unique())})
    ev_dates = df.loc[df["event_name_1"].notna(), "date"].drop_duplicates().sort_values()
    if len(ev_dates) == 0:
        days["days_to_event"], days["days_since_event"] = cap, cap
    else:
        ev = pd.DataFrame({"ev_date": ev_dates.values})
        nxt = pd.merge_asof(days, ev.rename(columns={"ev_date": "nxt"}), left_on="date",
                            right_on="nxt", direction="forward")
        prv = pd.merge_asof(days, ev.rename(columns={"ev_date": "prv"}), left_on="date",
                            right_on="prv", direction="backward")
        days["days_to_event"] = (nxt["nxt"] - days["date"]).dt.days.fillna(cap).clip(upper=cap)
        days["days_since_event"] = (days["date"] - prv["prv"]).dt.days.fillna(cap).clip(upper=cap)
    return df.merge(days, on="date", how="left")


def feature_columns(df: pd.DataFrame) -> list:
    drop = {"id", "date", "d", "sales", "wm_yr_wk", "weekday", "event_name_1", "event_name_2",
            "event_type_2", "year", "price_max", "price_mean_56"}
    return [c for c in df.columns if c not in drop]


def time_split(df: pd.DataFrame, horizon: int = HORIZON):
    """train | valid (last `horizon` days before test) | test (last `horizon` days)."""
    last = df["date"].max()
    test_start = last - pd.Timedelta(days=horizon - 1)
    valid_start = test_start - pd.Timedelta(days=horizon)
    train = df[df["date"] < valid_start]
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, valid, test
