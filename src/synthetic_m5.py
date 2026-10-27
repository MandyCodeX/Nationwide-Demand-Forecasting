"""Generate a small M5-shaped dataset (calendar.csv, sell_prices.csv, sales_train_evaluation.csv).

WHY THIS EXISTS: it lets the whole pipeline (loader -> features -> models -> optimizer -> app)
run and be tested without downloading the 450 MB Kaggle files. Results on this data only prove the
pipeline works; report real accuracy numbers from the true M5 data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

STATES = {"CA": ["CA_1", "CA_2", "CA_3"], "TX": ["TX_1", "TX_2", "TX_3"], "WI": ["WI_1", "WI_2", "WI_3"]}
DEPTS = {"FOODS": ["FOODS_1", "FOODS_2"], "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
         "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"]}
EVENTS = [  # (month, day, name, type)
    (1, 1, "NewYear", "National"), (2, 9, "SuperBowl", "Sporting"), (5, 26, "MemorialDay", "National"),
    (7, 4, "IndependenceDay", "National"), (9, 1, "LaborDay", "National"),
    (10, 31, "Halloween", "Cultural"), (11, 27, "Thanksgiving", "National"),
    (12, 25, "Christmas", "National"),
]


def generate(raw_dir, n_days=800, items_per_dept=20, start="2014-01-29", seed=42):
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start, periods=n_days, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["wm_yr_wk"] = 11000 + (np.arange(n_days) + dates[0].dayofweek) // 7
    cal["weekday"] = dates.day_name()
    cal["wday"] = (dates.dayofweek + 2) % 7 + 1
    cal["month"], cal["year"] = dates.month, dates.year
    cal["d"] = [f"d_{i+1}" for i in range(n_days)]
    cal["event_name_1"] = pd.Series([None] * n_days, dtype=object)
    cal["event_type_1"] = pd.Series([None] * n_days, dtype=object)
    for m, dd, name, typ in EVENTS:
        mask = (dates.month == m) & (dates.day == dd)
        cal.loc[mask, "event_name_1"] = name
        cal.loc[mask, "event_type_1"] = typ
    cal["event_name_2"] = None
    cal["event_type_2"] = None
    for s in STATES:
        cal[f"snap_{s}"] = ((dates.day <= 10) & 1).astype(int)  # SNAP first 10 days

    # items
    items = []
    for cat, depts in DEPTS.items():
        for dept in depts:
            for k in range(1, items_per_dept + 1):
                items.append((f"{dept}_{k:03d}", dept, cat))
    items = pd.DataFrame(items, columns=["item_id", "dept_id", "cat_id"])
    items["base_price"] = np.round(rng.lognormal(1.2, 0.6, len(items)), 2).clip(0.5, 30)
    items["popularity"] = rng.lognormal(0.2, 0.8, len(items))          # item popularity
    items["launch"] = np.where(rng.random(len(items)) < 0.1, rng.integers(0, 200, len(items)), 0)

    stores = [(s, st) for st, ss in STATES.items() for s in ss]
    store_scale = {s: rng.lognormal(0, 0.3) for s, _ in stores}

    price_rows, sales_rows = [], []
    t = np.arange(n_days)
    weeks = cal["wm_yr_wk"].values
    uniq_weeks = np.unique(weeks)
    dow = dates.dayofweek.values
    for _, it in items.iterrows():
        # weekly price path with occasional promotions (price cuts)
        for store, state in stores:
            p = np.full(len(uniq_weeks), it.base_price, dtype=float)
            promo_w = rng.random(len(uniq_weeks)) < 0.08
            p[promo_w] *= rng.uniform(0.7, 0.9, promo_w.sum())
            p = np.round(p * (1 + 0.0004 * np.arange(len(uniq_weeks))), 2)   # mild inflation
            price_rows.append(pd.DataFrame({"store_id": store, "item_id": it.item_id,
                                            "wm_yr_wk": uniq_weeks, "sell_price": p}))
            price_d = p[np.searchsorted(uniq_weeks, weeks)]
            rel = price_d / it.base_price
            weekly = np.array([0.9, 0.85, 0.9, 0.95, 1.1, 1.35, 1.25])[dow]
            yearly = 1 + 0.25 * np.sin(2 * np.pi * (t + 30) / 365.25)
            trend = 1 + 0.0003 * t
            snap = 1 + (0.25 if it.cat_id == "FOODS" else 0.05) * cal[f"snap_{state}"].values
            ev = 1 + 0.35 * cal["event_name_1"].notna().values
            lam = (0.5 * it.popularity * store_scale[store] * weekly * yearly * trend * snap * ev
                   * rel ** (-2.0))
            lam = np.where(t >= it.launch, lam, 0)
            # negative binomial -> over-dispersed, intermittent counts
            r = 4.0
            y = rng.negative_binomial(r, r / (r + lam + 1e-9))
            sales_rows.append({"id": f"{it.item_id}_{store}_evaluation", "item_id": it.item_id,
                               "dept_id": it.dept_id, "cat_id": it.cat_id, "store_id": store,
                               "state_id": state, **{f"d_{i+1}": int(v) for i, v in enumerate(y)}})
    sales = pd.DataFrame(sales_rows)
    prices = pd.concat(price_rows, ignore_index=True)
    cal.to_csv(raw_dir / "calendar.csv", index=False)
    prices.to_csv(raw_dir / "sell_prices.csv", index=False)
    sales.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    return sales.shape, prices.shape, cal.shape


if __name__ == "__main__":
    from src.config import RAW_DIR
    print(generate(RAW_DIR))
