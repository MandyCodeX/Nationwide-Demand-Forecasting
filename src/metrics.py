"""Forecast accuracy metrics: WAPE, RMSSE (M5 style), plus interval coverage."""
import numpy as np
import pandas as pd


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|y - yhat| / sum|y|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def rmsse_one(y_train, y_true, y_pred) -> float:
    """RMSSE for one series: scaled by the in-sample one-step naive MSE (M5 definition).
    Leading zeros (before the product started selling) are dropped from the scale."""
    y_train = np.asarray(y_train, dtype=float)
    nz = np.flatnonzero(y_train)
    if len(nz) == 0:
        return float("nan")
    y_train = y_train[nz[0]:]
    scale = np.mean(np.diff(y_train) ** 2) if len(y_train) > 1 else np.nan
    if not scale or np.isnan(scale):
        return float("nan")
    mse = np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)
    return float(np.sqrt(mse / scale))


def rmsse(train_df, test_df, pred_col, id_col="id", y_col="sales") -> float:
    """Mean RMSSE across series. train_df/test_df are long frames sorted by date."""
    train_df = train_df.sort_values([id_col, "date"])
    test_df = test_df.sort_values([id_col, "date"])
    train_by_id = {k: v[y_col].values for k, v in train_df.groupby(id_col)}
    scores = []
    for k, g in test_df.groupby(id_col):
        if k in train_by_id:
            scores.append(rmsse_one(train_by_id[k], g[y_col].values, g[pred_col].values))
    return float(np.nanmean(scores))


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actuals that fall inside [lower, upper]."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def pinball_loss(y_true, y_pred, q) -> float:
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))
