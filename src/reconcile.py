"""Hierarchical reconciliation helpers (numpy). Darts' MinTReconciliator is used in notebook 07;
this module is the dependency-free version used for tests and as a fallback.

Hierarchy: total -> state -> store -> (item, store) bottom series.
"""
import numpy as np
import pandas as pd


def build_hierarchy(bottom_meta: pd.DataFrame):
    """bottom_meta: columns id, store_id, state_id. Returns (names, S) where S is the summing matrix
    (n_all x n_bottom), rows ordered total, states, stores, bottom."""
    meta = bottom_meta.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    bottom = meta["id"].tolist()
    states = sorted(meta["state_id"].unique())
    stores = sorted(meta["store_id"].unique())
    rows, names = [np.ones(len(bottom))], ["total"]
    for s in states:
        rows.append((meta["state_id"] == s).values.astype(float)); names.append(f"state:{s}")
    for s in stores:
        rows.append((meta["store_id"] == s).values.astype(float)); names.append(f"store:{s}")
    for i, b in enumerate(bottom):
        e = np.zeros(len(bottom)); e[i] = 1.0
        rows.append(e); names.append(b)
    return names, np.vstack(rows), bottom


def mint_reconcile(base: np.ndarray, S: np.ndarray, W_diag: np.ndarray = None) -> np.ndarray:
    """Reconcile base forecasts (n_all x H) with weighted least squares (MinT 'wls_struct' when
    W_diag = row-sums of S). Returns coherent forecasts (n_all x H)."""
    if W_diag is None:
        W_diag = S.sum(axis=1)                       # structural scaling
    Winv = 1.0 / np.asarray(W_diag, float)
    A = S.T @ (S * Winv[:, None])                    # S' W^-1 S
    G = np.linalg.solve(A, (S * Winv[:, None]).T)    # (S'W^-1 S)^-1 S' W^-1
    return S @ (G @ base)


def bottom_up(base_bottom: np.ndarray, S: np.ndarray) -> np.ndarray:
    return S @ base_bottom


def coherence_gap(forecasts: np.ndarray, S: np.ndarray) -> float:
    """Max |aggregate - sum of children| across the hierarchy; 0 means perfectly coherent."""
    n_bottom = S.shape[1]
    bottom = forecasts[-n_bottom:]
    return float(np.abs(forecasts - S @ bottom).max())
