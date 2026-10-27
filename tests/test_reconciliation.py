import numpy as np
import pandas as pd
from src.reconcile import build_hierarchy, mint_reconcile, coherence_gap


def _meta():
    return pd.DataFrame({
        "id": ["a_CA_1", "b_CA_1", "a_CA_2", "a_TX_1", "b_TX_1"],
        "store_id": ["CA_1", "CA_1", "CA_2", "TX_1", "TX_1"],
        "state_id": ["CA", "CA", "CA", "TX", "TX"]})


def test_summing_matrix_shape_and_total_row():
    names, S, bottom = build_hierarchy(_meta())
    assert S.shape == (1 + 2 + 3 + 5, 5)
    assert (S[0] == 1).all()


def test_reconciled_forecasts_are_coherent():
    names, S, bottom = build_hierarchy(_meta())
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 20, size=(S.shape[0], 28))   # incoherent base forecasts
    assert coherence_gap(base, S) > 1
    rec = mint_reconcile(base, S)
    assert coherence_gap(rec, S) < 1e-8


def test_already_coherent_forecasts_unchanged():
    names, S, bottom = build_hierarchy(_meta())
    bottom_f = np.random.default_rng(1).uniform(1, 9, size=(5, 28))
    coherent = S @ bottom_f
    assert np.allclose(mint_reconcile(coherent, S), coherent)
