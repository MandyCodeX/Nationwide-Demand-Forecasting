import numpy as np
import pandas as pd
from src.metrics import wape, rmsse_one, interval_coverage, pinball_loss


def test_wape_perfect_and_known():
    assert wape([1, 2, 3], [1, 2, 3]) == 0
    assert abs(wape([10, 10], [8, 12]) - 0.2) < 1e-9


def test_rmsse_scaled_by_naive():
    train = [0, 0, 1, 2, 3, 4]               # leading zeros dropped; diffs all 1 -> scale 1
    assert abs(rmsse_one(train, [5, 6], [5, 6])) < 1e-9
    assert abs(rmsse_one(train, [5, 6], [6, 7]) - 1.0) < 1e-9


def test_interval_coverage_and_pinball():
    assert interval_coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5
    assert pinball_loss([10], [8], 0.5) == 1.0
