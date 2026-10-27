import numpy as np
from src.optimizer import optimize_orders, realised_cost, cost_params, sample_demand
from src.config import QUANTILES


def _scen(n=20, S=40, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(3, 30, n)
    return rng.poisson(mean[:, None], size=(n, S)).astype(float), rng.uniform(1, 10, n)


def test_orders_nonnegative_and_near_demand():
    scen, price = _scen()
    q, status = optimize_orders(scen, price)
    assert status == "OPTIMAL" and (q >= -1e-9).all()
    total_mean = scen.mean(axis=1).sum()
    # critical ratio ~0.98 -> order near the 98th percentile: above the mean, but not absurdly so
    assert total_mean <= q.sum() <= 2 * total_mean


def test_budget_constraint_respected():
    scen, price = _scen()
    unit_cost, _, _ = cost_params(price)
    q_free, _ = optimize_orders(scen, price)
    budget = 0.6 * (unit_cost * q_free).sum()
    q, status = optimize_orders(scen, price, budget=budget)
    assert status == "OPTIMAL"
    assert (unit_cost * q).sum() <= budget + 1e-6


def test_capacity_constraint_respected():
    scen, price = _scen()
    q, _ = optimize_orders(scen, price, capacity=100.0)
    assert q.sum() <= 100.0 + 1e-6


def test_higher_stockout_item_gets_more_stock_than_mean_when_costly():
    scen = np.tile(np.arange(1, 41, dtype=float), (1, 1))       # one item, demand 1..40
    q, _ = optimize_orders(scen, np.array([10.0]))
    assert q[0] > scen.mean()          # stockout cost >> holding cost -> order above the mean


def test_realised_cost_fill_rate():
    r = realised_cost(np.array([5.0, 5.0]), np.array([10.0, 2.0]), np.array([2.0, 2.0]))
    assert abs(r["fill_rate"] - (1 - 5 / 12)) < 1e-9 and r["units_left"] == 3


def test_sample_demand_shape_and_monotone_quantiles():
    q = np.sort(np.random.default_rng(0).uniform(0, 10, (4, 7, len(QUANTILES))), axis=2)
    d = sample_demand(q, n_scen=20)
    assert d.shape == (4, 20) and (d >= 0).all()


def test_round_orders_respects_budget_and_is_integer():
    from src.optimizer import round_orders
    rng = np.random.default_rng(3)
    q = rng.uniform(0, 10, 50); price = rng.uniform(1, 9, 50)
    budget = 0.7 * (0.7 * price * q).sum()
    r = round_orders(q, price, budget=budget)
    assert np.allclose(r, np.round(r)) and (0.7 * price * r).sum() <= budget + 1e-6


def test_category_specific_holding_cost():
    _, hold, _ = cost_params(np.array([10.0, 10.0]), cat=["FOODS", "HOBBIES"])
    assert hold[0] > hold[1]
