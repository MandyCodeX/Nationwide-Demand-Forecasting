"""Inventory optimisation with OR-Tools (GLOP linear program, scenario-based newsvendor).

For every item-store we choose an order quantity q for the next replenishment cycle. Demand
scenarios are sampled from the model's quantile forecasts. We minimise the average
holding cost (leftover units) + stockout cost (lost sales) subject to:
  * a total purchasing budget, and
  * a total shelf/storage capacity,
and q >= 0 (optionally a minimum order). Compare against a 'forecast mean' policy to see the savings.
"""
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from .config import COST_RATIO, HOLD_RATE, HOLD_RATE_BY_CAT, STOCKOUT_GOODWILL, QUANTILES


def sample_demand(q_forecast: np.ndarray, probs=QUANTILES, n_scen: int = 50, seed: int = 0):
    """q_forecast: (n_series, n_days, n_quantiles) -> scenario demand (n_series, n_scen) summed
    over the cycle. Days are sampled independently from a piecewise-linear inverse CDF."""
    rng = np.random.default_rng(seed)
    n, days, k = q_forecast.shape
    q_forecast = np.sort(q_forecast, axis=2)                    # fix quantile crossing
    p = np.concatenate([[0.0], probs, [1.0]])
    lo = np.maximum(0, q_forecast[:, :, :1] - (q_forecast[:, :, 1:2] - q_forecast[:, :, :1]))
    hi = q_forecast[:, :, -1:] + (q_forecast[:, :, -1:] - q_forecast[:, :, -2:-1])
    vals = np.concatenate([lo, q_forecast, hi], axis=2)         # (n, days, k+2)
    u = rng.random((n_scen, days))
    out = np.zeros((n, n_scen))
    for s in range(n_scen):
        for d in range(days):
            # vectorised interpolation across series for scenario s / day d
            idx = np.searchsorted(p, u[s, d]) - 1
            idx = min(max(idx, 0), len(p) - 2)
            w = (u[s, d] - p[idx]) / (p[idx + 1] - p[idx])
            out[:, s] += vals[:, d, idx] * (1 - w) + vals[:, d, idx + 1] * w
    return out


def cost_params(price: np.ndarray, cat=None, stockout_mult: float = 1.0, hold_mult: float = 1.0):
    """Per-unit purchase cost, holding (waste) cost and stockout cost. `cat` = category per series."""
    unit_cost = COST_RATIO * price
    rate = (np.array([HOLD_RATE_BY_CAT.get(c, HOLD_RATE) for c in cat]) if cat is not None
            else np.full(len(price), HOLD_RATE))
    hold = hold_mult * rate * unit_cost
    stockout = stockout_mult * ((price - unit_cost) + STOCKOUT_GOODWILL * price)   # lost margin + goodwill
    return unit_cost, hold, stockout


def optimize_orders(scen: np.ndarray, price: np.ndarray, on_hand: np.ndarray = None,
                    budget: float = None, capacity: float = None, volume: np.ndarray = None,
                    min_order: float = 0.0, time_limit_s: int = 60, cat=None,
                    stockout_mult: float = 1.0, hold_mult: float = 1.0):
    """scen: (n_series, n_scen) demand scenarios. Returns (orders float array, solver status str)."""
    n, S = scen.shape
    on_hand = np.zeros(n) if on_hand is None else on_hand
    unit_cost, hold, stockout = cost_params(price, cat, stockout_mult, hold_mult)
    volume = np.ones(n) if volume is None else volume
    solver = pywraplp.Solver.CreateSolver("GLOP")
    solver.SetTimeLimit(time_limit_s * 1000)
    q = [solver.NumVar(min_order, solver.infinity(), f"q{i}") for i in range(n)]
    obj = solver.Objective()
    for i in range(n):
        for s in range(S):
            over = solver.NumVar(0, solver.infinity(), "")
            under = solver.NumVar(0, solver.infinity(), "")
            rhs = float(scen[i, s] - on_hand[i])
            c = solver.Constraint(rhs, rhs)          # q + oh - D = over - under
            c.SetCoefficient(q[i], 1); c.SetCoefficient(over, -1); c.SetCoefficient(under, 1)
            obj.SetCoefficient(over, hold[i] / S)
            obj.SetCoefficient(under, stockout[i] / S)
    if budget is not None:
        c = solver.Constraint(0, float(budget))
        for i in range(n):
            c.SetCoefficient(q[i], float(unit_cost[i]))
    if capacity is not None:
        c = solver.Constraint(0, float(capacity))
        for i in range(n):
            c.SetCoefficient(q[i], float(volume[i]))
    obj.SetMinimization()
    status = solver.Solve()
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
             pywraplp.Solver.INFEASIBLE: "INFEASIBLE"}
    orders = np.array([v.solution_value() for v in q])
    return orders, names.get(status, str(status))


def realised_cost(orders, demand, price, on_hand=None, cat=None, stockout_mult: float = 1.0, hold_mult: float = 1.0):
    """Cost and service level when real demand arrives. Returns dict."""
    on_hand = np.zeros(len(orders)) if on_hand is None else on_hand
    unit_cost, hold, stockout = cost_params(price, cat, stockout_mult, hold_mult)
    avail = orders + on_hand
    over = np.maximum(avail - demand, 0)
    under = np.maximum(demand - avail, 0)
    holding = float((hold * over).sum())
    lost = float((stockout * under).sum())
    return dict(holding_cost=holding, stockout_cost=lost, total_cost=holding + lost,
                purchase_spend=float((unit_cost * orders).sum()),
                fill_rate=float(1 - under.sum() / max(demand.sum(), 1e-9)),
                units_ordered=float(orders.sum()), units_short=float(under.sum()),
                units_left=float(over.sum()))


def round_orders(q: np.ndarray, price: np.ndarray, budget: float = None) -> np.ndarray:
    """Round LP orders to whole units without breaking the budget: round to nearest, then, if the
    budget is exceeded, take back one unit from the items that were rounded up the most."""
    r = np.round(np.maximum(q, 0))
    if budget is None:
        return r
    unit_cost = COST_RATIO * price
    for idx in np.argsort(-(r - q)):
        if (unit_cost * r).sum() <= budget + 1e-9:
            break
        if r[idx] > q[idx] and r[idx] > 0:
            r[idx] -= 1
    while (unit_cost * r).sum() > budget + 1e-9:        # safety net (practically never reached)
        r[np.argmax(unit_cost * r)] -= 1
    return r
