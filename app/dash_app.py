"""Plotly Dash app: forecasts + intervals, accuracy, and an inventory order recommender.

Run:  python app/dash_app.py   ->  http://127.0.0.1:8050
Reads the files written by the notebooks (data/outputs/*, data/processed/long.parquet).
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CYCLE_DAYS, OUT_DIR, PROC_DIR, QUANTILES  # noqa: E402
from src.optimizer import (cost_params, optimize_orders, realised_cost, round_orders,  # noqa: E402
                           sample_demand)

QCOLS = [f"q{int(q * 100):02d}" for q in QUANTILES]
MODEL_COLS = {"LightGBM": "pred_lgbm", "LightGBM (reconciled)": "pred_reconciled",
              "Seasonal naive": "pred_snaive", "Prophet": "pred_prophet", "TFT": "pred_tft"}


# ----------------------------------------------------------------------------- data
def load_data():
    hist = pd.read_parquet(PROC_DIR / "long.parquet",
                           columns=["id", "date", "sales", "store_id", "state_id", "item_id", "cat_id", "sell_price"])
    pt = pd.read_parquet(OUT_DIR / "pred_point.parquet")
    tft_path = OUT_DIR / "pred_tft.parquet"
    if tft_path.exists():
        pt = pt.merge(pd.read_parquet(tft_path)[["id", "date", "pred_tft"]], on=["id", "date"], how="left")
    qf = pd.read_parquet(OUT_DIR / "forecast_quantiles.parquet")
    meta = hist.drop_duplicates("id").set_index("id")[["store_id", "state_id", "item_id", "cat_id"]]
    mode = json.load(open(OUT_DIR / "data_mode.json"))["data_mode"]
    rd = lambda n: pd.read_csv(OUT_DIR / n) if (OUT_DIR / n).exists() else pd.DataFrame()
    return dict(hist=hist, pt=pt, qf=qf, meta=meta, mode=mode, acc=rd("metrics_baselines.csv"),
                lgbm=rd("metrics_lgbm.csv"), tft=rd("metrics_tft.csv"), iv=rd("metrics_intervals.csv"),
                rec=rd("metrics_reconciliation.csv"), cost=rd("cost_comparison.csv"))


D = load_data()
IDS = sorted(D["qf"].id.unique())
DATES = sorted(D["qf"].date.unique())
WEEKS = [DATES[i:i + CYCLE_DAYS] for i in range(0, len(DATES), CYCLE_DAYS)]
TEST_START = DATES[0]
CAT = D["meta"].loc[IDS, "cat_id"].astype(str).values
AVAILABLE_MODELS = {k: v for k, v in MODEL_COLS.items() if v in D["pt"].columns and D["pt"][v].notna().any()}


def node_ids(level, node):
    m = D["meta"]
    if level == "Total":
        return list(m.index)
    if level == "State":
        return list(m.index[m.state_id == node])
    if level == "Store":
        return list(m.index[m.store_id == node])
    return [node]


def node_options(level):
    m = D["meta"]
    if level == "Total":
        return ["All stores"]
    if level == "State":
        return sorted(m.state_id.unique())
    if level == "Store":
        return sorted(m.store_id.unique())
    return IDS


# ----------------------------------------------------------------------------- forecast figure
def forecast_figure(level, node, models, history_days=90):
    ids = node_ids(level, node)
    h = D["hist"][D["hist"].id.isin(ids)].groupby("date")["sales"].sum()
    h = h[h.index < TEST_START].tail(history_days)
    p = D["pt"][D["pt"].id.isin(ids)].groupby("date").sum(numeric_only=True)
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h.values, name="History", line=dict(color="#888"))
    fig.add_scatter(x=p.index, y=p["sales"], name="Actual (test)", line=dict(color="black"), mode="lines+markers")
    if level == "Item-store":
        q = D["qf"][D["qf"].id == node].sort_values("date")
        fig.add_scatter(x=q.date, y=q.hi90_cal, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q.date, y=q.lo90_cal, fill="tonexty", fillcolor="rgba(31,119,180,.15)",
                        line=dict(width=0), name="90% interval")
        fig.add_scatter(x=q.date, y=q.hi80_cal, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q.date, y=q.lo80_cal, fill="tonexty", fillcolor="rgba(31,119,180,.3)",
                        line=dict(width=0), name="80% interval")
    for m in models:
        col = AVAILABLE_MODELS.get(m)
        if col and col in p.columns:
            fig.add_scatter(x=p.index, y=p[col], name=m, mode="lines")
    title = f"{level}: {node}" + ("" if level == "Item-store" else "  (intervals shown for item-store level)")
    fig.update_layout(title=title, template="plotly_white", height=460, yaxis_title="units / day",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
    return fig


# ----------------------------------------------------------------------------- order recommender
@lru_cache(maxsize=256)
def _week_arrays(store, week_idx):
    wk = WEEKS[week_idx]
    ids = [i for i in IDS if store == "All stores" or D["meta"].loc[i, "store_id"] == store]
    wq = D["qf"][D["qf"].date.isin(wk) & D["qf"].id.isin(ids)].sort_values(["id", "date"])
    Q = wq[QCOLS].values.reshape(len(ids), len(wk), len(QCOLS))
    actual = wq.pivot(index="id", columns="date", values="sales").loc[ids].sum(axis=1).values
    price = (D["hist"][D["hist"].date.isin(wk) & D["hist"].id.isin(ids)].groupby("id")["sell_price"].mean()
             .loc[ids].values)
    fmean = D["pt"][D["pt"].date.isin(wk) & D["pt"].id.isin(ids)].groupby("id")["pred_lgbm"].sum().loc[ids].values
    cat = D["meta"].loc[ids, "cat_id"].astype(str).values
    return ids, Q, actual, price, fmean, cat


def recommend(store, week_idx, budget_pct, stockout_mult, hold_mult=1.0, n_scen=40):
    ids, Q, actual, price, fmean, cat = _week_arrays(store, week_idx)
    scen = sample_demand(Q, n_scen=n_scen, seed=week_idx)
    unit_cost = cost_params(price, cat)[0]
    q_b = np.round(fmean)
    budget = budget_pct / 100 * float((unit_cost * q_b).sum())
    q, status = optimize_orders(scen, price, budget=budget, cat=cat, stockout_mult=stockout_mult, hold_mult=hold_mult)
    q = round_orders(q, price, budget=budget)
    kw = dict(cat=cat, stockout_mult=stockout_mult, hold_mult=hold_mult)
    return dict(ids=ids, q=q, q_b=q_b, fmean=fmean, actual=actual, price=price, status=status, budget=budget,
                opt=realised_cost(q, actual, price, **kw), base=realised_cost(q_b, actual, price, **kw))


def kpi(title, value, sub=""):
    return html.Div([html.Div(title, style={"fontSize": 12, "color": "#666"}),
                     html.Div(value, style={"fontSize": 24, "fontWeight": 600}),
                     html.Div(sub, style={"fontSize": 12, "color": "#888"})],
                    style={"border": "1px solid #ddd", "borderRadius": 8, "padding": "10px 16px", "minWidth": 170})


# ----------------------------------------------------------------------------- layout
def accuracy_layout():
    acc = pd.concat([x for x in [D["acc"], D["lgbm"], D["tft"]] if len(x)]).drop_duplicates(["model", "scope"])
    acc = acc.sort_values(["scope", "WAPE"])
    figs = []
    for scope in acc.scope.unique():
        s = acc[acc.scope == scope]
        f = go.Figure([go.Bar(x=s.model, y=s.WAPE, name="WAPE"), go.Bar(x=s.model, y=s.RMSSE, name="RMSSE")])
        f.update_layout(title=f"Accuracy - {scope} (lower is better)", barmode="group", template="plotly_white", height=330)
        figs.append(dcc.Graph(figure=f))
    tbl = lambda df: dash_table.DataTable(df.round(4).to_dict("records"), [{"name": c, "id": c} for c in df.columns],
                                          style_cell={"padding": "4px 10px"}, style_header={"fontWeight": 600})
    rec = D["rec"]
    return html.Div([html.H4("Point forecast accuracy (28-day test window)"), tbl(acc), *figs,
                     html.H4("Prediction interval coverage (actual share inside the interval)"), tbl(D["iv"]),
                     html.H4("WAPE by hierarchy level: base vs reconciled"), tbl(rec) if len(rec) else "run notebook 07"])


def build_layout():
    banner = (html.Div("SYNTHETIC DATA MODE - pipeline demo on an M5-shaped stand-in. Replace with real M5 files for real results.",
                       style={"background": "#fff3cd", "padding": 8, "borderRadius": 6, "margin": "8px 0"})
              if D["mode"] == "synthetic" else html.Div())
    return html.Div([
        html.H2("Demand Forecasting & Inventory Optimization"), banner,
        dcc.Tabs([
            dcc.Tab(label="Forecasts", children=[html.Div([
                html.Div([dcc.Dropdown(["Total", "State", "Store", "Item-store"], "Store", id="level", clearable=False,
                                       style={"width": 160}),
                          dcc.Dropdown(id="node", clearable=False, style={"width": 320}),
                          dcc.Checklist(list(AVAILABLE_MODELS), ["LightGBM"], id="models", inline=True,
                                        inputStyle={"marginLeft": 12})],
                         style={"display": "flex", "gap": 12, "alignItems": "center", "margin": "12px 0"}),
                dcc.Graph(id="forecast_fig")])]),
            dcc.Tab(label="Accuracy", children=[accuracy_layout()]),
            dcc.Tab(label="Order recommendations", children=[html.Div([
                html.Div([html.Div([html.Label("Store"), dcc.Dropdown(["All stores"] + sorted(D["meta"].store_id.unique()),
                                    sorted(D["meta"].store_id.unique())[0], id="store", clearable=False)], style={"width": 200}),
                          html.Div([html.Label("Week of the test window"), dcc.Slider(1, len(WEEKS), 1, value=1, id="week",
                                    marks={i: f"W{i}" for i in range(1, len(WEEKS) + 1)})], style={"width": 260}),
                          html.Div([html.Label("Budget (% of the forecast-mean policy's spend)"),
                                    dcc.Slider(60, 140, 10, value=100, id="budget", marks={v: f"{v}%" for v in range(60, 150, 20)})],
                                   style={"width": 340}),
                          html.Div([html.Label("Stockout cost multiplier"),
                                    dcc.Slider(0.25, 3, None, value=1, id="so", marks={0.25: "0.25x", 1: "1x", 2: "2x", 3: "3x"})],
                                   style={"width": 300})],
                         style={"display": "flex", "gap": 28, "margin": "14px 0", "flexWrap": "wrap"}),
                html.Div(id="kpis", style={"display": "flex", "gap": 12, "flexWrap": "wrap", "margin": "10px 0"}),
                dcc.Graph(id="roi_fig"),
                html.H4("Order recommendations (largest orders first)"),
                dash_table.DataTable(id="orders_tbl", page_size=15, sort_action="native",
                                     style_cell={"padding": "4px 10px"}, style_header={"fontWeight": 600}),
                html.Button("Download recommendations (CSV)", id="dl_btn", style={"margin": "10px 0"}),
                dcc.Download(id="dl")])]),
        ])], style={"maxWidth": 1200, "margin": "0 auto", "fontFamily": "sans-serif", "padding": 12})


app = Dash(__name__, title="Demand Forecasting & Inventory Optimization")
app.layout = build_layout


# ----------------------------------------------------------------------------- callbacks
@app.callback(Output("node", "options"), Output("node", "value"), Input("level", "value"))
def _node(level):
    opts = node_options(level)
    return opts, opts[0]


@app.callback(Output("forecast_fig", "figure"), Input("level", "value"), Input("node", "value"), Input("models", "value"))
def _fc(level, node, models):
    if node not in node_options(level):
        node = node_options(level)[0]
    return forecast_figure(level, node, models or [])


@app.callback(Output("kpis", "children"), Output("roi_fig", "figure"), Output("orders_tbl", "data"),
              Output("orders_tbl", "columns"), Input("store", "value"), Input("week", "value"),
              Input("budget", "value"), Input("so", "value"))
def _orders(store, week, budget, so):
    r = recommend(store, week - 1, budget, so)
    o, b = r["opt"], r["base"]
    cards = [kpi("Recommended spend", f"{o['purchase_spend']:,.0f}", f"budget cap {r['budget']:,.0f} | {r['status']}"),
             kpi("Fill rate (replayed)", f"{o['fill_rate']:.1%}", f"forecast-mean policy: {b['fill_rate']:.1%}"),
             kpi("Holding + stockout cost", f"{o['total_cost']:,.0f}", f"forecast-mean policy: {b['total_cost']:,.0f}"),
             kpi("Cost change vs forecast-mean", f"{(o['total_cost'] / max(b['total_cost'], 1e-9) - 1) * 100:+.1f}%",
                 "negative = optimiser is cheaper (replayed on real demand)")]
    f = go.Figure()
    for name, x, color in [("Forecast-mean policy", b, "#999"), ("Optimised", o, "#1f77b4")]:
        f.add_bar(x=["Holding cost", "Stockout cost"], y=[x["holding_cost"], x["stockout_cost"]], name=name, marker_color=color)
    f.update_layout(barmode="group", template="plotly_white", height=320, title="Expected savings vs cost (replayed on actual demand)")
    df = pd.DataFrame({"item_store": r["ids"], "category": D["meta"].loc[r["ids"], "cat_id"].astype(str).values,
                       "price": r["price"].round(2), "forecast_units": r["fmean"].round(1),
                       "recommended_order": r["q"].astype(int), "forecast_mean_order": r["q_b"].astype(int),
                       "actual_demand": r["actual"].astype(int)}).sort_values("recommended_order", ascending=False)
    return cards, f, df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


@app.callback(Output("dl", "data"), Input("dl_btn", "n_clicks"), Input("store", "value"), Input("week", "value"),
              Input("budget", "value"), Input("so", "value"), prevent_initial_call=True)
def _download(n, store, week, budget, so):
    from dash import ctx
    if ctx.triggered_id != "dl_btn":
        return None
    r = recommend(store, week - 1, budget, so)
    df = pd.DataFrame({"item_store": r["ids"], "recommended_order": r["q"].astype(int), "forecast_units": r["fmean"].round(2)})
    return dcc.send_data_frame(df.to_csv, f"orders_{store}_week{week}.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
