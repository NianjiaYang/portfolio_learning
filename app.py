"""
Tactical Asset Allocation Explorer — The Time Machine

Interactive tool to explore risk-adjusted returns for the Citi learning brief.
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    allocation_bounds,
    benchmark_weights,
    compare_metrics_table,
    compute_returns,
    load_prices,
    optimize_sharpe,
    portfolio_metrics,
    portfolio_value,
    slice_period,
    weights_to_frame,
)
from src.config import ASSETS, DEFAULT_START_DATE, PRICES_FILE, RISK_FREE_RATE  # noqa: E402

st.set_page_config(
    page_title="TAA Time Machine",
    page_icon="📈",
    layout="wide",
)

st.title("The Time Machine — Tactical Asset Allocation")
st.caption(
    "Start date: 20 Aug 2019 · Adjust weights within benchmark limits · Compare Sharpe vs benchmark"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        st.error("Price data not found. Run: python -m src.download_data")
        st.stop()
    return load_prices(PRICES_FILE)


prices_all = load_data()
min_date = prices_all.index.min().date()
max_date = prices_all.index.max().date()

with st.sidebar:
    st.header("Time Frame")
    start_date = st.date_input(
        "Start date",
        value=pd.Timestamp(DEFAULT_START_DATE).date(),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.date_input(
        "End date (evaluation checkpoint)",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.header("Risk-Free Rate")
    rf_rate = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=11,
        value=RISK_FREE_RATE * 100,
        step=0.1,
        help="Used for Sharpe ratio calculation (excess return / volatility)",
    ) / 100

    st.header("Allocation")
    st.markdown("Set weights within each asset's benchmark ± limit. Weights auto-normalize to 100%.")

    low, high = allocation_bounds()
    bench = benchmark_weights()

    custom_weights = []
    for i, asset in enumerate(ASSETS):
        default_pct = round(bench[i] * 100, 1)
        min_pct = round(low[i] * 100, 1)
        max_pct = round(high[i] * 100, 1)
        val = st.slider(
            asset["name"],
            min_value=float(min_pct),
            max_value=float(max_pct),
            value=float(default_pct),
            step=0.5,
            key=f"weight_{asset['key']}",
        )
        custom_weights.append(val / 100)

    w_custom = np.array(custom_weights)
    w_custom = w_custom / w_custom.sum()

    st.divider()
    preset = st.radio(
        "Load preset",
        ["Custom", "Benchmark", "Optimized (Max Sharpe)"],
        index=0,
    )

    if preset == "Benchmark":
        w_custom = bench.copy()
    elif preset == "Optimized (Max Sharpe)":
        pass  # computed below after returns are ready

    if st.button("Refresh optimized weights"):
        st.session_state["force_optimize"] = True

prices = slice_period(prices_all, start_date, end_date)
returns = compute_returns(prices)

if len(returns) < 20:
    st.warning("Not enough trading days in selected range. Widen the date range.")
    st.stop()

w_bench = benchmark_weights()
w_opt = optimize_sharpe(returns, risk_free_rate=rf_rate)

if preset == "Optimized (Max Sharpe)":
    w_custom = w_opt.copy()

metrics_custom = portfolio_metrics(returns, w_custom, rf_rate)
metrics_bench = portfolio_metrics(returns, w_bench, rf_rate)
metrics_opt = portfolio_metrics(returns, w_opt, rf_rate)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Your Sharpe", f"{metrics_custom.sharpe_ratio:.3f}")
col2.metric("Benchmark Sharpe", f"{metrics_bench.sharpe_ratio:.3f}")
col3.metric(
    "vs Benchmark",
    f"{metrics_custom.sharpe_ratio - metrics_bench.sharpe_ratio:+.3f}",
    delta_color="normal",
)
col4.metric("Optimized Sharpe", f"{metrics_opt.sharpe_ratio:.3f}")

st.subheader("Performance Comparison")
comparison = compare_metrics_table(
    returns, w_custom, w_bench, w_opt, risk_free_rate=rf_rate
)
st.dataframe(comparison, use_container_width=True, hide_index=True)

tab1, tab2, tab3 = st.tabs(["Allocation", "Charts", "Recommended Strategy"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.markdown("**Your allocation**")
        st.dataframe(weights_to_frame(w_custom), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Optimized allocation (max Sharpe, same constraints)**")
        st.dataframe(weights_to_frame(w_opt), use_container_width=True, hide_index=True)

with tab2:
    wealth_custom = portfolio_value(prices, w_custom)
    wealth_bench = portfolio_value(prices, w_bench)
    wealth_opt = portfolio_value(prices, w_opt)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Portfolio Value (rebased to 1.0)", "Rolling 63-day Sharpe"),
        row_heights=[0.6, 0.4],
    )

    for name, wealth, color in [
        ("Your Allocation", wealth_custom, "#2563eb"),
        ("Benchmark", wealth_bench, "#64748b"),
        ("Optimized", wealth_opt, "#16a34a"),
    ]:
        fig.add_trace(
            go.Scatter(x=wealth.index, y=wealth, name=name, line=dict(color=color, width=2)),
            row=1,
            col=1,
        )

    window = 63
    for name, w, color in [
        ("Your Allocation", w_custom, "#2563eb"),
        ("Benchmark", w_bench, "#64748b"),
        ("Optimized", w_opt, "#16a34a"),
    ]:
        port_ret = returns.values @ w
        daily_rf = (1 + rf_rate) ** (1 / 252) - 1
        excess = pd.Series(port_ret - daily_rf, index=returns.index)
        rolling_sharpe = (excess.rolling(window).mean() / port_ret.std()) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name=f"{name} Sharpe",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=650, hovermode="x unified", legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    pie_col1, pie_col2 = st.columns(2)
    with pie_col1:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=[a["name"] for a in ASSETS],
                    values=w_custom * 100,
                    hole=0.35,
                )
            ]
        )
        fig_pie.update_layout(title="Your Allocation", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    with pie_col2:
        fig_pie2 = go.Figure(
            data=[
                go.Pie(
                    labels=[a["name"] for a in ASSETS],
                    values=w_opt * 100,
                    hole=0.35,
                )
            ]
        )
        fig_pie2.update_layout(title="Optimized Allocation", height=400)
        st.plotly_chart(fig_pie2, use_container_width=True)

with tab3:
    st.markdown(
        """
        ### Recommended allocation (Aug 2019 → present, hindsight-informed)

        Based on historical performance from **20 Aug 2019** through the latest available data,
        the allocation below maximizes the **Sharpe ratio** while staying within every
        benchmark ± limit constraint from the brief.
        """
    )

    rec_df = weights_to_frame(w_opt)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        | Metric | Benchmark | Optimized | Improvement |
        |--------|-----------|-----------|-------------|
        | Sharpe | {metrics_bench.sharpe_ratio:.3f} | {metrics_opt.sharpe_ratio:.3f} | **{metrics_opt.sharpe_ratio - metrics_bench.sharpe_ratio:+.3f}** |
        | Ann. Return | {metrics_bench.annualized_return*100:.1f}% | {metrics_opt.annualized_return*100:.1f}% | {metrics_opt.annualized_return*100 - metrics_bench.annualized_return*100:+.1f} pp |
        | Ann. Volatility | {metrics_bench.annualized_volatility*100:.1f}% | {metrics_opt.annualized_volatility*100:.1f}% | {metrics_opt.annualized_volatility*100 - metrics_bench.annualized_volatility*100:+.1f} pp |
        | Total Return | {metrics_bench.cumulative_return*100:.1f}% | {metrics_opt.cumulative_return*100:.1f}% | {metrics_opt.cumulative_return*100 - metrics_bench.cumulative_return*100:+.1f} pp |

        **Key tilts vs benchmark:** overweight US Large Cap (+7pp), Commodities (+5pp),
        IG Corporate (+5pp), and Bitcoin (+1pp); underweight Mid Cap (−7pp), Treasuries (−2pp),
        Real Estate (−4pp), and HY Corporate (−5pp at floor).

        > **Note on the brief's catch:** performance will be checked at two undisclosed dates
        > between Nov 2019 and today. Use the **End date** slider in the sidebar to stress-test
        > your allocation at different checkpoints before those evaluation dates are revealed.
        """
    )

st.divider()
st.caption(
    "Data: Yahoo Finance via yfinance · Tickers: IEF, IGSB, IHYA.L, SPY, SPY4.L, CMOD.L, IYR, BTC-USD"
)
