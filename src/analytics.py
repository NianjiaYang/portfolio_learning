from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import ASSETS, RISK_FREE_RATE


@dataclass
class PortfolioMetrics:
    weights: np.ndarray
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    cumulative_return: float
    max_drawdown: float


def load_prices(path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def slice_period(
    prices: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp | None
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else prices.index.max()
    return prices.loc[start_ts:end_ts].copy()


def benchmark_weights() -> np.ndarray:
    return np.array([a["benchmark"] for a in ASSETS])


def allocation_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.array([a["benchmark"] - a["limit"] for a in ASSETS])
    high = np.array([a["benchmark"] + a["limit"] for a in ASSETS])
    return low, high


def portfolio_value(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    normalized = prices / prices.iloc[0]
    return (normalized * weights).sum(axis=1)


def portfolio_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PortfolioMetrics:
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    port_returns = returns.values @ weights
    excess = port_returns - daily_rf

    ann_return = (1 + port_returns.mean()) ** 252 - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (excess.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() > 0 else 0.0
    cumulative = (1 + port_returns).prod() - 1

    wealth = (1 + pd.Series(port_returns, index=returns.index)).cumprod()
    rolling_max = wealth.cummax()
    drawdown = wealth / rolling_max - 1
    max_drawdown = drawdown.min()

    return PortfolioMetrics(
        weights=weights,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        cumulative_return=cumulative,
        max_drawdown=max_drawdown,
    )


def optimize_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    n_starts: int = 30,
) -> np.ndarray:
    low, high = allocation_bounds()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(float(l), float(h)) for l, h in zip(low, high)]

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    ret_matrix = returns.values

    def neg_sharpe(weights: np.ndarray) -> float:
        port = ret_matrix @ weights
        excess = port - daily_rf
        vol = port.std()
        if vol <= 0:
            return 0.0
        return -(excess.mean() / vol) * np.sqrt(252)

    best_result = None
    for seed in range(n_starts):
        rng = np.random.default_rng(seed)
        x0 = rng.dirichlet(np.ones(len(ASSETS)))
        x0 = np.clip(x0, low, high)
        x0 = x0 / x0.sum()

        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        return benchmark_weights()

    weights = best_result.x
    weights = np.clip(weights, low, high)
    return weights / weights.sum()


def weights_to_frame(weights: np.ndarray) -> pd.DataFrame:
    rows = []
    low, high = allocation_bounds()
    for asset, weight, lo, hi in zip(ASSETS, weights, low, high):
        rows.append(
            {
                "Asset": asset["name"],
                "Ticker": asset["brief_ticker"],
                "Weight %": round(weight * 100, 2),
                "Benchmark %": round(asset["benchmark"] * 100, 2),
                "Min %": round(lo * 100, 2),
                "Max %": round(hi * 100, 2),
                "vs Benchmark": round((weight - asset["benchmark"]) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def compare_metrics_table(
    returns: pd.DataFrame,
    custom_weights: np.ndarray,
    benchmark_w: np.ndarray | None = None,
    optimized_w: np.ndarray | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    benchmark_w = benchmark_w if benchmark_w is not None else benchmark_weights()
    rows = []

    portfolios = [("Benchmark", benchmark_w)]
    if optimized_w is not None:
        portfolios.append(("Optimized (Max Sharpe)", optimized_w))
    portfolios.append(("Your Allocation", custom_weights))

    for label, weights in portfolios:
        metrics = portfolio_metrics(returns, weights, risk_free_rate)
        rows.append(
            {
                "Portfolio": label,
                "Ann. Return %": round(metrics.annualized_return * 100, 2),
                "Ann. Volatility %": round(metrics.annualized_volatility * 100, 2),
                "Sharpe Ratio": round(metrics.sharpe_ratio, 3),
                "Total Return %": round(metrics.cumulative_return * 100, 2),
                "Max Drawdown %": round(metrics.max_drawdown * 100, 2),
            }
        )
    return pd.DataFrame(rows)
