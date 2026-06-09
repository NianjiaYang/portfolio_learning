#!/usr/bin/env python3
"""Print optimized allocation and metrics for the default brief period."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    benchmark_weights,
    compare_metrics_table,
    compute_returns,
    load_prices,
    optimize_sharpe,
    slice_period,
    weights_to_frame,
)
from src.config import DEFAULT_START_DATE, PRICES_FILE  # noqa: E402


def main() -> None:
    prices = load_prices(PRICES_FILE)
    prices = slice_period(prices, DEFAULT_START_DATE, None)
    returns = compute_returns(prices)

    w_bench = benchmark_weights()
    w_opt = optimize_sharpe(returns)

    print("Optimized allocation (max Sharpe, Aug 20 2019 -> present)\n")
    print(weights_to_frame(w_opt).to_string(index=False))
    print()
    print(compare_metrics_table(returns, w_opt, w_bench, w_opt).to_string(index=False))


if __name__ == "__main__":
    main()
