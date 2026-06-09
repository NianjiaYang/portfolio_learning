"""Download and cache historical prices for all benchmark assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, PRICES_FILE, YFINANCE_TICKERS


def download_prices(start: str = "2019-08-01", end: str | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    series = {}
    for ticker, name in YFINANCE_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker} ({name})")

        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Close"][ticker]
        else:
            prices = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

        series[ticker] = prices
        print(f"  {ticker:10} {name:30} {len(prices):5} rows")

    prices_df = pd.DataFrame(series).dropna(how="any").sort_index()
    prices_df.to_csv(PRICES_FILE)
    print(f"\nSaved {len(prices_df)} aligned rows -> {PRICES_FILE}")
    return prices_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download asset price history")
    parser.add_argument("--start", default="2019-08-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    print("Downloading prices...")
    download_prices(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
