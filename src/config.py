from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_FILE = DATA_DIR / "asset_prices.csv"

DEFAULT_START_DATE = "2019-08-20"
RISK_FREE_RATE = 0.02  # annualized, approximate avg short-term rate over period

ASSETS = [
    {
        "key": "IEF",
        "name": "US Treasuries (7-10y)",
        "brief_ticker": "IEF",
        "benchmark": 0.08,
        "limit": 0.04,
    },
    {
        "key": "IGSB",
        "name": "US IG Corporate (5-10y)",
        "brief_ticker": "IGSB",
        "benchmark": 0.12,
        "limit": 0.05,
    },
    {
        "key": "IHYA.L",
        "name": "US HY Corporate (5-10y HY)",
        "brief_ticker": "IHYA",
        "benchmark": 0.10,
        "limit": 0.05,
    },
    {
        "key": "SPY",
        "name": "US Large Cap (S&P 500)",
        "brief_ticker": "SP 500",
        "benchmark": 0.25,
        "limit": 0.07,
    },
    {
        "key": "SPY4.L",
        "name": "US Mid Cap Equities",
        "brief_ticker": "SPY4",
        "benchmark": 0.25,
        "limit": 0.07,
    },
    {
        "key": "CMOD.L",
        "name": "Commodities",
        "brief_ticker": "CMOD",
        "benchmark": 0.10,
        "limit": 0.05,
    },
    {
        "key": "IYR",
        "name": "Real Estate",
        "brief_ticker": "IYR",
        "benchmark": 0.08,
        "limit": 0.04,
    },
    {
        "key": "BTC-USD",
        "name": "Bitcoin",
        "brief_ticker": "Bitcoin",
        "benchmark": 0.02,
        "limit": 0.01,
    },
]

YFINANCE_TICKERS = {a["key"]: a["name"] for a in ASSETS}
