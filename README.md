# TrueBeta

**Dynamic stock beta estimation using Kalman filtering.**

TrueBeta is a web application that estimates time-varying stock betas using a Kalman filter state-space model, compared side-by-side against traditional rolling OLS regression. It lets you look up any ticker and instantly see how its market sensitivity has changed over time — with interactive charts, confidence bands, and cumulative alpha analysis.

## Why TrueBeta?

Traditional beta estimation uses a fixed rolling window of historical returns (e.g., 252 days). This approach has fundamental problems:

- **Arbitrary window choice** — different windows give dramatically different results
- **Lagging** — slow to react to regime changes
- **Abrupt jumps** — extreme observations entering/leaving the window create discontinuities
- **Equal weighting** — treats data from 11 months ago the same as yesterday

TrueBeta uses a **Kalman filter** to model beta as a continuously evolving state variable. The filter optimally balances new observations against its model prediction at every time step, producing smooth, adaptive beta estimates with built-in uncertainty quantification.

### The Model

All estimates use **excess returns** (net of the risk-free rate via the 13-week Treasury bill), consistent with the CAPM framework:

```
Observation:  (r_stock - r_f) = α + β · (r_market - r_f) + ε
State:        β_t = β_{t-1} + w_t    (random walk)
```

The Kalman filter recursively updates α and β at each time step, with the **Kalman gain** determining the optimal weighting between the model prediction and new market data.

## Features

- **Kalman filter beta** with 95% confidence bands
- **Rolling OLS beta** for comparison, with configurable window (3M, 6M, 1Y, 2Y)
- **Configurable display range** (1Y, 2Y, 3Y, 5Y, 10Y)
- **Interactive Plotly charts** — beta comparison, normalized prices, cumulative alpha
- **Autocomplete ticker search** from S&P 500 constituents
- **Excess return estimation** using the 13-week T-bill rate (^IRX)
- **On-the-fly computation** with in-memory TTL caching
- **KaTeX-rendered mathematics** explaining the methodology

## Tech Stack

- **FastAPI** — async Python web framework
- **NumPy** — Kalman filter implementation (~50 lines, no external filter library)
- **Pandas / statsmodels** — data handling and rolling OLS
- **yfinance** — stock price and Treasury rate data
- **Plotly.js** — interactive client-side charts
- **TailwindCSS** — styling (via CDN)
- **KaTeX** — LaTeX math rendering (via CDN)

## Project Structure

```
truebeta/
├── main.py              # FastAPI app entry point
├── requirements.txt     # Python dependencies
├── Procfile             # Railway deployment
├── core/
│   ├── kalman.py        # Kalman filter beta estimation
│   ├── ols.py           # Rolling OLS beta estimation
│   ├── data.py          # yfinance data fetching + caching
│   ├── charts.py        # Plotly chart builders
│   └── sp500.py         # S&P 500 ticker list from Wikipedia
└── web/
    ├── router.py        # FastAPI routes
    └── templates/
        ├── base.html    # Layout with nav, footer, autocomplete
        ├── index.html   # Landing page with methodology explanation
        ├── ticker.html  # Ticker results with interactive charts
        └── 404.html     # Not found page
```

## Running Locally

```bash
pip install -r requirements.txt
python main.py
```

Then open [http://localhost:8000](http://localhost:8000).

## Deployment

Deployed on [Railway](https://railway.app) with automatic deploys from this repo. The `Procfile` runs uvicorn on the `$PORT` provided by Railway.

## Data Sources

- **Stock prices & S&P 500 index** — Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance)
- **Risk-free rate** — 13-week Treasury Bill (^IRX) from Yahoo Finance
- **S&P 500 constituents** — Wikipedia

## License

MIT
