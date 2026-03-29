import math
import numpy as np
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from core.data import fetch_prices, compute_returns
from core.kalman import kalman_beta
from core.ols import rolling_ols_beta
from core.charts import beta_chart, price_chart, alpha_chart
from core.sp500 import get_sp500_tickers

import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(_BASE_DIR, "web", "templates"))

# Allowed presets — display range in years and OLS window in trading days
RANGE_OPTIONS = [
    {"value": "1y", "label": "1Y", "years": 1},
    {"value": "2y", "label": "2Y", "years": 2},
    {"value": "3y", "label": "3Y", "years": 3},
    {"value": "5y", "label": "5Y", "years": 5},
    {"value": "10y", "label": "10Y", "years": 10},
]

WINDOW_OPTIONS = [
    {"value": "63", "label": "3M", "days": 63},
    {"value": "126", "label": "6M", "days": 126, "default": True},
    {"value": "252", "label": "1Y", "days": 252},
    {"value": "504", "label": "2Y", "days": 504},
]

RANGE_MAP = {r["value"]: r for r in RANGE_OPTIONS}
WINDOW_MAP = {w["value"]: w for w in WINDOW_OPTIONS}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail(
    request: Request,
    ticker: str,
    range: str = Query("5y", alias="range"),
    window: str = Query("126", alias="window"),
):
    ticker = ticker.upper().strip()

    # Validate presets (fall back to defaults if invalid)
    range_opt = RANGE_MAP.get(range, RANGE_MAP["5y"])
    window_opt = WINDOW_MAP.get(window, WINDOW_MAP["252"])
    display_years = range_opt["years"]
    ols_window = window_opt["days"]

    # Fetch enough data: display range + OLS warm-up + buffer
    fetch_years = display_years + math.ceil(ols_window / 252) + 1
    try:
        prices = fetch_prices(ticker, years=fetch_years)
    except ValueError as e:
        return templates.TemplateResponse(request, "404.html", {
            "message": str(e),
        }, status_code=404)

    returns = compute_returns(prices)
    stock_excess = returns["stock_excess"].values
    market_excess = returns["market_excess"].values

    # Run both estimators on excess returns (full warm-up on ALL data)
    kalman_result = kalman_beta(stock_excess, market_excess)
    ols_result = rolling_ols_beta(stock_excess, market_excess, window=ols_window)

    # Find the display trim index:
    # We want to show `display_years` of data counting back from today,
    # but never before OLS has its first valid value.
    display_trading_days = int(display_years * 252)
    total_days = len(returns)

    # Start of display range (counting from end)
    display_start = max(0, total_days - display_trading_days)
    # OLS first valid index
    ols_first_valid = ols_window - 1
    # The actual trim point is whichever comes later
    trim = max(display_start, ols_first_valid)

    # Trim all arrays
    dates_trimmed = returns.index[trim:]
    kalman_trimmed = {
        "betas": kalman_result["betas"][trim:],
        "alphas": kalman_result["alphas"][trim:],
        "beta_std": kalman_result["beta_std"][trim:],
    }
    ols_trimmed = {
        "betas": ols_result["betas"][trim:],
        "alphas": ols_result["alphas"][trim:],
    }
    stock_excess_trimmed = stock_excess[trim:]
    market_excess_trimmed = market_excess[trim:]

    # Trim prices for price chart (need the row BEFORE trim for returns alignment)
    # Use trim+1 on prices because returns[i] = prices[i+1]/prices[i] - 1
    prices_trimmed = prices.iloc[trim:]

    # Build charts with trimmed data
    chart_beta_data = beta_chart(dates_trimmed, kalman_trimmed, ols_trimmed, ticker, ols_window)
    chart_price_data = price_chart(prices_trimmed, ticker)
    chart_alpha_data = alpha_chart(
        dates_trimmed, stock_excess_trimmed, market_excess_trimmed,
        kalman_trimmed, ols_trimmed, ols_window,
    )

    # Current values
    current_kalman_beta = round(float(kalman_trimmed["betas"][-1]), 3)
    current_ols_beta = round(float(ols_trimmed["betas"][-1]), 3) if not np.isnan(ols_trimmed["betas"][-1]) else None
    last_date = dates_trimmed[-1].strftime("%Y-%m-%d") if hasattr(dates_trimmed[-1], "strftime") else str(dates_trimmed[-1])

    # Company name lookup
    company_name = ticker
    try:
        sp500 = get_sp500_tickers()
        match = next((s for s in sp500 if s["ticker"] == ticker), None)
        if match:
            company_name = match["name"]
    except Exception:
        pass

    return templates.TemplateResponse(request, "ticker.html", {
        "ticker": ticker,
        "company_name": company_name,
        "current_kalman_beta": current_kalman_beta,
        "current_ols_beta": current_ols_beta,
        "last_date": last_date,
        "chart_beta": chart_beta_data,
        "chart_price": chart_price_data,
        "chart_alpha": chart_alpha_data,
        "range_options": RANGE_OPTIONS,
        "window_options": WINDOW_OPTIONS,
        "selected_range": range_opt["value"],
        "selected_window": window_opt["value"],
        "selected_window_label": window_opt["label"],
    })


@router.get("/api/search")
async def search_tickers(q: str = Query("", min_length=1)):
    """Autocomplete endpoint for ticker search."""
    q = q.upper().strip()
    try:
        sp500 = get_sp500_tickers()
    except Exception:
        return JSONResponse([])

    matches = [
        {"ticker": s["ticker"], "name": s["name"]}
        for s in sp500
        if q in s["ticker"] or q in s["name"].upper()
    ]
    return JSONResponse(matches[:15])
