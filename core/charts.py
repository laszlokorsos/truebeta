import json
import numpy as np
import plotly.graph_objects as go


KALMAN_COLOR = "#2563eb"       # blue-600
KALMAN_BAND_COLOR = "rgba(37, 99, 235, 0.12)"
OLS_COLOR = "#f97316"          # orange-500
STOCK_COLOR = "#2563eb"
MARKET_COLOR = "#94a3b8"       # slate-400

WINDOW_LABELS = {63: "3M", 126: "6M", 252: "1Y", 504: "2Y"}

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#475569"),
    margin=dict(l=55, r=20, t=20, b=45),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.12,
        xanchor="center", x=0.5,
        font=dict(size=12),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        showgrid=False,
        linecolor="#e2e8f0",
    ),
    yaxis=dict(
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        zeroline=False,
    ),
)


def _window_label(window):
    return WINDOW_LABELS.get(window, f"{window}d")


def beta_chart(dates, kalman_result, ols_result, ticker, ols_window=252):
    """Interactive beta comparison chart: Kalman (with confidence band) vs OLS."""
    dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates]
    betas_k = kalman_result["betas"]
    std_k = kalman_result["beta_std"]
    betas_ols = ols_result["betas"]

    upper = betas_k + 2 * std_k
    lower = betas_k - 2 * std_k

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=dates_str + dates_str[::-1],
        y=np.concatenate([upper, lower[::-1]]).tolist(),
        fill="toself",
        fillcolor=KALMAN_BAND_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Kalman beta
    fig.add_trace(go.Scatter(
        x=dates_str, y=betas_k.tolist(),
        mode="lines", name="Kalman Beta",
        line=dict(color=KALMAN_COLOR, width=2.5),
    ))

    # OLS beta
    wl = _window_label(ols_window)
    fig.add_trace(go.Scatter(
        x=dates_str, y=[v if not np.isnan(v) else None for v in betas_ols],
        mode="lines", name=f"OLS Beta ({wl})",
        line=dict(color=OLS_COLOR, width=2, dash="dot"),
    ))

    # Reference line at beta=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="#cbd5e1", line_width=1,
                  annotation_text="&beta; = 1",
                  annotation_position="bottom right",
                  annotation_font_size=11,
                  annotation_font_color="#94a3b8")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis_title="Beta",
    )

    return json.loads(fig.to_json())


def price_chart(prices, ticker):
    """Normalized price chart (both series indexed to 100 at start of display range)."""
    norm = prices / prices.iloc[0] * 100
    dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in norm.index]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates_str, y=norm["stock"].round(2).tolist(),
        mode="lines", name=ticker,
        line=dict(color=STOCK_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates_str, y=norm["market"].round(2).tolist(),
        mode="lines", name="S&P 500",
        line=dict(color=MARKET_COLOR, width=2),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis_title="Indexed Price (100 = Start)",
    )

    return json.loads(fig.to_json())


def alpha_chart(dates, excess_stock, excess_market, kalman_result, ols_result, ols_window=252):
    """Cumulative residual alpha chart for both models.

    Uses excess returns (already risk-free adjusted).
    Both series start at 0 at the beginning of the (already-trimmed) display range.
    """
    dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates]

    # Alpha = excess_stock - beta * excess_market, cumulated from display start
    kalman_resid = excess_stock - kalman_result["betas"] * excess_market
    kalman_cum = ((1 + kalman_resid).cumprod() - 1) * 100

    ols_resid = excess_stock - ols_result["betas"] * excess_market
    ols_cum = ((1 + ols_resid).cumprod() - 1) * 100

    wl = _window_label(ols_window)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates_str, y=np.round(kalman_cum, 2).tolist(),
        mode="lines", name="Kalman Alpha",
        line=dict(color=KALMAN_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates_str, y=np.round(ols_cum, 2).tolist(),
        mode="lines", name=f"OLS Alpha ({wl})",
        line=dict(color=OLS_COLOR, width=2, dash="dot"),
    ))

    fig.add_hline(y=0, line_color="#e2e8f0", line_width=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        yaxis_title="Cumulative Alpha (%)",
    )

    return json.loads(fig.to_json())
