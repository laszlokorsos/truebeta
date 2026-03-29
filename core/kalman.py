import numpy as np


def kalman_beta(stock_returns, market_returns, q_beta=1e-4, q_alpha=1e-5, r=1e-3):
    """Estimate time-varying beta using a Kalman filter.

    State-space model:
        State:       x_t = [alpha_t, beta_t]'  (random walk)
        Transition:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
        Observation: r_i,t = [1, r_m,t] @ x_t + v_t,  v_t ~ N(0, R)

    Args:
        stock_returns: array of stock daily returns
        market_returns: array of market daily returns (same length)
        q_beta: state noise variance for beta (controls smoothness)
        q_alpha: state noise variance for alpha
        r: observation noise variance

    Returns:
        dict with keys: betas, alphas, beta_std (arrays, same length as input)
    """
    stock_returns = np.asarray(stock_returns, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    n = len(stock_returns)

    # State: [alpha, beta]
    x = np.array([0.0, 1.0])       # prior: alpha=0, beta=1 (market-neutral start)
    P = np.eye(2)                   # prior uncertainty (diffuse)
    Q = np.diag([q_alpha, q_beta])  # state noise covariance
    F = np.eye(2)                   # transition matrix (random walk)

    betas = np.empty(n)
    alphas = np.empty(n)
    beta_var = np.empty(n)

    for t in range(n):
        # --- Predict ---
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # --- Observation model ---
        H = np.array([[1.0, market_returns[t]]])
        y = stock_returns[t]

        # --- Update ---
        S = (H @ P_pred @ H.T)[0, 0] + r   # innovation variance (scalar)
        K = (P_pred @ H.T) / S              # Kalman gain (2x1)
        innovation = y - (H @ x_pred)[0]    # measurement residual
        x = x_pred + K.flatten() * innovation
        P = (np.eye(2) - K @ H) @ P_pred

        # --- Store ---
        betas[t] = x[1]
        alphas[t] = x[0]
        beta_var[t] = P[1, 1]

    return {
        "betas": betas,
        "alphas": alphas,
        "beta_std": np.sqrt(np.maximum(beta_var, 0)),
    }
