import numpy as np
from scipy.optimize import minimize

def kf_log_lik_dt(par, df):
    """
    Kalman Filter Log-Likelihood calculation.
    """
    # 1. Define matrices based on parameters (Example structure)
    # You would map 'par' values to these matrices here
    A = np.array([[par[0], par[1]], [par[2], par[3]]]) 
    B = np.array([[par[4], par[5]], [par[6], par[7]]])
    
    # System covariance via Cholesky decomposition (Lower Triangle)
    Qlt = np.array([[par[8], 0], [par[9], par[10]]])
    Q = Qlt @ Qlt.T
    
    H = np.eye(1) # Observation matrix (assuming 1D observation Y)
    R_obs = np.array([[1e-2]]) # Observation noise
    X0 = np.zeros((2, 1)) # Initial state
    
    # 2. Pull out data
    obs_cols = ["Y"]
    input_cols = ["Ta", "S", "I"]
    
    Y = df[obs_cols].values.T      # (m x Tn)
    U = df[input_cols].values.T    # (p x Tn)
    Tn = df.shape[0]
    n = A.shape[0]

    # 3. Initialization
    x_est = X0
    P_est = np.eye(n) * 10
    log_lik = 0

    for t in range(Tn):
        u_t = U[:, t:t+1]
        y_t = Y[:, t:t+1]

        # --- Prediction Step ---
        # x_pred = A*x + B*u
        x_pred = A @ x_est + B @ u_t
        P_pred = A @ P_est @ A.T + Q

        # --- Innovation Step ---
        # y_pred = H*x_pred
        y_pred = H @ x_pred
        innov = y_t - y_pred
        
        # S_t = H*P*H' + R
        S_t = H @ P_pred @ H.T + R_obs
        S_inv = np.linalg.inv(S_t)

        # --- Log-Likelihood Contribution ---
        # -0.5 * (log|2*pi*S| + innov' * S^-1 * innov)
        term1 = np.log(np.linalg.det(2 * np.pi * S_t))
        term2 = innov.T @ S_inv @ innov
        log_lik -= 0.5 * (term1 + term2.item())

        # --- Update Step ---
        K_t = P_pred @ H.T @ S_inv
        x_est = x_pred + K_t @ innov
        P_est = (np.eye(n) - K_t @ H) @ P_pred

    return float(log_lik)

def estimate_dt(start_par, df, bounds=None):
    """
    Optimizer wrapper using L-BFGS-B.
    """
    def neg_ll(par):
        # maximize log-lik = minimize negative log-lik
        return -kf_log_lik_dt(par, df)

    res = minimize(
        fun=neg_ll,
        x0=start_par,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    return res