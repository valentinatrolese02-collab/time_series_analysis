import numpy as np

def myKalmanFilter(y, theta, R, x_prior=0.0, P_prior=10.0):
    """
    Scalar Kalman filter for:
        X_{t+1} = a - b*X_t + c*e_t
        y_t = X_t + noise
    """

    a = theta[0]
    b = theta[1]
    sigma1 = theta[2]
    Q = sigma1**2

    y = np.asarray(y)
    N = len(y)

    x_pred = np.zeros(N)
    P_pred = np.zeros(N)
    x_filt = np.zeros(N)
    P_filt = np.zeros(N)
    innovation = np.zeros(N)
    innovation_var = np.zeros(N)

    for t in range(N):

        # -----------------------------
        # Prediction step
        # -----------------------------
        if t == 0:
            x_pred[t] = a - b * x_prior
            P_pred[t] = b**2 * P_prior + Q
        else:
            x_pred[t] = a - b * x_filt[t-1]
            P_pred[t] = b**2 * P_filt[t-1] + Q

        # -----------------------------
        # Update step
        # -----------------------------
        innovation[t] = y[t] - x_pred[t]
        innovation_var[t] = P_pred[t] + R

        K_t = P_pred[t] / innovation_var[t]

        x_filt[t] = x_pred[t] + K_t * innovation[t]
        P_filt[t] = (1 - K_t) * P_pred[t]

    return {
        "x_pred": x_pred,
        "P_pred": P_pred,
        "x_filt": x_filt,
        "P_filt": P_filt,
        "innovation": innovation,
        "innovation_var": innovation_var
    }
