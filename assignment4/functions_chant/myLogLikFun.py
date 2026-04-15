import numpy as np

def myLogLikFun(theta, y, R, x_prior=0.0, P_prior=10.0):
    """
    Negative log-likelihood for the scalar Kalman filter model:
        X_{t+1} = a - b*X_t + c*e_t
        y_t = X_t + noise
    """

    # Run the Kalman filter
    kf_result = myKalmanFilter(y, theta, R, x_prior, P_prior)

    err = kf_result["innovation"]        # innovations ε_t
    S = kf_result["innovation_var"]      # innovation variances S_t

    # Gaussian log-likelihood
    logL = -0.5 * np.sum(np.log(2 * np.pi * S) + (err**2) / S)

    return -logL   # return NEGATIVE log-likelihood for minimization
