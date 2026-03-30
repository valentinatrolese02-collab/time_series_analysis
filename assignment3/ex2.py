import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import seaborn as sns

sns.set_theme(style="whitegrid")
# ============================================================
# 1. Settings
# ============================================================
DATA_PATH = "data/datasolar.csv"
IMG_DIR = "images"

mu = 5.72
phi1 = -0.38
Phi1 = -0.94
max_lag = 12

os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================
# 2. Load data and compute transformed series
# ============================================================
df = pd.read_csv(DATA_PATH)
power = df["power"].values
X = np.log(power) - mu


# ============================================================
# 3. Compute residuals
# Model:
#   (1 + phi1 B)(1 + Phi1 B^12) X_t = eps_t
# Expanded:
#   eps_t = X_t + phi1 X_{t-1} + Phi1 X_{t-12} + phi1*Phi1 X_{t-13}
# ============================================================
eps = np.full(len(X), np.nan)

for t in range(13, len(X)):
    eps[t] = (
        X[t]
        + phi1 * X[t - 1]
        + Phi1 * X[t - 12]
        + phi1 * Phi1 * X[t - 13]
    )

eps_valid = eps[13:]
time_index = np.arange(14, len(X) + 1)  # residuals start at t=14

print(f"Residual mean: {np.mean(eps_valid):.6f}")


# ============================================================
# 4. Residual plot & Histogram of residuals
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ============================================================
# Left: Residual time series
# ============================================================
axes[0].plot(time_index, eps_valid, marker="o", linewidth=1.8, markersize=5)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Residuals of the Seasonal AR Model")
axes[0].set_xlabel("Time index")
axes[0].set_ylabel(r"$\hat{\varepsilon}_t$")
axes[0].grid(True, alpha=0.3)

# ============================================================
# Right: Histogram
# ============================================================
axes[1].hist(eps_valid, bins=8, edgecolor="black", alpha=0.8)
axes[1].axvline(np.mean(eps_valid), color="red", linestyle="--", linewidth=1.5, label="Mean")
axes[1].set_title("Histogram of Residuals")
axes[1].set_xlabel(r"$\hat{\varepsilon}_t$")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.2)
axes[1].legend()

# ============================================================
# Layout + save
# ============================================================
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "plot2_1_combined.png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 5. Residual ACF plot
# ============================================================
lags = min(max_lag, len(eps_valid) - 1)

fig, ax = plt.subplots(figsize=(8, 3))

plot_acf(
    eps_valid,
    lags=lags,
    alpha=0.05,      # 95% confidence bands
    ax=ax,
    zero=True
)

ax.set_title("ACF of Residuals")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "plot2_1_acf.png"), dpi=300, bbox_inches="tight")
plt.show()
# ============================================================
# 6. Histogram of residuals
# ======================================================


# ============================================================
# 7. QQ plot
# ============================================================
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
stats.probplot(eps_valid, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot of Residuals")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "plot2_1(3).png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 8. Ljung-Box test
# ============================================================
lb_results = acorr_ljungbox(eps_valid, lags=[10], return_df=True)
print("\nLjung-Box test results:")
print(lb_results)

# ============================================================
# 1. Forecast X_t
# ============================================================

X_extended = list(X.copy())  # convert to list for appending

n_forecast = 12
T = len(X)

for k in range(1, n_forecast + 1):
    t = T + k - 1  # index in python (0-based)

    def get_value(idx):
        if idx < T:
            return X_extended[idx]
        else:
            return X_extended[idx]  # already forecasted

    x_t_1 = get_value(t - 1)
    x_t_12 = get_value(t - 12)
    x_t_13 = get_value(t - 13)

    x_hat = (
        -phi1 * x_t_1
        -Phi1 * x_t_12
        -phi1 * Phi1 * x_t_13
    )

    X_extended.append(x_hat)

# Extract forecasts
X_forecast = np.array(X_extended[T:])

Y_forecast = np.exp(X_forecast + mu)

forecast_df = pd.DataFrame({
    "k": np.arange(1, 13),
    "X_hat": X_forecast,
    "Y_hat": Y_forecast
})

print(forecast_df)

plt.figure(figsize=(10,5))

# original data
plt.plot(power, label="Observed", marker='o')

# forecast index
forecast_index = np.arange(len(power), len(power) + 12)

plt.plot(forecast_index, Y_forecast, label="Forecast", marker='o')

plt.axvline(len(power)-1, color='black', linestyle='--')

plt.title("Observed and Forecasted Power")
plt.xlabel("Time")
plt.ylabel("Power (MWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR, "plot2_2.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Prediction intervals using only the AR(1) part
# ============================================================
phi1 = -0.38
sigma_eps = 0.22
mu = 5.72

a = -phi1   # AR(1) coefficient = 0.38
k_vals = np.arange(1, 13)

# forecast error std for each horizon
s_k = sigma_eps * np.sqrt((1 - a**(2 * k_vals)) / (1 - a**2))

# 95% bounds on X-scale
X_lower = X_forecast - 1.96 * s_k
X_upper = X_forecast + 1.96 * s_k

# transform back to original power scale
Y_lower = np.exp(X_lower + mu)
Y_upper = np.exp(X_upper + mu)

# table
pi_df = pd.DataFrame({
    "k": k_vals,
    "X_hat": X_forecast,
    "X_lower": X_lower,
    "X_upper": X_upper,
    "Y_hat": Y_forecast,
    "Y_lower": Y_lower,
    "Y_upper": Y_upper
})

print(pi_df.round(3))

# observed series
plt.figure(figsize=(10, 5))

t_obs = np.arange(1, len(power) + 1)
t_for = np.arange(len(power) + 1, len(power) + 13)

plt.plot(t_obs, power, marker='o', label='Observed')
plt.plot(t_for, Y_forecast, marker='o', label='Forecast', color='C1')

# prediction intervals
plt.fill_between(t_for, Y_lower, Y_upper, color='C1', alpha=0.25, label='95% PI')

plt.axvline(len(power), color='black', linestyle='--', linewidth=1)
plt.title('Observed series and 12-step forecasts with 95% prediction intervals')
plt.xlabel('Time')
plt.ylabel('Power (MWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "plot2_2-3.png"), dpi=300, bbox_inches="tight")
plt.show()