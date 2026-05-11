import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t as t_dist, norm

# ============================================================
# 1.1 – Simulate 5 Trajectories (state only)
# ============================================================

def simulate_state(a, b, sigma1, X0, n, seed=None):
    """Simulate X_t = a*X_{t-1} + b + e1_t, e1_t ~ N(0, sigma1^2)."""
    rng = np.random.default_rng(seed)
    X = np.empty(n)
    X[0] = a * X0 + b + rng.normal(0, sigma1)
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + rng.normal(0, sigma1)
    return X


np.random.seed(42)
a, b, sigma1, X0, n = 0.9, 1.0, 1.0, 5.0, 100

fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
for i in range(5):
    traj = simulate_state(a, b, sigma1, X0, n, seed=i)
    ax.plot(range(1, n + 1), traj, color=colors[i], label=f"Realization {i+1}")
ax.set_xlabel("t")
ax.set_ylabel("$X_t$")
ax.set_title("1.1 – Five independent state trajectories")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("fig_1_1.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 1.2 – Simulate State + Noisy Observations
# ============================================================

def simulate_state_obs(a, b, sigma1, sigma2, X0, n, seed=None):
    """Return (X, Y) where Y_t = X_t + e2_t, e2_t ~ N(0, sigma2^2)."""
    rng = np.random.default_rng(seed)
    X = np.empty(n)
    X[0] = a * X0 + b + rng.normal(0, sigma1)
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + rng.normal(0, sigma1)
    Y = X + rng.normal(0, sigma2, size=n)
    return X, Y


R_obs = 1.0  # observation noise variance (sigma2^2), fixed throughout
X_true, Y_obs = simulate_state_obs(a, b, sigma1, np.sqrt(R_obs), X0, n, seed=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, n + 1), X_true, color="steelblue", label="True state $X_t$")
ax.plot(range(1, n + 1), Y_obs, color="tomato", linestyle="--", alpha=0.7,
        label="Observations $Y_t$")
ax.set_xlabel("t")
ax.set_ylabel("Value")
ax.set_title("1.2 – True state vs. noisy observations")
ax.legend()
plt.tight_layout()
plt.savefig("fig_1_2.png", dpi=150, bbox_inches="tight")
plt.show()
# Y_t tracks X_t but is scattered around it; individual deviations can be large,
# making direct recovery of X_t from Y_t non-trivial.

# ============================================================
# 1.3 – Kalman Filter
# ============================================================

def kalman_filter(y, a, b, sigma1, R, x_prior=0.0, P_prior=10.0):
    """
    Scalar 1-D Kalman filter.

    Predict:  x_pred[t] = a*x_filt[t-1] + b
              P_pred[t] = a^2 * P_filt[t-1] + sigma1^2
    Update:   innovation[t]     = y[t] - x_pred[t]
              innovation_var[t] = P_pred[t] + R
              K                 = P_pred[t] / innovation_var[t]
              x_filt[t]         = x_pred[t] + K * innovation[t]
              P_filt[t]         = (1 - K) * P_pred[t]
    """
    N = len(y)
    x_pred = np.empty(N)
    P_pred = np.empty(N)
    x_filt = np.empty(N)
    P_filt = np.empty(N)
    innovation = np.empty(N)
    innovation_var = np.empty(N)

    Q = sigma1 ** 2  # system noise variance

    for t in range(N):
        # --- predict ---
        if t == 0:
            x_pred[t] = a * x_prior + b
            P_pred[t] = a ** 2 * P_prior + Q
        else:
            x_pred[t] = a * x_filt[t - 1] + b
            P_pred[t] = a ** 2 * P_filt[t - 1] + Q

        # --- update ---
        innovation[t] = y[t] - x_pred[t]
        innovation_var[t] = P_pred[t] + R
        K = P_pred[t] / innovation_var[t]
        x_filt[t] = x_pred[t] + K * innovation[t]
        P_filt[t] = (1.0 - K) * P_pred[t]

    return dict(x_pred=x_pred, P_pred=P_pred, x_filt=x_filt, P_filt=P_filt,
                innovation=innovation, innovation_var=innovation_var)


kf = kalman_filter(Y_obs, a, b, sigma1, R_obs)

ci = 1.96 * np.sqrt(kf["P_pred"])
t_axis = np.arange(1, n + 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_axis, X_true, color="steelblue", label="True state $X_t$", zorder=3)
ax.plot(t_axis, Y_obs, ".", color="tomato", alpha=0.5, label="Observations $Y_t$", zorder=2)
ax.plot(t_axis, kf["x_pred"], color="darkorange", label=r"Predicted $\hat{X}_{t|t-1}$", zorder=4)
ax.fill_between(t_axis,
                kf["x_pred"] - ci,
                kf["x_pred"] + ci,
                color="darkorange", alpha=0.2, label="95% CI")
ax.set_xlabel("t")
ax.set_ylabel("Value")
ax.set_title("1.3 – Kalman filter: predicted state with 95% CI")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_1_3.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 1.4 – Maximum Likelihood Estimation (Monte Carlo)
# ============================================================

def neg_log_likelihood(theta, y, R, x_prior=0.0, P_prior=10.0):
    a_, b_, sigma1_ = theta
    if sigma1_ <= 0:
        return 1e10
    kf_ = kalman_filter(y, a_, b_, sigma1_, R, x_prior, P_prior)
    v = kf_["innovation"]
    S = kf_["innovation_var"]
    logL = -0.5 * np.sum(np.log(2 * np.pi * S) + v ** 2 / S)
    return -logL


def estimate_parameters(y, R=1.0, x_prior=0.0, P_prior=10.0):
    theta0 = [0.5, 0.0, 1.0]
    bounds = [(-0.999, 0.999), (None, None), (1e-4, None)]
    result = minimize(neg_log_likelihood, theta0, args=(y, R, x_prior, P_prior),
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-12})
    return result.x


cases = [
    {"a": 0.9, "b": 1.0, "sigma1": 1.0, "label": r"$a=0.9,\,b=1,\,\sigma_1=1$"},
    {"a": 0.9, "b": 5.0, "sigma1": 1.0, "label": r"$a=0.9,\,b=5,\,\sigma_1=1$"},
    {"a": 0.9, "b": 1.0, "sigma1": 5.0, "label": r"$a=0.9,\,b=1,\,\sigma_1=5$"},
]

M = 100  # Monte Carlo replications
param_names = ["a", "b", r"$\sigma_1$"]
true_labels = ["a", "b", "sigma1"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for case_idx, case in enumerate(cases):
    estimates = np.empty((M, 3))
    for m in range(M):
        _, Y_m = simulate_state_obs(case["a"], case["b"], case["sigma1"],
                                    np.sqrt(R_obs), X0, n, seed=1000 * case_idx + m)
        estimates[m] = estimate_parameters(Y_m, R=R_obs)

    true_vals = [case["a"], case["b"], case["sigma1"]]
    for j, ax in enumerate(axes):
        ax.boxplot(estimates[:, j], positions=[case_idx + 1],
                   widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=f"C{case_idx}", alpha=0.5),
                   medianprops=dict(color="black"),
                   showfliers=False)

for j, ax in enumerate(axes):
    true_by_case = [c[true_labels[j]] for c in cases]
    for case_idx, tv in enumerate(true_by_case):
        ax.hlines(tv, case_idx + 0.6, case_idx + 1.4, colors=f"C{case_idx}",
                  linestyles="dashed", linewidth=1.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([c["label"] for c in cases], fontsize=7)
    ax.set_ylabel("Estimated value")
    ax.set_title(f"Parameter {param_names[j]}")

fig.suptitle("1.4 – MLE parameter estimates (100 replications per case)")
plt.tight_layout()
plt.savefig("fig_1_4.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 1.5 – Robustness to Heavy-Tailed (Student-t) System Noise
# ============================================================

# --- Part A: density comparison ---
x_range = np.linspace(-6, 6, 400)
nu_values = [100, 5, 2, 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_range, norm.pdf(x_range), "k-", linewidth=2, label=r"$N(0,1)$")
for nu in nu_values:
    ax.plot(x_range, t_dist.pdf(x_range, df=nu), label=rf"$t(\nu={nu})$")
ax.set_xlim(-6, 6)
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title(r"1.5 – t-distribution densities vs. $N(0,1)$")
ax.legend()
plt.tight_layout()
plt.savefig("fig_1_5a.png", dpi=150, bbox_inches="tight")
plt.show()


# --- Part B: simulate with t-distributed system noise ---
def simulate_state_obs_t(a, b, sigma1, sigma2, nu, X0, n, seed=None):
    """Like simulate_state_obs but system noise is sigma1 * t(nu)."""
    rng = np.random.default_rng(seed)
    X = np.empty(n)
    X[0] = a * X0 + b + sigma1 * t_dist.rvs(df=nu, random_state=rng)
    for t in range(1, n):
        X[t] = a * X[t - 1] + b + sigma1 * t_dist.rvs(df=nu, random_state=rng)
    Y = X + rng.normal(0, sigma2, size=n)
    return X, Y


# --- Part C & D: estimate on t-noise data, compare to Gaussian baseline ---

# Gaussian baseline (nu → ∞, same as Case 1 in 1.4)
gaussian_estimates = np.empty((M, 3))
for m in range(M):
    _, Y_m = simulate_state_obs(0.9, 1.0, 1.0, np.sqrt(R_obs), X0, n, seed=m)
    gaussian_estimates[m] = estimate_parameters(Y_m, R=R_obs)

t_estimates = {}
for nu in nu_values:
    est = np.empty((M, 3))
    for m in range(M):
        _, Y_m = simulate_state_obs_t(0.9, 1.0, 1.0, np.sqrt(R_obs), nu,
                                      X0, n, seed=m)
        est[m] = estimate_parameters(Y_m, R=R_obs)
    t_estimates[nu] = est

# Build boxplot figure
all_labels = [r"$\nu=\infty$ (Gaussian)"] + [rf"$\nu={nu}$" for nu in nu_values]
all_data = [gaussian_estimates] + [t_estimates[nu] for nu in nu_values]
true_vals_15 = [0.9, 1.0, 1.0]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
positions = np.arange(1, len(all_labels) + 1)

for j, ax in enumerate(axes):
    data_j = [d[:, j] for d in all_data]
    bp = ax.boxplot(data_j, positions=positions, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black"), showfliers=False)
    colors_bp = ["dimgray"] + [f"C{i}" for i in range(len(nu_values))]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(true_vals_15[j], color="red", linestyle="--", linewidth=1.5,
               label="True value")
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=7, rotation=15)
    ax.set_ylabel("Estimated value")
    ax.set_title(f"Parameter {param_names[j]}")
    ax.legend(fontsize=7)

fig.suptitle("1.5 – MLE estimates under Gaussian KF with t-distributed system noise")
plt.tight_layout()
plt.savefig("fig_1_5b.png", dpi=150, bbox_inches="tight")
plt.show()
