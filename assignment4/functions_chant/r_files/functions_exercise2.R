kf_filter_dt <- function(par, df) {
  # 1D state-space model for Exercise 2.2:
  # X_{t+1} = A X_t + B_Ta Ta_t + B_S S_t + B_I I_t + eta_t
  # Y_t     = X_t + epsilon_t
  A <- matrix(par[1], 1, 1)
  B <- matrix(par[2:4], 1, 3)
  Q <- matrix(exp(par[5])^2, 1, 1)
  H <- matrix(1, 1, 1)
  R_obs <- matrix(exp(par[6])^2, 1, 1)
  X0 <- matrix(par[7], 1, 1)

  obs_cols <- c("Y")
  input_cols <- c("Ta", "S", "I")

  Y <- as.matrix(df[, obs_cols, drop = FALSE])
  U <- as.matrix(df[, input_cols, drop = FALSE])
  Tn <- nrow(df)

  n <- nrow(A)
  m <- nrow(H)

  x_est <- X0
  P_est <- diag(1e1, n)
  logLik <- 0

  x_pred_store <- matrix(NA, nrow = Tn, ncol = n)
  x_filt_store <- matrix(NA, nrow = Tn, ncol = n)
  y_pred_store <- matrix(NA, nrow = Tn, ncol = m)
  innov_store <- matrix(NA, nrow = Tn, ncol = m)
  innov_var_store <- matrix(NA, nrow = Tn, ncol = m)

  for (t in 1:Tn) {
    u_t <- matrix(U[t, ], ncol = 1)
    y_t <- matrix(Y[t, ], ncol = 1)

    x_pred <- A %*% x_est + B %*% u_t
    P_pred <- A %*% P_est %*% t(A) + Q

    y_pred <- H %*% x_pred
    S_t <- H %*% P_pred %*% t(H) + R_obs
    innov <- y_t - y_pred

    if (det(S_t) <= 0 || !is.finite(det(S_t))) {
      logLik <- -Inf
      break
    }

    logLik <- logLik - 0.5 * (
      m * log(2 * pi) +
        log(det(S_t)) +
        as.numeric(t(innov) %*% solve(S_t, innov))
    )

    K_t <- P_pred %*% t(H) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- P_pred - K_t %*% H %*% P_pred

    x_pred_store[t, ] <- as.numeric(x_pred)
    x_filt_store[t, ] <- as.numeric(x_est)
    y_pred_store[t, ] <- as.numeric(y_pred)
    innov_store[t, ] <- as.numeric(innov)
    innov_var_store[t, ] <- diag(S_t)
  }

  standardized_innov <- innov_store[, 1] / sqrt(innov_var_store[, 1])

  return(list(
    logLik = as.numeric(logLik),
    x_pred = x_pred_store,
    x_filt = x_filt_store,
    y_pred = y_pred_store[, 1],
    innovation = innov_store[, 1],
    innovation_var = innov_var_store[, 1],
    standardized_innovation = standardized_innov,
    A = A,
    B = B,
    Q = Q,
    H = H,
    R = R_obs,
    X0 = X0
  ))
}

kf_logLik_dt <- function(par, df) {
  kf <- kf_filter_dt(par, df)
  return(kf$logLik)
}

estimate_dt <- function(start_par, df, lower = NULL, upper = NULL) {
  negLL <- function(par) {
    ll <- kf_logLik_dt(par, df)

    if (!is.finite(ll)) {
      return(1e12)
    }

    return(-ll)
  }

  optim(
    par = start_par,
    fn = negLL,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = list(maxit = 1000, trace = 1)
  )
}

map_par_2d <- function(par) {
  A <- matrix(par[1:4], nrow = 2, ncol = 2, byrow = TRUE)
  B <- matrix(par[5:10], nrow = 2, ncol = 3, byrow = TRUE)

  Qlt <- matrix(c(
    exp(par[11]), 0,
    par[12], exp(par[13])
  ), nrow = 2, ncol = 2, byrow = TRUE)

  Q <- Qlt %*% t(Qlt)
  H <- matrix(c(1, 0), nrow = 1, ncol = 2)
  R_obs <- matrix(exp(par[14])^2, nrow = 1, ncol = 1)
  X0 <- matrix(c(par[15], par[16]), nrow = 2, ncol = 1)

  list(A = A, B = B, Q = Q, H = H, R_obs = R_obs, X0 = X0)
}

kf_filter_dt_2d <- function(par, df, return_all = TRUE) {
  mats <- map_par_2d(par)

  A <- mats$A
  B <- mats$B
  Q <- mats$Q
  H <- mats$H
  R_obs <- mats$R_obs
  X0 <- mats$X0

  obs_cols <- c("Y")
  input_cols <- c("Ta", "S", "I")

  Y <- as.matrix(df[, obs_cols, drop = FALSE])
  U <- as.matrix(df[, input_cols, drop = FALSE])

  Tn <- nrow(df)
  n <- nrow(A)
  m <- nrow(H)

  x_est <- X0
  P_est <- diag(10, n)
  logLik <- 0

  x_pred_store <- matrix(NA, nrow = Tn, ncol = n)
  x_filt_store <- matrix(NA, nrow = Tn, ncol = n)
  y_pred_store <- numeric(Tn)
  innov_store <- numeric(Tn)
  innov_var_store <- numeric(Tn)

  for (tt in 1:Tn) {
    u_t <- matrix(U[tt, ], ncol = 1)
    y_t <- matrix(Y[tt, ], ncol = 1)

    x_pred <- A %*% x_est + B %*% u_t
    P_pred <- A %*% P_est %*% t(A) + Q

    y_pred <- H %*% x_pred
    S_t <- H %*% P_pred %*% t(H) + R_obs
    innov <- y_t - y_pred

    if (!is.finite(det(S_t)) || det(S_t) <= 0) {
      return(list(logLik = -Inf))
    }

    log_det_S <- as.numeric(determinant(S_t, logarithm = TRUE)$modulus)
    quad_form <- as.numeric(t(innov) %*% solve(S_t, innov))

    logLik <- logLik - 0.5 * (m * log(2 * pi) + log_det_S + quad_form)

    K_t <- P_pred %*% t(H) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- P_pred - K_t %*% H %*% P_pred

    x_pred_store[tt, ] <- as.numeric(x_pred)
    x_filt_store[tt, ] <- as.numeric(x_est)
    y_pred_store[tt] <- as.numeric(y_pred)
    innov_store[tt] <- as.numeric(innov)
    innov_var_store[tt] <- as.numeric(S_t)
  }

  std_resid <- innov_store / sqrt(innov_var_store)

  if (!return_all) {
    return(list(logLik = as.numeric(logLik)))
  }

  list(
    logLik = as.numeric(logLik),
    x_pred = x_pred_store,
    x_filt = x_filt_store,
    y_pred = y_pred_store,
    innovation = innov_store,
    innovation_var = innov_var_store,
    std_resid = std_resid,
    A = A,
    B = B,
    Q = Q,
    H = H,
    R = R_obs,
    X0 = X0
  )
}

kf_logLik_dt_2d <- function(par, df) {
  res <- kf_filter_dt_2d(par, df, return_all = FALSE)
  res$logLik
}

estimate_dt_2d <- function(start_par, df, lower = NULL, upper = NULL) {
  negLL <- function(par) {
    val <- kf_logLik_dt_2d(par, df)

    if (!is.finite(val)) {
      return(1e12)
    }

    -val
  }

  optim(
    par = start_par,
    fn = negLL,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = list(maxit = 2000, trace = 1)
  )
}
