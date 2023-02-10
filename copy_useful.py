import numpy as np
from scipy.optimize import minimize


def make_design_matrix(stim, d=25):
    """Create time-lag design matrix from stimulus intensity vector.
  Args:
    stim (1D array): Stimulus intensity at each time point.
    d (number): Number of time lags to use.
  Returns
    X (2D array): GLM design matrix with shape T, d
  """
    # Create version of stimulus vector with zeros before onset
    padded_stim = np.concatenate([np.zeros(d - 1), stim])

    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(stim)  # Total number of timepoints (hint: number of stimulus frames)
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t:t + d]

    return X


def neg_log_lik_lnp(theta, X, y):
    """Return -loglike for the Poisson GLM model.
  Args:
    theta (1D array): Parameter vector.
    X (2D array): Full design matrix.
    y (1D array): Data values.
  Returns:
    number: Negative log likelihood.
  """
    # Compute the Poisson log likelihood
    rate = np.exp(X @ theta)
    log_lik = y @ np.log(rate) - rate.sum()
    return -log_lik


def fit_lnp(stim, spikes, d=25):
    """Obtain MLE parameters for the Poisson GLM.
  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    d (number): Number of time lags to use.
  Returns:
    1D array: MLE parameters
  """

    # Build the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim)])

    # Use a random vector of weights to start (mean 0, sd .2)
    x0 = np.random.normal(0, .2, d + 1)

    # Find parameters that minmize the negative log likelihood function
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))

    return res["x"]


def predict_spike_counts_lnp(stim, spikes, theta=None, d=25):
    """Compute a vector of predicted spike counts given the stimulus.
  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    theta (1D array): Filter weights; estimated if not provided.
    d (number): Number of time lags to use.
  Returns:
    yhat (1D array): Predicted spikes at each timepoint.
  """
    y = spikes
    constant = np.ones_like(spikes)
    X = np.column_stack([constant, make_design_matrix(stim)])
    if theta is None:  # Allow pre-cached weights, as fitting is slow
        theta = fit_lnp(X, y, d)

    yhat = np.exp(X @ theta)
    return yhat
