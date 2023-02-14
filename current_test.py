import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def make_stim_time_course(variate, d):
    n, t = variate.shape
    padding = np.zeros(shape=(n, d-1))
    variate_pad = np.concatenate([padding, variate], axis=1).reshape(-1, n)  # shape: [t + d - 1, n]
    res = np.zeros(shape=(t, d * n))
    for idx in range(t):
        res[idx] = variate_pad[idx: idx+d].reshape(-1)  # shape: [d * n]
    return res


def make_stim_time_course_3d(variate, d):
    n, t = variate.shape
    padding = np.zeros(shape=(n, d-1))
    variate_pad = np.concatenate([padding, variate], axis=1).reshape(-1, n)  # shape: [t + d - 1, n]
    res = np.zeros(shape=(t, d, n))
    for idx in range(t):
        res[idx] = variate_pad[idx: idx+d]
    return res


def make_padding_stim(variate, d):
    n, t = variate.shape
    padding = np.zeros(shape=(n, d-1))
    variate_pad = np.concatenate([padding, variate], axis=1).reshape(-1, n)  # shape: [t + d - 1, n]
    return variate_pad


def get_positive_likelihood(theta, X, y, d):
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_stim_time_course(X, d)])
    return -negative_log_likelihood(theta, X, y, 0)


def negative_log_likelihood(theta, X, y, gamma):
    # theta shape: [n * d]
    rate = np.exp(X @ theta)
    log_likelihood = y @ np.log(rate) - rate.sum() - gamma * np.linalg.norm(theta, 2)
    return -log_likelihood


def fit_reshaped_mat(stim, y, n, gamma, d):
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_stim_time_course(stim, d)])
    x0 = np.random.normal(0, .2, n * d + 1)
    res = minimize(negative_log_likelihood, x0, args=(X, y, gamma))
    return res["x"]


def predict_reshaped_mat(stim, y, theta, d):
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_stim_time_course(stim, d)])
    y_hat = np.exp(X @ theta)
    return y_hat


def process_theta(theta, d=25):
    res = theta[1:]
    res = res.reshape(d, -1).transpose((1, 0))
    return res


def plot_all(f, spike, pred, title=None):
    plt.subplot(311)
    plt.plot(f, label='F', c='b')
    plt.legend()
    plt.subplot(312)
    plt.plot(spike, label='gt', c='g')
    plt.legend()
    plt.subplot(313)
    plt.plot(pred, label='pred', c='r')
    plt.legend()
    if title is not None:
        plt.suptitle(title)
    plt.show(block=True)


def cal_aic(theta, variate, spike, d):
    return 2 * len(variate) - get_positive_likelihood(theta, variate, spike, d)


def cal_single_delta_aic(spike, variate, gamma, d):
    def ablation(_index, _mat):
        if _index == 0:
            excluded_mat = _mat[1:]
        elif _index == len(_mat) - 1:
            excluded_mat = _mat[: -1]
        else:
            excluded_mat = np.concatenate([_mat[: _index], _mat[_index + 1:]], axis=0)
        return excluded_mat
    n = len(variate)
    res = np.zeros(shape=(n, ))
    raw = np.zeros_like(res)
    partial = np.zeros_like(res)
    theta_all = fit_reshaped_mat(variate, spike, n, gamma, d)
    aic_all = cal_aic(theta_all, variate, spike, d)
    for idx in range(n):
        ablated_v = ablation(idx, variate)
        theta_tmp = fit_reshaped_mat(ablated_v, spike, n-1, gamma, d)
        aic_tmp = cal_aic(theta_tmp, ablated_v, spike, d)
        res[idx] = aic_tmp - aic_all
        raw[idx] = aic_all
        partial[idx] = aic_tmp
    return res, raw, partial
