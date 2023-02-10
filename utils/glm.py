import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


def make_stim_time_course(variate, d):
    n, t = variate.shape
    padding = np.zeros(shape=(n, d-1))
    variate_pad = np.concatenate([padding, variate], axis=1).reshape(-1, n)  # shape: [t + d - 1, n]
    res = np.zeros(shape=(t, d * n))
    for idx in range(t):
        res[idx] = variate_pad[idx: idx+d].reshape(-1)  # shape: [d * n]
    return res


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


def cal_neuron_mat_nmf(mat, index):
    if index == 0:
        excluded_mat = mat[1:]
    elif index == len(mat) - 1:
        excluded_mat = mat[: -1]
    else:
        excluded_mat = np.concatenate([mat[: index], mat[index+1:]], axis=0)
    nmf = NMF(n_components=1)
    return nmf.fit_transform(excluded_mat)


def generate_stim_mat(stim_index):
    length = np.max(stim_index)
    res = np.zeros(shape=(4, length))
    k, c, _ = stim_index.shape
    for ii in range(k):
        for jj in range(c):
            tmp_idx = stim_index[ii][jj]
            res[ii][tmp_idx[0]: tmp_idx[1]] = 1
    resting = 1 - res[0] - res[1] - res[2] - res[3]
    res = np.concatenate([res, resting.reshape(1, -1)], axis=0)
    return res


def cal_single_delta_aic(spike, variate, gamma, d):
    n = len(variate)
    res = np.zeros(shape=(n, ))
    theta_all = fit_reshaped_mat(variate, spike, n, gamma, d)
    aic_all = cal_aic(theta_all, variate, spike, d)
    for idx in range(n):
        def ablation(index, mat):
            if index == 0:
                excluded_mat = mat[1:]
            elif index == len(mat) - 1:
                excluded_mat = mat[: -1]
            else:
                excluded_mat = np.concatenate([mat[: index], mat[index + 1:]], axis=0)
            return excluded_mat
        ablated_v = ablation(idx, variate)
        theta_tmp = fit_reshaped_mat(ablated_v, spike, n-1, gamma, d)
        aic_tmp = cal_aic(theta_tmp, ablated_v, spike, d)
        res[idx] = aic_tmp - aic_all
    return res