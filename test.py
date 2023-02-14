import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from suite2p.extraction import dcnv
from base_data_two_photo import trial1_stim_index, f_trial1
from utils import generate_firing_curve_config, visualize_firing_curves, set_seed, generate_spike, z_score
from sklearn.linear_model import PoissonRegressor
from sklearn.decomposition import NMF
from scipy.stats import poisson
from scipy.optimize import minimize


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


trial1_stim_box = generate_stim_mat(trial1_stim_index)

tau = 1.
fs = 10
neu_coef = 0
baseline = 'maximin'
sig_baseline = 10
win_baseline = 1
bs = 128
ops = {
    'tau': tau,
    'fs': fs,
    'neucoeff': neu_coef,
    'baseline': baseline,
    'sig_baseline': sig_baseline,
    'win_baseline': win_baseline,
    'batch_size': bs
}

Fc = dcnv.preprocess(
    F=f_trial1,
    baseline=ops['baseline'],
    sig_baseline=ops['sig_baseline'],
    win_baseline=ops['win_baseline'],
    fs=ops['fs'],
)
spks = dcnv.oasis(Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

# firing_curve_cluster_config = generate_firing_curve_config()
# firing_curve_cluster_config['mat'] = spks  # [rest_index]
# firing_curve_cluster_config['stim_kind'] = 'multi'
# firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
# firing_curve_cluster_config['show_part'] = 50
# firing_curve_cluster_config['axis'] = False
# firing_curve_cluster_config['raw_index'] = np.arange(len(spks))  # [rest_index]
# firing_curve_cluster_config['show_id'] = True
# visualize_firing_curves(firing_curve_cluster_config)

# plt.plot(spks[38], label='spike')
# plt.plot(f_trial1[13], label='F')
# plt.legend()
# plt.show(block=True)

set_seed(16, True)
# interpolated, new_index = direct_interpolation(f_trial1, trial1_stim_index, 10)
# f_trial1 = interpolated
# trial1_stim_index = new_index

sel_thr = 100
f_test_sum = np.sum(f_trial1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1[selected_index]
print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

stim_variable = generate_stim_mat(trial1_stim_index)
trial1_spike = generate_spike(f_selected, sig_baseline=5, win_baseline=1, bs=64)
trial1_test = z_score(trial1_spike)
trial1_test[trial1_test < 0] = 0

firing_curve_cluster_config = generate_firing_curve_config()
firing_curve_cluster_config['mat'] = trial1_test  # [rest_index]
firing_curve_cluster_config['stim_kind'] = 'multi'
firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
firing_curve_cluster_config['show_part'] = 50
firing_curve_cluster_config['axis'] = False
firing_curve_cluster_config['raw_index'] = np.arange(len(trial1_test))  # [rest_index]
firing_curve_cluster_config['show_id'] = True
visualize_firing_curves(firing_curve_cluster_config)

# plt.plot(trial1_test[5], label='spike')
# plt.plot(f_selected[5], label='F')
# plt.legend()
# plt.show(block=True)

trial1_f = z_score(f_selected)

trial1_binary = deepcopy(trial1_test)
trial1_binary[trial1_binary > 0] = 1

# pr = PoissonRegressor(max_iter=10000)
# pr.fit(co_variate.reshape(-1, 6), trial1_binary[0])
# res = pr.predict(co_variate.reshape(-1, 6))
# plt.plot(res)
# plt.show(block=True)
#
# plt.plot(co_variate[5])
# plt.show(block=True)
#
# delta_t = .1

import torch
from torch import nn, optim


# class PoiGLM_estimation(nn.Module):
#     def __init__(self, input_dim, delta):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, 1)
#         self.delta = delta
#
#     def forward(self, input_data, gt):  # input_data shape: [n, t], gt shape: [t, 1]
#         n, t = input_data.shape
#         linear_res = self.linear(input_data.reshape(t, -1))  # shape: [t, 1]
#         non_linear_res = torch.exp(linear_res)
#         inverse_likelihood = torch.sum(gt * non_linear_res * self.delta) - self.delta * torch.sum(non_linear_res)
#         return inverse_likelihood
# device = 'cuda'
# co_variate_tensor = torch.tensor(co_variate).reshape(6, -1)
# spike_tensor = torch.tensor(trial1_binary).reshape(-1)
# net = PoiGLM_estimation(6, .1)
# cri = nn.MSELoss()
# opt = optim.Adam(net.parameters(), lr=1e-4)
#
# max_epoch = 10
# loss_list = []
# net = net.to(device)
# for epoch in range(max_epoch):
#     co_variate_tensor = co_variate_tensor.to(device)
#     tmp = net(co_variate_tensor)
#     loss = cri(tmp, -999)
#     loss_list.append(loss.item())
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
# plt.plot(loss_list)
# plt.show(block=True)


# def make_design_matrix(stim, d=25):
#     padded_stim = np.concatenate([np.zeros(d - 1), stim])
#     T = len(stim)
#     X = np.zeros((T, d))
#     for t in range(T):
#         X[t] = padded_stim[t:t + d]
#     return X
#
#
# def neg_log_lik_lnp(theta, X, y):
#     rate = np.exp(X @ theta)
#     log_lik = y @ np.log(rate) - rate.sum()
#     return -log_lik
#
#
# def fit_lnp(stim, spikes, d=25):
#     y = spikes
#     constant = np.ones_like(y)
#     X = np.column_stack([constant, make_design_matrix(stim)])
#     x0 = np.random.normal(0, .2, d + 1)
#     res = minimize(neg_log_lik_lnp, x0, args=(X, y))
#     return res["x"]
#
#
# def predict_spike_counts_lnp(stim, spikes, theta=None, d=25):
#     y = spikes
#     constant = np.ones_like(spikes)
#     X = np.column_stack([constant, make_design_matrix(stim)])
#     if theta is None:  # Allow pre-cached weights, as fitting is slow
#         theta = fit_lnp(X, y, d)
#     yhat = np.exp(X @ theta)
#     return yhat


# co_variate = deepcopy(stim_variable)
# resting_variable = 1 - stim_variable[0] - stim_variable[1] - stim_variable[2] - stim_variable[3]
# co_variate = np.concatenate([co_variate, resting_variable.reshape(1, -1)], axis=0)
# stim = co_variate  # shape: [stim, ]
# y = trial1_f[7]
# y_spike = trial1_binary[7]
# theta = fit_lnp(stim, y_spike)
# y_pred = predict_spike_counts_lnp(stim, y_spike, theta)
# plt.plot(y_spike, label='gt')
# plt.plot(y_pred, label='pred')
# plt.show(block=True)


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


def cal_aic_theta(theta, variate, spike, d):
    return 2 * len(theta) - get_positive_likelihood(theta, variate, spike, d)


def cal_aic_num(theta, variate, spike, d):
    return 2 * len(variate) - get_positive_likelihood(theta, variate, spike, d)


co_variate = deepcopy(stim_variable)

stim = co_variate  # shape: [n, t]
gamma_ = .1
d_ = 100
index = 100
n_, t_ = stim.shape
y = trial1_f[index]
y_spike = trial1_binary[index]
index = np.where(np.sum(trial1_binary, axis=1) == 0)[0]
trial1_binary[index, 0] = 1
theta = fit_reshaped_mat(stim, y_spike, n_, gamma_, d_)
y_pred = predict_reshaped_mat(stim, y_spike, theta, d_)

plot_all(y, y_spike, y_pred, 'neuron {}'.format(index))
# aic = get_positive_likelihood(theta, co_variate, y_spike, d_)


def cal_single_delta_aic(spike, variate, gamma, d):
    n = len(variate)
    res = np.zeros(shape=(n, ))
    theta_all = fit_reshaped_mat(variate, spike, n, gamma, d)
    aic_all = cal_aic_theta(theta_all, variate, spike, d)
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
        aic_tmp = cal_aic_theta(theta_tmp, ablated_v, spike, d)
        res[idx] = aic_tmp - aic_all
    return res


# for item in range(len(trial1_binary)):
#     print(item)
#     y_spike = trial1_binary[item]
#     res = cal_single_delta_aic(y_spike, co_variate, gamma_, d_)
# # plt.plot(y_spike)
# # plt.show(block=True)

nmf = NMF(n_components=1)
coupling = nmf.fit_transform(trial1_binary[1:].reshape(728, 350))
plt.subplot(211)
plt.plot(co_variate[0], label='stimuli')
plt.legend()
plt.subplot(212)
plt.plot(coupling, label='coupling', c='g')
plt.legend()
plt.show(block=True)

plt.plot(trial1_binary[0])
plt.title('calcium event')
plt.show(block=True)
