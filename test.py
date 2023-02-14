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


sel_thr = 100
f_test_sum = np.sum(f_trial1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1[selected_index]
print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

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
    F=f_selected,
    baseline=ops['baseline'],
    sig_baseline=ops['sig_baseline'],
    win_baseline=ops['win_baseline'],
    fs=ops['fs'],
)
spks = dcnv.oasis(Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

x = 47
plt.subplot(211)
plt.plot(f_selected[x], label='f')
plt.legend()
plt.subplot(212)
plt.plot(spks[x], label='spike')
plt.legend()
plt.show(block=True)


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

# sel_thr = 100
# f_test_sum = np.sum(f_trial1, axis=-1)
# selected_index = np.where(f_test_sum > sel_thr)[0]
# f_selected = f_trial1[selected_index]
# print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

stim_variable = generate_stim_mat(trial1_stim_index)
trial1_spike = generate_spike(f_selected, sig_baseline=5, win_baseline=1, bs=64)
trial1_test = z_score(trial1_spike)
trial1_test[trial1_test < 0] = 0

# plt.plot(trial1_test[5], label='spike')
# plt.plot(f_selected[5], label='F')
# plt.legend()
# plt.show(block=True)

trial1_f = z_score(f_selected)

trial1_binary = deepcopy(trial1_test)
trial1_binary[trial1_binary > 0] = 1

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
#
# stim = co_variate  # shape: [stim, ]
# y = trial1_f[7]
# y_spike = trial1_binary[7]
# theta = fit_lnp(stim, y_spike)
#
# y_pred = predict_spike_counts_lnp(stim, y_spike, theta)
#
# plt.plot(y_spike, label='gt')
# plt.plot(y_pred, label='pred')
# plt.show(block=True)

from current_test import fit_reshaped_mat, predict_reshaped_mat, plot_all, get_positive_likelihood, cal_single_delta_aic, negative_log_likelihood, make_stim_time_course_3d, make_padding_stim


# co_variate = deepcopy(stim_variable)
# stim = co_variate  # shape: [n, t]
# gamma_ = .01
# d_ = 10
# index = 7
# n_, t_ = stim.shape
# y = trial1_f[index]
# y_spike = trial1_binary[index]
# theta = fit_reshaped_mat(stim, y_spike, n_, gamma_, d_)
# y_pred = predict_reshaped_mat(stim, y_spike, theta, d_)
#
# plot_all(y, y_spike, y_pred, 'neuron {}'.format(index))
# # aic = get_positive_likelihood(theta, co_variate, y_spike, d_)
#
#
# to_be_v = np.concatenate([y.reshape(1, -1), y_spike.reshape(1, -1), y_pred.reshape(1, -1)], axis=0)
# result = cal_single_delta_aic(y_spike, co_variate, gamma_, d_)
#
# firing_curve_cluster_config = generate_firing_curve_config()
# firing_curve_cluster_config['mat'] = to_be_v
# firing_curve_cluster_config['stim_kind'] = 'multi'
# firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
# firing_curve_cluster_config['show_part'] = 0
# firing_curve_cluster_config['axis'] = False
# firing_curve_cluster_config['raw_index'] = np.arange(len(to_be_v))  # [rest_index]
# firing_curve_cluster_config['show_id'] = True
# visualize_firing_curves(firing_curve_cluster_config)


# tmp_index = trial1_stim_index[:, 0].reshape((4, 1, 2))
# stim = generate_stim_mat(tmp_index)[:4]
# stop = trial1_stim_index[3, 0, 1]
# index = 69
# y = trial1_f[index, : stop]
# gamma_ = .1
# d_ = 10
# n_, t_ = stim.shape
# y_spike = trial1_binary[index, : stop]
# theta = fit_reshaped_mat(stim, y_spike, n_, gamma_, d_)
# y_pred = predict_reshaped_mat(stim, y_spike, theta, d_)
# plot_all(y, y_spike, y_pred, 'neuron {}'.format(index))
#
# to_be_v = np.concatenate([y.reshape(1, -1), y_spike.reshape(1, -1), y_pred.reshape(1, -1)], axis=0)
# res, all_, partial = cal_single_delta_aic(y_spike, stim, gamma_, d_)
#
# firing_curve_cluster_config = generate_firing_curve_config()
# firing_curve_cluster_config['mat'] = to_be_v
# firing_curve_cluster_config['stim_kind'] = 'multi'
# firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
# firing_curve_cluster_config['show_part'] = 0
# firing_curve_cluster_config['axis'] = False
# firing_curve_cluster_config['raw_index'] = np.arange(len(to_be_v))  # [rest_index]
# firing_curve_cluster_config['show_id'] = True
# visualize_firing_curves(firing_curve_cluster_config)


import torch
from torch import nn, optim


tmp_index = trial1_stim_index[:, 0].reshape((4, 1, 2))
stop = trial1_stim_index[3, 0, 1]
stim = generate_stim_mat(tmp_index)
index = 69
y = trial1_f[index, : stop]
gamma_ = .1
d_ = 10
n_, t_ = stim.shape
stim_volume = make_padding_stim(stim, d_)
delta_t, n = stim_volume.shape
stim_volume = stim_volume.transpose((1, 0)).reshape((1, n, delta_t))
y_spike = trial1_binary[index, : stop]


class Fitter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(n, 1, kernel_size=d, stride=1)
        self.linear = nn.Linear(t_, t_)

    def forward(self, input_data):
        tmp = self.conv(input_data).reshape(1, -1)
        tmp = self.linear(tmp)
        return tmp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Fitter(d_).to(device)
cri = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=.01)
stim_tensor = torch.FloatTensor(stim_volume)
y_tensor = torch.FloatTensor(y_spike)
stim_tensor_gpu = stim_tensor.to(device)
y_tensor_gpu = y_tensor.to(device)

net.train()
for item in range(20):
    pred = net(stim_tensor_gpu)
    opt.zero_grad()
    loss = cri(pred, y_tensor_gpu)
    loss.backward()
    opt.step()
    print(loss.item())


net.eval()
res = net(stim_tensor_gpu).detach().cpu().numpy().reshape(-1)
plt.subplot(211)
plt.plot(y_spike)
plt.subplot(212)
plt.plot(res)
plt.show(block=True)


weight = torch.tensor()
