import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from suite2p.extraction import dcnv
from base_data_two_photo import trial1_stim_index, f_trial1
from utils import generate_firing_curve_config, visualize_firing_curves, set_seed, generate_spike, z_score


def generate_stim_mat(stim_index):
    length = np.max(stim_index)
    res = np.zeros(shape=(4, length))
    k, c, _ = stim_index.shape
    for ii in range(k):
        for jj in range(c):
            tmp_idx = stim_index[ii][jj]
            res[ii][tmp_idx[0]: tmp_idx[1]] = 1
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

plt.plot(trial1_test[5], label='spike')
plt.plot(f_selected[5], label='F')
plt.legend()
plt.show(block=True)

trial1_f = z_score(f_selected)

trial1_binary = deepcopy(trial1_test)
trial1_binary[trial1_binary > 0] = 1


co_variate = deepcopy(stim_variable)
resting_variable = 1 - stim_variable[0] - stim_variable[1] - stim_variable[2] - stim_variable[3]
co_variate = np.concatenate([co_variate, resting_variable.reshape(1, -1)], axis=0)


from sklearn.linear_model import PoissonRegressor
from sklearn.decomposition import NMF
from scipy.stats import poisson


time_len = trial1_test.shape[-1]

a0 = trial1_test[1:].reshape(time_len, -1)
nmf = NMF(n_components=1, init='random')
a_t = nmf.fit_transform(a0)
a_t /= a_t.max()
co_variate = np.concatenate([co_variate, a_t.reshape(1, -1)], axis=0)

pr = PoissonRegressor(max_iter=10000)
pr.fit(co_variate.reshape(-1, 6), trial1_binary[0])
res = pr.predict(co_variate.reshape(-1, 6))
plt.plot(res)
plt.show(block=True)

plt.plot(co_variate[5])
plt.show(block=True)

delta_t = .1

import torch
from torch import nn, optim


class PoiGLM_estimation(nn.Module):
    def __init__(self, input_dim, delta):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.delta = delta

    def forward(self, input_data, gt):  # input_data shape: [n, t], gt shape: [t, 1]
        n, t = input_data.shape
        linear_res = self.linear(input_data.reshape(t, -1))  # shape: [t, 1]
        non_linear_res = torch.exp(linear_res)
        inverse_likelihood = torch.sum(gt * non_linear_res * self.delta) - self.delta * torch.sum(non_linear_res)
        return inverse_likelihood

device = 'cuda'
co_variate_tensor = torch.tensor(co_variate).reshape(6, -1)
spike_tensor = torch.tensor(trial1_binary).reshape(-1)
net = PoiGLM_estimation(6, .1)
cri = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=1e-4)

max_epoch = 10
loss_list = []
net = net.to(device)
for epoch in range(max_epoch):
    co_variate_tensor = co_variate_tensor.to(device)
    tmp = net(co_variate_tensor)
    loss = cri(tmp, -999)
    loss_list.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

plt.plot(loss_list)
plt.show(block=True)
