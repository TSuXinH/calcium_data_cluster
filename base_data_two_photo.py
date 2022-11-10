import os
import h5py
import numpy as np
from copy import deepcopy
from scipy import io as scio
from os.path import join as join
import matplotlib.pyplot as plt
from utils import set_seed, normalize, z_score, visualize_firing_curves, generate_firing_curve_config, generate_contrast


def get_data(path):
    """ imports the data directly. """
    data = h5py.File(path, 'r')
    key = list(data.keys())[0]
    return np.array(data.get(key)).T


basic_path = '.\\data_alter\\two_photon\\'  # windows path
basic_dir_list = list(os.listdir(basic_path))
f_path = join(basic_path, 'F.mat')
f_dff_path = join(basic_path, 'F_dff.mat')
frame_between_trial_path = join(basic_path, 'frameNumBetweenTrials.mat')
start_frame_path = join(basic_path, 'startFrame.mat')
stdExt_path = join(basic_path, 'stdExt.mat')
ca_path = join(basic_path, 'CA_background.mat')

f = get_data(f_path)
f_dff = get_data(f_dff_path)

frame_between_trial = scio.loadmat(frame_between_trial_path)['frameNumBetweenTrials'].item()
start_frame = scio.loadmat(start_frame_path)['startFrame'].item() - 1
stdExt = h5py.File(stdExt_path, 'r')
cn = np.array(stdExt.get('Cn'))
center = np.array(stdExt.get('center')).T

# make sure that all the stimuli and responses are aligned
ca = scio.loadmat(ca_path)
bg = ca['background'].squeeze()

# extract the triggers
thr = 1500
candidate = np.where(bg > thr)[0]
cand_dff = candidate[1:] - candidate[: -1]
cand_dff = np.concatenate([cand_dff, np.array([1])])
divider = np.where(cand_dff > 1)[0]
divider_plus = divider + 1
divider = np.sort(np.concatenate([np.array([0]), divider, divider_plus]))
index = candidate[divider]
index[1::2] += 1
final_index = index[: -1].reshape(-1, 2)

# extract the neural tensor
neu_num = 822
f_aligned = []
for item in final_index[:, 0]:
    f_aligned.append(f_dff[:, item: item + 9])
f_aligned = np.array(f_aligned).reshape((5, 40, neu_num, 9))
# shape: [type, trial, repeat, neuron, time]
f_tensor = np.concatenate([
    np.expand_dims(f_aligned[:, ::4], axis=0),
    np.expand_dims(f_aligned[:, 1::4], axis=0),
    np.expand_dims(f_aligned[:, 2::4], axis=0),
    np.expand_dims(f_aligned[:, 3::4], axis=0),
])
f_long_stim = []
for item in final_index[:, 0]:
    f_long_stim.append(f_dff[:, item: item + 9].T)
f_long_stim = np.array(f_long_stim).reshape(-1, neu_num)
f_long_stim = f_long_stim.T
f_mean_stim = f_long_stim.reshape(-1, 360, 5).mean(-1)

trial_stim_index = np.concatenate([final_index[:: 40, 0], final_index[39:: 40, 1]]).reshape(2, 5).T

f_trial1 = f_dff[:, final_index[0, 0]: final_index[39, 1]]
f_trial1_rest1 = f_dff[:, final_index[0, 0]: final_index[40, 0]]
tmp_stim_index1 = final_index[: 40]
stim_index = []
for idx in range(4):
    stim_index.append(tmp_stim_index1[idx:: 4].reshape(1, 10, -1))
trial1_stim_index = np.concatenate(stim_index, axis=0) - final_index[0, 0]

f_trial2 = f_dff[:, final_index[40, 0]: final_index[79, 1]]
tmp_stim_index2 = final_index[40: 80]
stim_index = []
for idx in range(4):
    stim_index.append(tmp_stim_index2[idx:: 4].reshape(1, 10, -1))
trial2_stim_index = np.concatenate(stim_index, axis=0) - final_index[40, 0]

f_trial3 = f_dff[:, final_index[80, 0]: final_index[119, 1]]
tmp_stim_index3 = final_index[80: 120]
stim_index = []
for idx in range(4):
    stim_index.append(tmp_stim_index3[idx:: 4].reshape(1, 10, -1))
trial3_stim_index = np.concatenate(stim_index, axis=0) - final_index[80, 0]

f_trial4 = f_dff[:, final_index[120, 0]: final_index[159, 1]]
tmp_stim_index4 = final_index[120: 160]
stim_index = []
for idx in range(4):
    stim_index.append(tmp_stim_index4[idx:: 4].reshape(1, 10, -1))
trial4_stim_index = np.concatenate(stim_index, axis=0) - final_index[120, 0]

f_trial5 = f_dff[:, final_index[160, 0]: final_index[199, 1]]
tmp_stim_index5 = final_index[160: 200]
stim_index = []
for idx in range(4):
    stim_index.append(tmp_stim_index5[idx:: 4].reshape(1, 10, -1))
trial5_stim_index = np.concatenate(stim_index, axis=0) - final_index[160, 0]

stim_index1 = trial1_stim_index.transpose((1, 0, 2)).reshape(-1, 2)
stim_index2 = trial2_stim_index.transpose((1, 0, 2)).reshape(-1, 2)
stim_index3 = trial3_stim_index.transpose((1, 0, 2)).reshape(-1, 2)
stim_index4 = trial4_stim_index.transpose((1, 0, 2)).reshape(-1, 2)
stim_index5 = trial5_stim_index.transpose((1, 0, 2)).reshape(-1, 2)

stim_index_kind = []
for idx in range(4):
    stim_index_kind.append(final_index[idx:: 4].reshape(1, 50, -1))
stim_index_kind = np.concatenate(stim_index_kind, axis=0)

# sel_thr = 10
# f_test_sum = np.sum(f_trial1, axis=-1)
# selected_index = np.where(f_test_sum > sel_thr)[0]
# f_selected = f_trial1[selected_index]
# print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

# tmp_selected = f_trial1[255: 257]
# tmp_cont = generate_contrast(tmp_selected, trial1_stim_index, mode=['negative_rest', 'positive_stim'])
#
# plt.plot(tmp_selected[0], c='g')
# plt.plot(tmp_cont[0], c='r')
# plt.show(block=True)


# test = f_trial1[540]
# N = len(test)
# test_fft_c = np.fft.fftshift(np.fft.fft(test, N))
# test_fft = np.abs(test_fft_c)
# plt.plot(test, c='g')
# plt.plot(test_fft, c='r')
# plt.show(block=True)


# from utils import normalize, z_score, cal_pearson_mat, bin_curve, down_up_sample

# f1 = f_selected[68]
# f1_resample = down_up_sample(f1, 2)
# plt.plot(f1, 'g')
# plt.plot(f1_resample, 'r')
# plt.show(block=True)

# plt.plot(f1)
# plt.plot(f2)
# plt.show(block=True)
#
# pearson_mat = cal_pearson_mat(f_selected)
# index = np.where(pearson_mat > .8)[0]
# pair_list = []
# for idx in range(len(pearson_mat)):
#     if len(np.where(idx == index)[0]) == 1:
#         continue
#     else:
#         pair_list.append(idx)
# max_count = 1
# hash_arr = np.zeros(shape=(len(pearson_mat),))
# for item in pair_list:
#     if hash_arr[item] > 0:
#         continue
#     else:
#         if item == 104:
#             print(1)
#         tmp_pear = pearson_mat[item]
#         index = np.where(tmp_pear > .8)[0]
#         hash_arr[index] = max_count
#         if max_count == 12:
#             print(item, index)
#         max_count += 1
#
# x = np.random.randint(1, max_count)
# lucky_choice = np.random.choice(np.where(x == hash_arr)[0], 2, replace=False)
# f1 = f_selected[lucky_choice[0]]
# f2 = f_selected[lucky_choice[1]]
# print(lucky_choice)
# plt.plot(f1)
# plt.plot(f2)
# plt.show(block=True)

# count = 0
# for i in range(len(pearson_mat)):
#     for j in range(i):
#         if pearson_mat[i][j] > .8:
#             print(i, j)
#             count += 1

# plt.plot(f_selected[102])
# plt.plot(f_selected[343])
# plt.show(block=True)
# x = np.where(pearson_mat > .8)[0]
# a = []
# for i in range(len(f_selected)):
#     if len(np.where(x == i)[0]) == 1:
#         a.append(i)
# print(a)

# f_trial1_binned = bin_curve(f_trial1, stim_index=trial1_stim_index)
# fig, ax = plt.subplots(2, 1)
# x = 478

# ax[0].plot(f_trial1_binned[x])
# ax[1].plot(f_trial1[x])
# plt.show(block=True)

# config = generate_firing_curve_config()
# config['stim_kind'] = 'single_stim'
# config['stim_index'] = trial_stim_index
# visualize_firing_curves(f_dff[: 50], config)

# config1 = generate_firing_curve_config()
# config1['multi_stim_index'] = stim_index
# config1['stim_kind'] = 'multi_stim'
# x, y = visualize_firing_curves(f_trial1[: 50], config1)

# trial_time0 = final_index[:: 40, 0].reshape(1, -1)
# trial_time1 = final_index[39:: 40, 1].reshape(1, -1)
# trial_time = np.concatenate([trial_time0, trial_time1], axis=0).T

# f_trial1 = f_dff[:, trial_time[0][0]: trial_time[0][1]]
# f_trial_list = []
# for idx in range(len(trial_time)):
#     f_trial_list.append(f_dff[:, trial_time[idx][0]: trial_time[idx][1]])
# trial1_stim = final_index[: 40]

# import torch
# z = torch.zeros([32, 9330, 1])
# z_ = z.reshape(32, 1, 9330)
# layer = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=4)
# z__ = layer(z_)
# z__ = z__.reshape(32, -1, 16)
# rnn = torch.nn.GRU(input_size=16, hidden_size=128, bidirectional=True, num_layers=2, batch_first=True)
# hidden = torch.zeros(2*2, 32, 128)
# out_r, h_r = rnn(z__, hidden)
# h_feat = torch.cat((h_r[-2], h_r[-1]), dim=-1)
# rnn1 = torch.nn.GRU(input_size=256, hidden_size=16, bidirectional=True, num_layers=2, batch_first=True)
# hidden1 = torch.zeros(2*2, 32, 16)
# out_r1, h_r1 = rnn1(out_r, hidden1)
# out_r1 = out_r1.reshape(32, 32, -1)
# layer1 = torch.nn.ConvTranspose1d(in_channels=16*2, out_channels=1, kernel_size=6, stride=4)
# z___ = layer1(out_r1)
# z___ = z___.reshape(32, -1, 1)
