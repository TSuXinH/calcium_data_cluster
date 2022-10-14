import os
import h5py
import numpy as np
from copy import deepcopy
from scipy import io as scio
from os.path import join as join

from plot import visualize_firing_curves, config_dict


def get_data(path):
    """ get the data directly. """
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

config = deepcopy(config_dict)
config['stim_kind'] = 'single_stim'
config['stim_index'] = trial_stim_index
visualize_firing_curves(f_dff[: 50], config)

# trial_time0 = final_index[:: 40, 0].reshape(1, -1)
# trial_time1 = final_index[39:: 40, 1].reshape(1, -1)
# trial_time = np.concatenate([trial_time0, trial_time1], axis=0).T
#
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
