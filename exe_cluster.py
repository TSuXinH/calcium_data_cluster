import os
import h5py
import numpy as np
from copy import deepcopy
from scipy import io as scio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class customDataset(Dataset):
    def __init__(self, dataset):
        super(customDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        return torch.tensor(self.dataset[item]).float()

    def __len__(self):
        return len(self.dataset)


class autoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, inner_dim):
        super(autoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(.1, inplace=True),
            nn.Linear(hidden_dim, inner_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            nn.LeakyReLU(.1, inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, input_data):
        encoded = self.encoder(input_data)
        return self.decoder(encoded)

    def get_inner_vec(self, input_data):
        return self.encoder(input_data)


def split_data_via_stim(data, angle_num):
    """
    data shape: [k * n, neuron number, time dimension]
    k is the kind of visual stimuli(number of different stimuli)
    result shape: [k, n, neuron number, time dimension]
    """
    result = []
    for angle in range(angle_num):
        tmp = list(range(angle, len(data), angle_num))
        result.append(data[tmp])
    result = np.array(result)
    return result


# data extraction
path_result = './3_visual_4angle/all_infered_results_filtered.mat'
path_trials = './3_visual_4angle/4_class_angle_mark_40trials.mat'
inferred_result = h5py.File(path_result, 'r')
trials = scio.loadmat(path_trials)

angle_kind = 4
stimulus_time = 40
break_interval = 10
period = .1
neuron_num = 8700
evt17 = trials['EVT17']  # RUSH mark, freq: 10Hz, cover all the experiment.
evt19 = trials['EVT19']  # visual stimulus angle mark, used to describe the stimulus.

keys = inferred_result.keys()
valid_c = np.array(inferred_result['valid_C']).T
spike = np.array(inferred_result['valid_S']).T

bins = 10  # each bin covers 1 second
# data_binned = divide_data_with_bins(data_array, bins)

# This splits the data apart, leveraging processed data.
spike_division_list = []
for item in evt19:
    tmp_idx = np.where(evt17 > item)[0][0]
    spike_division_list.append(spike[:, tmp_idx: tmp_idx + stimulus_time])
spike_stimulus = np.array(spike_division_list)  # shape: [160, 8700, 40]

list_spike_division_with_break = []
for item in evt19:
    tmp_idx = np.where(evt17 > item)[0][0]
    list_spike_division_with_break.append(spike[:, tmp_idx: tmp_idx + stimulus_time + break_interval])
array_spike_stimulus_with_break = np.array(list_spike_division_with_break)

spike_angle = split_data_via_stim(spike_stimulus, angle_kind)  # shape: [4, 40, 8700, 40]
spike_stimulus_cat = np.transpose(spike_stimulus, (1, 0, 2)).reshape(neuron_num, -1)  # shape: [8700, 6400]
spike_stimulus_break_cat = np.transpose(array_spike_stimulus_with_break, (1, 0, 2)).reshape(neuron_num, -1)

# pca = PCA(n_components=3)
# spike_stimulus_cat_pca = pca.fit_transform(spike_stimulus_cat)
# ax = plt.subplot(111, projection='3d')
# ax.scatter(spike_stimulus_cat_pca[:, 0], spike_stimulus_cat_pca[:, 1], spike_stimulus_cat_pca[:, 2])
# plt.show(block=True)

spike_stimulus_cat_first = spike_stimulus_cat[:, : angle_kind * stimulus_time]

pca = PCA(n_components=3)
spike_stimulus_cat_first_pca = pca.fit_transform(spike_stimulus_cat_first)
ax = plt.subplot(111, projection='3d')
ax.scatter(spike_stimulus_cat_first_pca[:, 0], spike_stimulus_cat_first_pca[:, 1],
           spike_stimulus_cat_first_pca[:, 2])
ax.set_title('pca dimension reduction')
plt.show(block=True)

tsne = TSNE(n_components=3)
spike_stimulus_cat_first_tsne = tsne.fit_transform(spike_stimulus_cat_first)
ax = plt.subplot(111, projection='3d')
ax.scatter(spike_stimulus_cat_first_tsne[:, 0], spike_stimulus_cat_first_tsne[:, 1],
           spike_stimulus_cat_first_tsne[:, 2])
ax.set_title('tsne dimension reduction')
plt.show(block=True)


""" auto encoder """
input_dim = spike.shape[-1]
hidden_dim = 2048
inner_dim = 128
max_epoch = 50
lr = .05
bs = 64

dataset = customDataset(spike)
train_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False)
seq2seq = autoEncoder(input_dim, hidden_dim, inner_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(params=seq2seq.parameters(), lr=lr)

total_loss = []
for epoch in range(max_epoch):
    current_loss = .0
    for idx, data in enumerate(train_loader):
        reconstructed = seq2seq(data)
        loss = criterion(reconstructed, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss += loss.item() * data.shape[0]
    total_loss.append(current_loss / len(spike))
    print('epoch {}, current loss {:.4f}'.format(epoch, current_loss / len(spike)))
plt.plot(list(range(len(total_loss))), total_loss)
plt.show(block=True)


""" raw data from the start of experiment to the end """
# pca + k-means
start_point = np.where(evt17 > evt19[0])[0][0]
end_point = np.where(evt17 > evt19[-1] + 5.)[0][0]
array_exp = spike[:, start_point: end_point]
pca = PCA(n_components=2)
array_exp_pca = pca.fit_transform(array_exp)

k_means = KMeans(n_clusters=4)
result = k_means.fit_transform(array_exp_pca)

