import os
from os.path import join as join
import h5py
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def get_data(path):
    data = h5py.File(path, 'r')
    key = list(data.keys())[0]
    return np.array(data.get(key)).T


def visualize_firing_curve(response_tensor, start, num, stim_index):
    """
    Visualize the firing curve of neurons, distinct the stimulus time.
    response tensor shape: [neuron, whole-time stimulus]
    """
    stim_start = stim_index[:, 0]
    stim_end = stim_index[:, 1]
    for neuron in range(start, start + num):
        plt.subplot(num, 1, neuron+1-start)
        plt.plot(response_tensor[neuron])
        plt.vlines(stim_start, ymin=0, ymax=np.max(response_tensor[neuron]).item(), colors='g', linestyles='dashed')
        plt.vlines(stim_end, ymin=0, ymax=np.max(response_tensor[neuron]).item(), colors='r', linestyles='dashed')
        # todo: add gray background during the stimulus time, learn to use function `fill_between`
    plt.xlabel('time point')
    plt.suptitle('firing curve')
    plt.show(block=True)


def pca_kmeans(exp_mat, component, cluster_num):
    pca = PCA(n_components=component, whiten=True)
    pca_res = pca.fit_transform(exp_mat)
    exp_var = pca.explained_variance_ratio_

    k_means = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=1000)
    k_means_result = k_means.fit_predict(pca_res)
    return pca_res, exp_var, k_means_result


def draw_inertia_map_under_diff_clusters(pca_res):
    inertia_list = []
    for cluster in range(2, len(color_map)):
        k_means_cluster = KMeans(n_clusters=cluster, init='k-means++', max_iter=1000)
        k_means_result = k_means_cluster.fit_predict(pca_res)
        inertia_list.append(k_means_cluster.inertia_)
    plt.plot(list(range(len(inertia_list))), inertia_list)
    plt.xticks(list(range(len(inertia_list))), list(range(2, len(inertia_list) + 2)))
    plt.title('cluster inertia via number of cluster centers')
    plt.show(block=True)


def visualize_cluster_2d(cluster, res, kmeans_result, title=''):
    for item in range(cluster):
        index = np.where(kmeans_result == item)[0]
        tmp_result = res[index]
        color = color_map[item]
        plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
    if title:
        plt.title(title)
    plt.legend()
    plt.show(block=True)


def visualize_cluster_3d(cluster, res, kmeans_result, title=''):
    ax = plt.subplot(111, projection='3d')
    for item in range(cluster):
        index = np.where(kmeans_result == item)[0]
        tmp_result = res[index]
        color = color_map[item]
        ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=color, label='cluster {}'.format(item))
    if title:
        ax.set_title(title)
    plt.legend()
    plt.show(block=True)


def visualize_sampled_spikes(mat, res_kmeans, clusters, show_all=True, stim_index=None, head=0):
    for item in range(clusters):
        index = np.where(res_kmeans == item)[0]
        print(len(index))
        tmp_f = mat[index]
        if show_all:
            show_all_neurons(tmp_f, stim_index, head)
        else:
            if len(index) > 10:
                x = np.random.randint(0, len(tmp_f) - 10)
                visualize_firing_curve(tmp_f, x, 10, final_index)
            else:
                visualize_firing_curve(tmp_f, 0, len(index), final_index)


def show_all_neurons(mat, stim_index=None, head=0):
    length = len(mat)
    for idx in range(length):
        plt.subplot(length, 1, idx+1)
        plt.plot(mat[idx])
        plt.axis('off')
        if stim_index is not None:
            stim_start = stim_index[:, 0] + head
            stim_end = stim_index[:, 1] + head
            plt.vlines(stim_start, ymin=0, ymax=max(np.max(mat[idx]).item(), 1), colors='g', linestyles='dashed')
            plt.vlines(stim_end, ymin=0, ymax=max(np.max(mat[idx]).item(), 1), colors='r', linestyles='dashed')
    plt.show(block=True)


color_map = {
    0: 'y',
    1: 'b',
    2: 'c',
    3: 'g',
    4: 'k',
    5: 'm',
    6: 'r',
    7: 'pink',
}

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
# # pay attention to difference between matlab and python: the start number
start_frame = scio.loadmat(start_frame_path)['startFrame'].item() - 1
stdExt = h5py.File(stdExt_path, 'r')
cn = np.array(stdExt.get('Cn'))
center = np.array(stdExt.get('center')).T
# results = np.array(stdExt.get('results'))

""" obtain background, do visualization and judge the critical point """
ca = scio.loadmat(ca_path)
bg = ca['background'].squeeze()

""" used the light data to align neural data """
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

""" get the stimulus tensor """
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

# show_all_neurons(f_mean_stim[50: 100])

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

trial_time0 = final_index[:: 40, 0].reshape(1, -1)
trial_time1 = final_index[39:: 40, 1].reshape(1, -1)
trial_time = np.concatenate([trial_time0, trial_time1], axis=0).T

f_trial1 = f_dff[:, trial_time[0][0]: trial_time[0][1]]
f_trial_list = []
for idx in range(len(trial_time)):
    f_trial_list.append(f_dff[:, trial_time[idx][0]: trial_time[idx][1]])
    print(f_trial_list[idx].shape)
trial1_stim = final_index[: 40]

# show_all_neurons(f_trial1[: 50], trial1_stim, head=-trial1_stim[0][0])
diff = final_index[:, 1] - final_index[:, 0]
