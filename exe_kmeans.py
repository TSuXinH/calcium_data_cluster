import os
from os.path import join as join
import h5py
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


def get_data(path):
    data = h5py.File(path, 'r')
    key = list(data.keys())[0]
    return np.array(data.get(key)).T


color_map = {
    0: 'y',
    1: 'b',
    2: 'c',
    3: 'g',
    4: 'k',
    5: 'm',
    6: 'r',
    7: 'w',
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


def firing_curve_visualization(response_tensor, start, num, stim_index):
    """
    Visualize the firing curve of neurons, distinct the stimulus time.
    response tensor shape: [neuron, whole-time stimulus]
    """
    stim_time = 740
    stim_start = stim_index[::40, 0]
    stim_end = stim_start + stim_time
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


sub_title1 = 'PCA + K means + stim only'
pca_2d = PCA(n_components=2, whiten=True)
pca_2d_res = pca_2d.fit_transform(f_long_stim)
exp_var = pca_2d.explained_variance_ratio_

# # pick the best cluster number
inertia_list = []
for cluster in range(2, len(color_map)):
    k_means_cluster = KMeans(n_clusters=cluster, init='k-means++', max_iter=1000)
    k_means_result = k_means_cluster.fit_predict(pca_2d_res)
    inertia_list.append(k_means_cluster.inertia_)
plt.plot(list(range(len(inertia_list))), inertia_list)
plt.xticks(list(range(len(inertia_list))), list(range(2, len(inertia_list) + 2)))
plt.title('cluster inertia via number of cluster centers')
plt.show(block=True)

clusters = 3
k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
k_means_result = k_means.fit_predict(pca_2d_res)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    tmp_result = pca_2d_res[index]
    color = color_map[item]
    plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
plt.title('PCA + K Means cluster')
plt.legend()
plt.show(block=True)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    print(len(index))
    tmp_f = f_dff[index]
    if len(index) > 10:
        x = np.random.randint(0, len(tmp_f)-10)
        firing_curve_visualization(tmp_f, x, 10, final_index)
    else:
        firing_curve_visualization(tmp_f, 0, len(index), final_index)


sub_title2 = 'PCA + k means + the whole record'
pca_2d = PCA(n_components=2, whiten=True)
pca_2d_res = pca_2d.fit_transform(f_dff)
exp_var = pca_2d.explained_variance_ratio_

# # pick the best cluster number
inertia_list = []
for cluster in range(2, len(color_map)):
    k_means_cluster = KMeans(n_clusters=cluster, init='k-means++', max_iter=1000)
    k_means_result = k_means_cluster.fit_predict(pca_2d_res)
    inertia_list.append(k_means_cluster.inertia_)
plt.plot(list(range(len(inertia_list))), inertia_list)
plt.xticks(list(range(len(inertia_list))), list(range(2, len(inertia_list) + 2)))
plt.title('cluster inertia via number of cluster centers')
plt.show(block=True)

clusters = 5
k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
k_means_result = k_means.fit_predict(pca_2d_res)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    tmp_result = pca_2d_res[index]
    color = color_map[item]
    plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
plt.title('PCA + K Means cluster')
plt.legend()
plt.show(block=True)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    print(len(index))
    tmp_f = f_dff[index]
    if len(index) > 10:
        x = np.random.randint(0, len(tmp_f)-10)
        firing_curve_visualization(tmp_f, x, 10, final_index)
    else:
        firing_curve_visualization(tmp_f, 0, len(index), final_index)


sub_title3 = 'PCA3d + k means + stim only'
pca_3d = PCA(n_components=3, whiten=True)
pca_3d_res = pca_3d.fit_transform(f_long_stim)
exp_var = pca_3d.explained_variance_ratio_
# # pick the best cluster number
inertia_list = []
for cluster in range(2, len(color_map)):
    k_means_cluster = KMeans(n_clusters=cluster, init='k-means++', max_iter=1000)
    k_means_result = k_means_cluster.fit_predict(pca_3d_res)
    inertia_list.append(k_means_cluster.inertia_)
plt.plot(list(range(len(inertia_list))), inertia_list)
plt.xticks(list(range(len(inertia_list))), list(range(2, len(inertia_list) + 2)))
plt.title('cluster inertia via number of cluster centers')
plt.show(block=True)

clusters = 5
k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
k_means_result = k_means.fit_predict(pca_3d_res)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    tmp_result = pca_3d_res[index]
    color = color_map[item]
    ax = plt.subplot(111, projection='3d')
    ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=color, label='cluster {}'.format(item), s=6)
plt.title('PCA + K Means cluster')
plt.legend()
plt.show(block=True)


sub_title4 = 'PCA + K Means + selected neurons'
clusters = 4
selected_index_list = []
for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    if len(index) > 5:
        selected_index_list.append(index)
selected_index_list = np.sort(np.concatenate(selected_index_list))
selected_f_long_stim = f_long_stim[selected_index_list]
pca_res, exp_var, k_means_result = pca_kmeans(selected_f_long_stim, 2, clusters)

for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    tmp_result = pca_res[index]
    color = color_map[item]
    plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
plt.title('PCA + K Means cluster')
plt.legend()
plt.show(block=True)


for item in range(clusters):
    index = np.where(k_means_result == item)[0]
    print(len(index))
    tmp_f = f_dff[index]
    if len(index) > 10:
        x = np.random.randint(0, len(tmp_f)-10)
        print(x)
        firing_curve_visualization(tmp_f, x, 10, final_index)
    else:
        firing_curve_visualization(tmp_f, 0, len(index), final_index)

