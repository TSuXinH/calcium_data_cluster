import sys
from tqdm import tqdm
from copy import deepcopy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from base_data_two_photo import f_trial1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, get_cluster_index, set_seed, cal_pearson_mat, bin_curve, generate_contrast
from utils import direct_interpolation, generate_stim_mat, generate_spike
from utils import generate_stim_mat, cal_neuron_mat_nmf, cal_single_delta_aic

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(16, True)
    # interpolated, new_index = direct_interpolation(f_trial1, trial1_stim_index, 10)
    # f_trial1 = interpolated
    # trial1_stim_index = new_index

    sel_thr = 100
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    spike_thr = 0
    trial1_f = z_score(f_selected)
    stim_variable = generate_stim_mat(trial1_stim_index)
    trial1_spike = generate_spike(f_trial1)
    trial1_test = z_score(trial1_spike)
    trial1_binary = deepcopy(trial1_test)
    trial1_binary[trial1_binary < spike_thr] = 0
    trial1_binary[trial1_binary > spike_thr] = 1
    # This step is to make up for the zeros.
    zero_index = np.where(np.sum(trial1_binary, axis=1) == 0)[0]
    trial1_makeup = deepcopy(trial1_binary)
    trial1_makeup[zero_index, 0] = 1

    gamma = .1
    d = 80
    bg_variate = generate_stim_mat(trial1_stim_index)
    n_bg = len(bg_variate)
    new_feature_mat = np.zeros(shape=(len(trial1_f), n_bg))

    for idx in tqdm(range(len(trial1_f))):
        # print('current idx: {}'.format(idx))
        # internal_variable = cal_neuron_mat_nmf(trial1_binary, idx)
        # co_variate = np.concatenate([bg_variate, trial1_binary.reshape(1, -1)], axis=0)
        new_feature_mat[idx] = cal_single_delta_aic(trial1_binary[idx], bg_variate, gamma, d)

    np.save('./new_feature.npy', new_feature_mat)

    new_feature_mat = np.load('./new_feature.npy')
    new_feature_mat[np.isnan(new_feature_mat)] = 0

    # f_test = trial1_f[index]
    # firing_curve_cluster_config = generate_firing_curve_config()
    # firing_curve_cluster_config['mat'] = f_test  # [rest_index]
    # firing_curve_cluster_config['stim_kind'] = 'multi'
    # firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
    # firing_curve_cluster_config['show_part'] = 0
    # firing_curve_cluster_config['axis'] = False
    # firing_curve_cluster_config['raw_index'] = index  # [rest_index]
    # firing_curve_cluster_config['show_id'] = True
    # firing_curve_cluster_config['use_heatmap'] = True
    # firing_curve_cluster_config['h_clus'] = True
    # firing_curve_cluster_config['dist'] = 'euclidean'
    # firing_curve_cluster_config['method'] = 'ward'
    # visualize_firing_curves(firing_curve_cluster_config)
    # sys.exit()

    clus_num = 7
    kmeans = KMeans(n_clusters=7)
    clus_res = kmeans.fit_predict(new_feature_mat)

    pca = PCA(n_components=3)
    dim_rdc_res = pca.fit_transform(trial1_f)

    cluster_config = generate_cluster_config()
    firing_curve_cluster_config = generate_firing_curve_config()
    firing_curve_cluster_config['mat'] = trial1_f  # [rest_index]
    firing_curve_cluster_config['stim_kind'] = 'multi'
    firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
    firing_curve_cluster_config['show_part'] = 0
    firing_curve_cluster_config['axis'] = False
    firing_curve_cluster_config['raw_index'] = selected_index  # [rest_index]
    firing_curve_cluster_config['show_id'] = True
    firing_curve_cluster_config['use_heatmap'] = True
    firing_curve_cluster_config['h_clus'] = True
    firing_curve_cluster_config['dist'] = 'euclidean'
    firing_curve_cluster_config['method'] = 'ward'

    cluster_config['dim'] = 3
    cluster_config['sample_config'] = firing_curve_cluster_config
    visualize_cluster(clus_num=clus_num, dim_rdc_res=dim_rdc_res, clus_res=clus_res, config=cluster_config)

