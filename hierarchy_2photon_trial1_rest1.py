import sys
import seaborn as sns
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from base_data_two_photo import f_trial1_rest1, trial1_stim_index, f_dff
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, get_cluster_index, set_seed, cal_pearson_mat, bin_curve, generate_contrast
from linear import choose_pca_component

warnings.filterwarnings('ignore')
method_dict = {
    0: 'average',
    1: 'single',
    2: 'complete',
    3: 'ward',
}

if __name__ == '__main__':
    set_seed(16, True)
    sel_thr = 10
    trans = z_score
    method_key = 3

    f_test_sum = np.sum(f_trial1_rest1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1_rest1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    # f_selected_binned = bin_curve(f_selected, trial1_stim_index)
    # f_selected_cat = np.concatenate([f_selected, f_selected_binned], axis=-1)
    # f_selected_fft = np.fft.fftshift(np.fft.fft(f_selected))
    # f_selected_fft = np.abs(f_selected_fft)
    # f_selected_fft = normalize(f_selected_fft)
    # f_selected_cat1 = np.concatenate([normalize(f_selected), f_selected_fft], axis=-1)

    print('method: {}, pre-process: {}'.format(method_dict[method_key], 'z_score'))
    sns.clustermap(f_selected, method=method_dict[method_key], col_cluster=False, cmap='mako', standard_scale=None, z_score=0)
    plt.show()

    f_test = trans(f_selected)
    Y = pdist(f_test)
    Z = hierarchy.ward(Y)

    plt.plot(Z[:, 2])
    plt.show()

    clus_res = hierarchy.fcluster(Z, t=100, criterion='distance')
    clus_num = int(np.max(clus_res)) + 1
    print('cluster number: {}'.format(int(np.max(clus_res) + 1)))

    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()
    cluster_config['dim'] = 3
    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    clus_method = AgglomerativeClustering(n_clusters=clus_num)
    clus_res = clus_method.fit_predict(f_test)

    dim_rdc_method = TSNE(n_components=cluster_config['dim'])
    res_rdc_dim = dim_rdc_method.fit_transform(f_test)
    firing_curve_config['mat'] = f_selected  # [rest_index]
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = selected_index  # [rest_index]
    firing_curve_config['show_id'] = True
    cluster_config['single_color'] = True
    cluster_config['sample_config'] = firing_curve_config
    visualize_cluster(clus_num, res_rdc_dim, clus_res, cluster_config)
