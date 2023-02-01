import sys
import seaborn as sns
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from base_data_two_photo import f_trial1, trial1_stim_index, f_dff, trial_stim_index, stim_index_kind
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
    set_seed(0, True)
    sel_thr = 10
    trans = z_score

    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    f_test = trans(f_selected)

    pearson_mat = cal_pearson_mat(f_test)

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
    cluster_config['single_color'] = False
    cluster_config['sample_config'] = firing_curve_config
    visualize_cluster(clus_num, res_rdc_dim, clus_res, cluster_config)

    config_cluster_raw = generate_cluster_config()
    raw_curve_config = generate_firing_curve_config()
    raw_curve_config['mat'] = f_dff[selected_index]  # [rest_index]
    raw_curve_config['stim_kind'] = 'multi'
    raw_curve_config['multi_stim_index'] = stim_index_kind
    raw_curve_config['show_part'] = 0
    raw_curve_config['axis'] = False
    raw_curve_config['raw_index'] = selected_index  # [rest_index]
    raw_curve_config['show_id'] = True
    config_cluster_raw['single_color'] = False
    config_cluster_raw['sample_config'] = raw_curve_config
    config_cluster_raw['dim'] = 3
    visualize_cluster(clus_num, res_rdc_dim, clus_res, config_cluster_raw)
