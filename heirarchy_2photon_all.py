import sys
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from base_data_two_photo import f_dff, trial_stim_index
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
    method_key = 0

    # print('method: {}, pre-process: {}'.format(method_dict[method_key], 'z_score'))
    # sns.clustermap(f_dff, method=method_dict[method_key], col_cluster=False, cmap='mako', standard_scale=None, z_score=0)
    # plt.show()

    f_test = trans(f_dff)
    Y = pdist(f_test)
    Z = hierarchy.ward(Y)
    plt.plot(Z[:, 2])
    plt.show()

    clus_res = hierarchy.fcluster(Z, t=300, criterion='distance')
    clus_res -= 1
    clus_num = int(np.max(clus_res)) + 1
    print('cluster number: {}'.format(int(np.max(clus_res) + 1)))

    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()
    cluster_config['dim'] = 3
    # cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    # clus_method = AgglomerativeClustering(n_clusters=clus_num)
    # clus_res = clus_method.fit_predict(f_test)

    dim_rdc_method = PCA(n_components=cluster_config['dim'])
    res_rdc_dim = dim_rdc_method.fit_transform(f_test)
    firing_curve_config['mat'] = f_test
    firing_curve_config['stim_kind'] = 'single'
    firing_curve_config['stim_index'] = trial_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = np.arange(len(f_test))
    firing_curve_config['show_id'] = True
    cluster_config['single_color'] = False
    cluster_config['sample_config'] = firing_curve_config
    visualize_cluster(clus_num, res_rdc_dim, clus_res, cluster_config)
