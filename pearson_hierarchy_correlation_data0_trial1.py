import sys
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
from linear import choose_pca_component

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(16, True)

    sel_thr = 10
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    f_test = z_score(f_selected)
    pearson_mat = cal_pearson_mat(f_test)

    # sns.clustermap(pearson_mat, method='ward')
    # plt.show(block=True)

    Y = pdist(pearson_mat)
    Z = hierarchy.ward(Y)

    # _, axs = plt.subplots(2, 1)
    # axs[0].plot(Z[:, 2])
    # axs[1].plot(Z[:, 3])
    # plt.show(block=True)
    # sys.exit()

    clus_res = hierarchy.fcluster(Z, t=15, criterion='distance') - 1  # 15 in ward mode
    clus_num = int(np.max(clus_res)) + 1
    print('cluster number: {}'.format(clus_num))

    for idx in range(clus_num):
        print(idx, end=' ')
        print(len(np.where(clus_res == idx)[0]))
    # sys.exit()

    pca = PCA(n_components=3)
    dim_rdc_res = pca.fit_transform(f_test)

    cluster_config = generate_cluster_config()
    firing_curve_cluster_config = generate_firing_curve_config()
    firing_curve_cluster_config['mat'] = f_test  # [rest_index]
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
