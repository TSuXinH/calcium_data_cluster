import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from base_data_two_photo import f_trial1_rest1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch
from linear import choose_pca_component


if __name__ == '__main__':
    np.random.seed(16)
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    sel_thr = 10
    f_test_sum = np.sum(f_trial1_rest1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1_rest1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    # n, bins, patches = plt.hist(f_test_sum[f_test_sum < 1000], bins=20, rwidth=.5, align='left')
    # for i in range(len(n)):
    #     plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    # plt.tight_layout()
    # plt.title('sum less than 10')
    # plt.show(block=True)
    # sys.exit()

    f_test = z_score(f_selected)
    thr = .9
    component = choose_pca_component(f_test, thr)
    print('threshold: {}, component: {}'.format(thr, component))
    pca = PCA(n_components=component)
    pca_res = pca.fit_transform(f_test)

    # loop = 20
    # ss_array = np.zeros(shape=(20, ))
    # ch_array = np.zeros(shape=(20, ))
    # for i in range(loop):
    #     print(i)
    #     for item in range(3, 20):
    #         kmeans_tmp = KMeans(n_clusters=item, random_state=6)
    #         tmp_res = kmeans_tmp.fit_predict(pca_res)
    #         ss_array[item] += silhouette_score(f_test, tmp_res)
    #         ch_array[item] += calinski_harabasz_score(f_test, tmp_res)
    # ss_array = ss_array[3:] / loop
    # ch_array = ch_array[3:] / loop
    # plot_ss_ch(ss_array, ch_array, 3)
    # sys.exit()

    clus_num = 6
    cluster_config['dim'] = 3
    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    kmeans = KMeans(n_clusters=clus_num, random_state=6)
    kmeans_res = kmeans.fit_predict(pca_res)
    # tsne = TSNE(n_components=cluster_config['dim'])
    # dim_rdc_res = tsne.fit_transform(pca_res)
    pca_dim_rdc = PCA(n_components=cluster_config['dim'])
    res_rdc_dim = pca_dim_rdc.fit_transform(pca_res)

    firing_curve_config['mat'] = f_test
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = np.arange(len(f_test))
    firing_curve_config['show_id'] = True
    cluster_config['sample_config'] = firing_curve_config
    print(cluster_config)
    visualize_cluster(clus_num, res_rdc_dim, kmeans_res, cluster_config)
