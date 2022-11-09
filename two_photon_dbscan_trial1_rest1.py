import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from base_data_two_photo import f_trial1_rest1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, show_config, set_seed, get_cluster_index
from linear import choose_pca_component


if __name__ == '__main__':
    set_seed(16, True)
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    sel_thr = 10
    f_test_sum = np.sum(f_trial1_rest1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1_rest1[selected_index]

    raw_index = [13, 44, 55, 58, 91, 93, 125, 129, 157, 160, 178, 188, 212, 255, 272, 305, 306, 326, 334, 342, 345, 367, 390, 425, 431, 457, 467, 474, 476, 478, 487, 777]
    raw_index = np.array(raw_index)
    valid_index = [13, 37, 46, 49, 75, 77, 107, 111, 137, 139, 153, 160, 182, 218, 235, 263, 264, 277, 283, 289, 292, 311, 330, 359, 364, 382, 391, 397, 398, 400, 409, 602]
    valid_index = np.array(valid_index)
    # for item in raw_index:
    #     valid_index.append(np.where(selected_index == item)[0].item())
    # print(valid_index)
    # valid_index = np.array(valid_index)
    # firing_curve_config = generate_firing_curve_config()
    # firing_curve_config['mat'] = f_selected[valid_index]
    # print(firing_curve_config['mat'].shape)
    # firing_curve_config['stim_kind'] = 'multi'
    # firing_curve_config['multi_stim_index'] = trial1_stim_index
    # firing_curve_config['show_part'] = 0
    # firing_curve_config['axis'] = False
    # firing_curve_config['raw_index'] = raw_index
    # firing_curve_config['show_id'] = True
    # visualize_firing_curves(firing_curve_config)
    # sys.exit()
    print('selected threshold: {}, selected neuron numbers: {}'.format(sel_thr, len(selected_index)))

    # n, bins, patches = plt.hist(f_test_sum[f_test_sum < 1000], bins=20, rwidth=.5, align='left')
    # for i in range(len(n)):
    #     plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    # plt.tight_layout()
    # plt.title('sum less than 10')
    # plt.show(block=True)
    # sys.exit()

    f_test = z_score(f_selected)
    print('current test shape: {}'.format(f_test.shape))
    thr = .9
    component = choose_pca_component(f_test, thr)
    print('summed variance threshold: {}, selected component via pca: {}'.format(thr, component))
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

    cluster_config['dim'] = 3
    # z score, eps 30; normalize, eps
    dbscan = DBSCAN(eps=32, min_samples=5)
    dbscan_res = dbscan.fit_predict(pca_res)
    print(dbscan_res)
    reserved_index = np.where(dbscan_res >= 0)[0]
    f_reserved = f_test[dbscan_res >= 0]
    dbscan_res = dbscan_res[dbscan_res >= 0]
    print('rest length: {}'.format(len(dbscan_res)))
    # tsne = TSNE(n_components=cluster_config['dim'])
    # dim_rdc_res = tsne.fit_transform(pca_res)
    pca_dim_rdc = PCA(n_components=cluster_config['dim'])
    res_rdc_dim = pca_dim_rdc.fit_transform(pca_res)
    clus_num = np.max(dbscan_res) + 1

    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    firing_curve_config['mat'] = f_reserved
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    res_index = np.arange(len(f_trial1_rest1))[selected_index]
    res_index = res_index[reserved_index]
    firing_curve_config['raw_index'] = res_index
    firing_curve_config['show_id'] = True
    cluster_config['sample_config'] = firing_curve_config
    show_config(cluster_config)
    visualize_cluster(clus_num, res_rdc_dim, dbscan_res, cluster_config)

    index_all = get_cluster_index(dbscan_res, clus_num)
    index_sum = 0
    for item in index_all:
        index_sum += len(item)
    print(index_sum, len(res_index))

    count = 0
    for i in res_index:
        if i in raw_index:
            count += 1
    print(count)


