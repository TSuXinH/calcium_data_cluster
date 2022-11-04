import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from base_data_two_photo import f_trial1_rest1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, show_config, set_seed, get_cluster_index
from linear import choose_pca_component

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    set_seed(16, True)
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    sel_thr = 10
    f_test_sum = np.sum(f_trial1_rest1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1_rest1[selected_index]
    print('selected threshold: {}, selected neuron numbers: {}'.format(sel_thr, len(selected_index)))

    # raw_index = [13, 44, 55, 58, 91, 93, 125, 129, 157, 160, 178, 188, 212, 255, 272, 305, 306, 326, 334, 342, 345, 367, 390, 425, 431, 457, 467, 474, 476, 478, 487, 777]
    # raw_index = np.array(raw_index)
    # valid_index = [13, 37, 46, 49, 75, 77, 107, 111, 137, 139, 153, 160, 182, 218, 235, 263, 264, 277, 283, 289, 292, 311, 330, 359, 364, 382, 391, 397, 398, 400, 409, 602]
    # valid_index = np.array(valid_index)
    # valid_clus_res = np.zeros(shape=(len(f_selected), ))
    # valid_clus_res[valid_index] = 1

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
    # start = 2
    # ss_array = np.zeros(shape=(20, ))
    # ch_array = np.zeros(shape=(20, ))
    # for i in range(loop):
    #     print(i)
    #     for item in range(2, 20):
    #         kmeans_tmp = KMeans(n_clusters=item, random_state=6)
    #         tmp_res = kmeans_tmp.fit_predict(pca_res)
    #         ss_array[item] += silhouette_score(f_test, tmp_res)
    #         ch_array[item] += calinski_harabasz_score(f_test, tmp_res)
    # ss_array = ss_array[2:] / loop
    # ch_array = ch_array[2:] / loop
    # plot_ss_ch(ss_array, ch_array, 2)
    # sys.exit()

    clus_num = 20

    cluster_config['dim'] = 3
    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    kmeans = KMeans(n_clusters=clus_num, random_state=6)
    kmeans_res = kmeans.fit_predict(pca_res)
    # tsne = TSNE(n_components=cluster_config['dim'])
    # dim_rdc_res = tsne.fit_transform(pca_res)
    pca_dim_rdc = PCA(n_components=cluster_config['dim'])
    res_rdc_dim = pca_dim_rdc.fit_transform(pca_res)

    firing_curve_config['mat'] = f_selected
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = selected_index
    firing_curve_config['show_id'] = True
    firing_curve_config['single_color'] = True
    cluster_config['sample_config'] = firing_curve_config
    show_config(cluster_config)
    visualize_cluster(clus_num, res_rdc_dim, kmeans_res, cluster_config)
    # visualize_cluster(2, res_rdc_dim, valid_clus_res, cluster_config)
    sys.exit()

    index_all = get_cluster_index(kmeans_res, clus_num)
    selected_index_alter = index_all[3]
    # acc = 0
    # for item in selected_index_alter:
    #     if item in valid_index:
    #         acc += 1
    # print(acc)
    # sys.exit()

    test_index = index_all[0]
    for idx in range(1, len(index_all)):
        if idx != 3:
            test_index = np.concatenate([test_index, index_all[idx]])
        else:
            continue
    test_index = np.sort(test_index)
    print('selected index alter', selected_index_alter)
    print('test index', test_index)
    print('length: {}'.format(len(test_index)))
    f_train = f_test[selected_index_alter]
    f_test_knn = f_test[test_index]
    f_label = []
    for item in selected_index_alter:
        if item in valid_index:
            f_label.append(1)
        else:
            f_label.append(0)
    f_label = np.array(f_label)
    print(f_train.shape, f_label)
    f_train_bin = f_train.reshape(len(f_label), -1, 8).mean(axis=-1)
    f_test_knn_bin = f_test_knn.reshape(len(f_test_knn), -1, 8).mean(axis=-1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(f_train_bin, f_label)
    # svm = SVC()
    # svm.fit(f_train, f_label)
    res_pred = knn.predict(f_test_knn_bin)
    tp, fp, tn, fn = 0, 0, 0, 0
    pred_index = []
    for idx in range(len(res_pred)):
        if res_pred[idx] == 1:
            if test_index[idx] in valid_index:
                tp += 1
                pred_index.append(test_index[idx])
            else:
                fp += 1
        elif res_pred[idx] == 0:
            if test_index[idx] not in valid_index:
                tn += 1
                pred_index.append(test_index[idx])
            else:
                fn += 1
        else:
            continue
    pred_pos_index = []
    for idx in range(len(res_pred)):
        if res_pred[idx] == 1:
            pred_pos_index.append(test_index[idx])

    print('accuracy: {}'.format((tp + tn) / len(res_pred)))
    print('total length: {}'.format(len(res_pred)))
    print('ture positive: {}'.format(tp))
    print('predicted positive {}'.format(tp + fp))
    print('precision: {}'.format(tp / (tp + fp)))
    pred_index = np.array(pred_pos_index)
    firing_curve_config['mat'] = f_test[pred_index]
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = pred_index
    firing_curve_config['show_id'] = True
    visualize_firing_curves(firing_curve_config)
