import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from base_data_two_photo import f_trial1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, get_cluster_index, set_seed
from linear import choose_pca_component

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(16, True)

    sel_thr = 10
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    valid_index = [8, 23, 30, 32, 51, 53, 68, 71, 81, 83, 90, 95, 108, 129, 137, 157, 158, 166, 169, 174, 176, 185, 199, 216, 221, 231, 237, 240, 241, 242, 248, 349]
    valid_clus_res = np.zeros(shape=(len(f_selected), ))
    valid_clus_res[valid_index] = 1
    # raw_index = [13, 44, 55, 58, 91, 93, 125, 129, 157, 160, 178, 188, 212, 255, 272, 305, 306, 326, 334, 342, 345, 367, 390, 425, 431, 457, 467, 474, 476, 478, 487, 777]
    # valid_index = np.array(valid_index)
    # config = generate_firing_curve_config()
    # config['mat'] = f_selected[valid_index]
    # config['stim_kind'] = 'multi'
    # config['multi_stim_index'] = trial1_stim_index
    # visualize_firing_curves(config)
    # sys.exit()

    # n, bins, patches = plt.hist(f_test_sum[f_test_sum < 10], bins=20, rwidth=.5, align='left')
    # for i in range(len(n)):
    #     plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    # plt.tight_layout()
    # plt.title('sum less than 10')
    # plt.show(block=True)
    # sys.exit()

    f_test = normalize(f_selected)
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
    #     for item in range(2, 20):
    #         kmeans_tmp = KMeans(n_clusters=item, random_state=6)
    #         tmp_res = kmeans_tmp.fit_predict(pca_res)
    #         ss_array[item] += silhouette_score(f_test, tmp_res)
    #         ch_array[item] += calinski_harabasz_score(f_test, tmp_res)
    # ss_array = ss_array[2:] / loop
    # ch_array = ch_array[2:] / loop
    # plot_ss_ch(ss_array, ch_array, 2)
    # sys.exit()

    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()
    clus_num = 10
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
    firing_curve_config['show_part'] = 50
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = selected_index
    firing_curve_config['show_id'] = True
    cluster_config['sample_config'] = firing_curve_config
    visualize_cluster(clus_num, res_rdc_dim, kmeans_res, cluster_config)

    sys.exit()

    index_all = get_cluster_index(kmeans_res, clus_num)
    selected_index_alter = index_all[0]
    # acc = 0
    # for item in selected_index_alter:
    #     if item in valid_index:
    #         acc += 1
    # print('k means clustered', acc)
    # sys.exit()

    test_index = np.sort(np.concatenate(index_all[1:]))
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
    k = 14
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=k, metri1c='dtw')
    print('k value: {}'.format(k))
    knn.fit(f_train, f_label)
    # svm = SVC()
    # svm.fit(f_train, f_label)
    res_pred = knn.predict(f_test_knn)
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
    print('true positive: {}'.format(tp))
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

    sys.exit()
    """ Second cluster. """
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    clus_index = get_cluster_index(kmeans_res, clus_num)
    second_clus_index = np.concatenate([clus_index[1], clus_index[3]])
    f_test_sec = f_test[second_clus_index]
    print(f_test_sec.shape)
    comp_sec = choose_pca_component(f_test_sec, thr)
    print('second trial, threshold: {}, component: {}'.format(thr, comp_sec))
    pca = PCA(n_components=comp_sec)
    pca_res = pca.fit_transform(f_test_sec)

    clus_num = 5
    cluster_config['dim'] = 3
    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    kmeans = KMeans(n_clusters=clus_num)
    kmeans_res = kmeans.fit_predict(pca_res)
    # pca_dim_rdc = PCA(n_components=cluster_config['dim'])
    # res_rdc_dim = pca_dim_rdc.fit_transform(pca_res)
    res_rdc_dim = pca_res

    firing_curve_config['mat'] = f_test_sec
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = selected_index[second_clus_index]
    firing_curve_config['show_id'] = True
    cluster_config['sample_config'] = firing_curve_config
    print(cluster_config)
    visualize_cluster(clus_num, res_rdc_dim, kmeans_res, cluster_config)

