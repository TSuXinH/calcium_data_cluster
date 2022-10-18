import sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from base_data_two_photo import f_trial1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize


def choose_component(mat, thr):
    assert 0 < thr <= 1, 'the threshold should be between 0 and 1.'
    pca = PCA(n_components=np.min(mat.shape).item())
    _ = pca.fit_transform(mat)
    var_ratio = pca.explained_variance_ratio_
    var_ratio_cumsum = np.cumsum(var_ratio)
    return np.where(var_ratio_cumsum > thr)[0][0] + 1


if __name__ == '__main__':
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    f_test = z_score(f_trial1)
    thr = .9
    component = choose_component(f_test, thr)
    print('threshold: {}, component: {}'.format(thr, component))

    clus_num = 5
    cluster_config['dim'] = 3
    pca = PCA(n_components=component)
    pca_res = pca.fit_transform(f_test)
    kmeans = KMeans(n_clusters=clus_num)
    kmeans_res = kmeans.fit_predict(pca_res)
    tsne = TSNE(n_components=cluster_config['dim'])
    tsne_res = tsne.fit_transform(pca_res)

    firing_curve_config['mat'] = f_trial1
    firing_curve_config['stim_kind'] = 'multi'
    firing_curve_config['multi_stim_index'] = trial1_stim_index
    firing_curve_config['show_part'] = 0
    cluster_config['sample_config'] = firing_curve_config
    print(cluster_config)
    visualize_cluster(clus_num, tsne_res, kmeans_res, cluster_config)
