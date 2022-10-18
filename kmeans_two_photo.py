import sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from base_data_two_photo import f_dff, f_trial1, trial_stim_index, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, \
    visualize_cluster, visualize_firing_curves, z_score, normalize


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

    f_test = z_score(f_dff)
    component = choose_component(f_test, .9)

    clus_num = 4
    pca = PCA(n_components=component)
    pca_res = pca.fit_transform(f_test)
    kmeans = KMeans(n_clusters=clus_num)
    kmeans_res = kmeans.fit_predict(pca_res)
    tsne_3d = TSNE(n_components=3)
    tsne_res_3d = tsne_3d.fit_transform(pca_res)

    cluster_config['dim'] = 3
    visualize_cluster(clus_num, tsne_res_3d, kmeans_res, cluster_config)
    firing_curve_config['stim_kind'] = 'single_stim'
    firing_curve_config['stim_index'] = trial_stim_index

    visualize_firing_curves(firing_curve_config)
