import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from base_data_two_photo import f_dff, trial_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, set_seed
from linear import choose_pca_component


if __name__ == '__main__':
    set_seed(16, True)
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    f_test = z_score(f_dff)
    pca = PCA(n_components=1)
    pca_res = pca.fit_transform(f_test).reshape(-1)
    index = np.argsort(pca_res)[:: -1]

    f_shown = f_dff[index]
    sns.heatmap(f_shown, annot=False, vmin=0, vmax=1, cmap='Greys')
    plt.show(block=True)
    sys.exit()

    clus_num = 3
    cluster_config['dim'] = 3
    cluster_config['title'] = 'Cluster number: {}, Visualization: {}d'.format(clus_num, cluster_config['dim'])
    kmeans = KMeans(n_clusters=clus_num, random_state=6)
    kmeans_res = kmeans.fit_predict(pca_res)
    # tsne = TSNE(n_components=cluster_config['dim'])
    # dim_rdc_res = tsne.fit_transform(pca_res)
    pca_dim_rdc = PCA(n_components=cluster_config['dim'])
    res_rdc_dim = pca_dim_rdc.fit_transform(pca_res)

    firing_curve_config['mat'] = f_test
    firing_curve_config['stim_kind'] = 'single'
    firing_curve_config['stim_index'] = trial_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = np.arange(len(f_test))
    firing_curve_config['show_id'] = True
    cluster_config['sample_config'] = firing_curve_config
    print(cluster_config)
    visualize_cluster(clus_num, res_rdc_dim, kmeans_res, cluster_config)
