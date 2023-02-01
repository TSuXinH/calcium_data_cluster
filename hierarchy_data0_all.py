import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from base_data_two_photo import f_dff, trial_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, \
    visualize_firing_curves, z_score, normalize, plot_ss_ch, set_seed
from linear import choose_pca_component
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    set_seed(16, True)
    f_test = z_score(f_dff)
    pca = PCA(n_components=1)
    pca_res = pca.fit_transform(f_test).reshape(-1)
    index = np.argsort(pca_res)[:: -1]

    f_shown = f_dff[index]
    cluster_config = generate_cluster_config()
    firing_curve_config = generate_firing_curve_config()

    firing_curve_config['mat'] = f_shown  # [rest_index]
    firing_curve_config['stim_kind'] = 'single'
    firing_curve_config['stim_index'] = trial_stim_index
    firing_curve_config['show_part'] = 0
    firing_curve_config['axis'] = False
    firing_curve_config['raw_index'] = index  # [rest_index]
    firing_curve_config['show_id'] = True
    firing_curve_config['use_heatmap'] = True
    firing_curve_config['h_clus'] = True
    firing_curve_config['method'] = 'ward'
    visualize_firing_curves(firing_curve_config)
