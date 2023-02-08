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

from base_data_two_photo import f_trial1, f_trial2, f_trial3, f_trial4, f_trial5, \
    trial1_stim_index, trial2_stim_index, trial3_stim_index, trial4_stim_index, trial5_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, get_cluster_index, set_seed, cal_pearson_mat, bin_curve, generate_contrast
from utils import direct_interpolation

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(16, True)
    interpolated1, new_index = direct_interpolation(f_trial1, trial1_stim_index, 10)
    interpolated2, _ = direct_interpolation(f_trial2, trial2_stim_index, 10)
    interpolated3, _ = direct_interpolation(f_trial3, trial3_stim_index, 10)
    interpolated4, _ = direct_interpolation(f_trial4, trial4_stim_index, 10)
    interpolated5, _ = direct_interpolation(f_trial5, trial5_stim_index, 10)
    interpolated_ave = (interpolated1 + interpolated2 + interpolated3 + interpolated4 + interpolated5) / 5
    f_trial_ave = interpolated_ave

    sel_thr = 100
    f_test_sum = np.sum(f_trial_ave, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial_ave[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))
    cur_index = np.arange(len(f_trial_ave))

    f_trial1_sel = z_score(interpolated1[selected_index])
    f_trial2_sel = z_score(interpolated2[selected_index])
    f_trial3_sel = z_score(interpolated3[selected_index])
    f_trial4_sel = z_score(interpolated4[selected_index])
    f_trial5_sel = z_score(interpolated5[selected_index])

    firing_curve_cluster_config = generate_firing_curve_config()
    firing_curve_cluster_config['mat'] = z_score(f_trial1_sel)  # [rest_index]
    firing_curve_cluster_config['stim_kind'] = 'multi'
    firing_curve_cluster_config['multi_stim_index'] = new_index
    firing_curve_cluster_config['show_part'] = 50
    firing_curve_cluster_config['axis'] = False
    firing_curve_cluster_config['raw_index'] = cur_index  # [rest_index]
    firing_curve_cluster_config['show_id'] = True
    firing_curve_cluster_config['use_heatmap'] = False
    # firing_curve_cluster_config['h_clus'] = True
    # firing_curve_cluster_config['dist'] = 'euclidean'
    # firing_curve_cluster_config['method'] = 'ward'

    visualize_firing_curves(firing_curve_cluster_config)
