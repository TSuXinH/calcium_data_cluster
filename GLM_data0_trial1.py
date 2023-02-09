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
from utils import direct_interpolation, generate_stim_mat, generate_spike

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_seed(16, True)
    # interpolated, new_index = direct_interpolation(f_trial1, trial1_stim_index, 10)
    # f_trial1 = interpolated
    # trial1_stim_index = new_index

    sel_thr = 100
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    trial1_f = z_score(f_selected)
    stim_variable = generate_stim_mat(trial1_stim_index)
    trial1_spike = generate_spike(f_trial1)
    trial1_test = z_score(trial1_spike)
    trial1_test[trial1_test < 0] = 0
