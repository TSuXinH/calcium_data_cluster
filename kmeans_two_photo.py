import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from base_data_two_photo import f_dff, f_trial1, trial_stim_index, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves


cluster_config = generate_cluster_config()
firing_curve_config = generate_firing_curve_config()

fig

