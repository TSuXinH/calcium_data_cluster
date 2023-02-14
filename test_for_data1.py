import h5py
import numpy as np
from utils import generate_firing_curve_config, visualize_firing_curves


path = './data_alter/alter/bb001L-1.mat'
file = h5py.File(path)
Caa0 = file['CaA0']

cell_len = len(Caa0['celltype'][:].reshape(-1))
f_df0 = Caa0[Caa0['F_dF'][:][0][0]][:].reshape(cell_len, -1)

firing_curve_cluster_config = generate_firing_curve_config()
firing_curve_cluster_config['mat'] = f_df0  # [rest_index]
firing_curve_cluster_config['stim_kind'] = 'without'
# firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
firing_curve_cluster_config['show_part'] = 50
firing_curve_cluster_config['axis'] = False
firing_curve_cluster_config['raw_index'] = np.arange(cell_len)  # [rest_index]
firing_curve_cluster_config['show_id'] = True
# firing_curve_cluster_config['use_heatmap'] = True
# firing_curve_cluster_config['h_clus'] = True
# firing_curve_cluster_config['dist'] = 'euclidean'
# firing_curve_cluster_config['method'] = 'ward'
visualize_firing_curves(firing_curve_cluster_config)
