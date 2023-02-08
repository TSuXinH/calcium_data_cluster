import numpy as np
import h5py
import scipy.io as scio
src_path = './data_alter/alter/cc028-8.mat'
file = h5py.File(src_path)

caa0 = file['CaA0']
caa1 = file['CaA1']
licks = file['licks']
summary = file['summary']
trials = file['trials']

caa0_dict = {}
for item in caa0.keys():
    print(item)
    caa0_dict[item] = np.array(caa0[item])

f_df0 = caa0['F_dF']
tmp = caa0[f_df0[0][0]]
x = tmp[:]

from utils import visualize_firing_curves, generate_firing_curve_config, z_score
x = z_score(x)
selected_index = np.arange(len(x))
firing_curve_cluster_config = generate_firing_curve_config()
firing_curve_cluster_config['mat'] = x  # [rest_index]
firing_curve_cluster_config['stim_kind'] = 'without'
# firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
firing_curve_cluster_config['show_part'] = 50
firing_curve_cluster_config['axis'] = False
firing_curve_cluster_config['raw_index'] = selected_index  # [rest_index]
firing_curve_cluster_config['show_id'] = True
# firing_curve_cluster_config['use_heatmap'] = True
# firing_curve_cluster_config['h_clus'] = True
# firing_curve_cluster_config['dist'] = 'euclidean'
# firing_curve_cluster_config['method'] = 'ward'
visualize_firing_curves(firing_curve_cluster_config)
