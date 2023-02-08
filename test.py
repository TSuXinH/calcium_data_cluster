import numpy as np
import h5py
import scipy.io as scio
import matplotlib.pyplot as plt
from utils import visualize_firing_curves, generate_firing_curve_config, z_score

src_path = './data_alter/alter/bb001L-1.mat'
file = h5py.File(src_path)

f_df_ref = file['CaA0/F_dF'][:][0]  # shape: [the number of trials] 454
cell_type_0 = file['CaA0/celltype'][:].reshape(-1)
cell_type_1 = file['CaA1/celltype'][:].reshape(-1)
cell_len_0 = len(cell_type_0)
cell_len_1 = len(cell_type_1)
trial_info_0 = file['CaA0/trial_info'][:].reshape(-1)  # shape: [the number of trials] 454 within the trial
trial_info_1 = file['CaA1/trial_info'][:].reshape(-1)  # should be the same as previous as the observation is the same
print(trial_info_0.shape)

t0 = file[f_df_ref[0]][:].reshape(cell_len_0, -1)  # shape: [cell_len, time]
ti0 = file[trial_info_0[0]]

trials = file['trials']
time_stamp_ref = trials['time_stamp'][:][0]
print(time_stamp_ref.shape)

ts0 = file[time_stamp_ref[0]][:]

for keys in trials.keys():
    print(keys, trials[keys])
summary = file['summary']


# x = z_score(t0)
# selected_index = np.arange(len(x))
# firing_curve_cluster_config = generate_firing_curve_config()
# firing_curve_cluster_config['mat'] = x  # [rest_index]
# firing_curve_cluster_config['stim_kind'] = 'without'
# # firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
# firing_curve_cluster_config['show_part'] = 50
# firing_curve_cluster_config['axis'] = False
# firing_curve_cluster_config['raw_index'] = selected_index  # [rest_index]
# firing_curve_cluster_config['show_id'] = True
# # firing_curve_cluster_config['use_heatmap'] = True
# # firing_curve_cluster_config['h_clus'] = True
# # firing_curve_cluster_config['dist'] = 'euclidean'
# # firing_curve_cluster_config['method'] = 'ward'
# visualize_firing_curves(firing_curve_cluster_config)
#
# " ['FOV', 'F_dF', 'ROIs', 'cellid', 'celltype', 'deconv', 'sampling_rate', 'trial_info']"

index = [1, 2, 3, 4, 5, 7, 8]
dict_record = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}
cell_type_dict = {
    0: 'undecoded',
    1: 'Agmat',
    2: 'Baz1a',
    3: 'Adamts2',
    4: 'Pvalb',
    5: 'Sst',
    6: 'Vip'
}

for item in index:
    src_path1 = 'data_alter/alter/bb001L-{}.mat'.format(item)
    src = h5py.File(src_path1)
    print(src.keys())
    cell_type = src['CaA0/celltype'][:][0]
    cell_type_alter = src['CaA1/celltype'][:][0]
    print(len(cell_type), len(cell_type_alter))
    for key in dict_record.keys():
        print(key)
        dict_record[key] += np.sum(cell_type == key)
        dict_record[key] += np.sum(cell_type_alter == key)
    print(np.max(cell_type))
for key in dict_record.keys():
    print(cell_type_dict[key], dict_record[key])


for item in index:  # loop in different sessions
    src_path1 = 'data_alter/alter/bb001L-{}.mat'.format(item)
    src = h5py.File(src_path1)
    cell_type0 = src['CaA0/celltype'][:][0]
    cell_type1 = src['CaA1/celltype'][:][0]
    print('caa0')
    print(cell_type0.shape)
    print(src['CaA0/F_dF'][:][0].shape)
    print('caa1')
    print(cell_type1.shape)
    print(src['CaA1/F_dF'][:][0].shape)
    print('\n')
