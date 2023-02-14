import h5py
import numpy as np
from utils import generate_firing_curve_config, visualize_firing_curves


path = './data_alter/alter/bb001L-1.mat'
file = h5py.File(path)
c0 = file['CaA0']


# firing_curve_cluster_config = generate_firing_curve_config()
# firing_curve_cluster_config['mat'] = f_df0  # [rest_index]
# firing_curve_cluster_config['stim_kind'] = 'without'
# # firing_curve_cluster_config['multi_stim_index'] = trial1_stim_index
# firing_curve_cluster_config['show_part'] = 50
# firing_curve_cluster_config['axis'] = False
# firing_curve_cluster_config['raw_index'] = np.arange(cell_len)  # [rest_index]
# firing_curve_cluster_config['show_id'] = True
# # firing_curve_cluster_config['use_heatmap'] = True
# # firing_curve_cluster_config['h_clus'] = True
# # firing_curve_cluster_config['dist'] = 'euclidean'
# # firing_curve_cluster_config['method'] = 'ward'
# visualize_firing_curves(firing_curve_cluster_config)


idx_list = [1, 2, 3, 4, 5, 7, 8]
for idx in idx_list:
    print('current idx {}'.format(idx))
    tmp_path = 'data_alter/alter/bb001L-{}.mat'.format(idx)
    tmp_file = h5py.File(tmp_path)
    c0 = tmp_file['CaA0']
    c1 = tmp_file['CaA1']
    cell_len = len(c0['celltype'][:].reshape(-1))
    print(cell_len)
    print(c0.keys())
    print(c0['FOV'][:])
    # f_df0 = c0[c0['F_dF'][:][0][0]][:].reshape(cell_len, -1)

for item in c0['F_dF'][:][0]:
    print(file[item][:].shape)

for item in c0['trial_info'][:][0]:
    print(file[item].keys())
    break

x = 1
f_all = c0['F_dF'][:][0]  # shape: 454
fdfx = c0[f_all[x]]
ti0 = file[c0['trial_info'][:][0][0]]
fl = ti0['fileloc'][:]
mf = ti0['mat_file'][:]
mm = ti0['motion_metric'][:]
ts_ = ti0['time_stamp'][:]


from util_for_data1 import fetch_trial_info
res = fetch_trial_info(c0)
for idx in range(len(res)):
    print(res[idx][2].shape)

licks = file['licks']
lv = licks['lick_vector'][:][0]
ts = licks['time_stamp'][:][0]
t = licks['trial'][:][0]  # trivial


for item in ts:
    print(file[item][:])

summary = file['summary']
lo0 = summary['CaA0_leave_out'][:]
lo1 = summary['CaA1_leave_out'][:]
bd_ref = summary['bhv_dir'][:][0][0]
bd = file[bd_ref][:]
imgd = summary['img_dir'][:]
table = summary['table'][:].transpose(1, 0)

trials = file['trials']


tt0 = table[0]
for item in tt0:
    print(file[item])

print(licks[ts[0]], )