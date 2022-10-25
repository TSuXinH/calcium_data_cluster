import numpy as np

from base_data_two_photo import f_trial1_rest1, trial1_stim_index
from utils import generate_cluster_config, generate_firing_curve_config, visualize_cluster, visualize_firing_curves, z_score, normalize, plot_ss_ch, show_config, set_seed, get_cluster_index


cluster_config = generate_cluster_config()
firing_curve_config = generate_firing_curve_config()

sel_thr = 10
f_test_sum = np.sum(f_trial1_rest1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1_rest1[selected_index]
print('selected threshold: {}, selected neuron numbers: {}'.format(sel_thr, len(selected_index)))

firing_curve_config['mat'] = f_selected
firing_curve_config['stim_kind'] = 'multi'
firing_curve_config['multi_stim_index'] = trial1_stim_index
firing_curve_config['show_part'] = 20
firing_curve_config['axis'] = True
firing_curve_config['raw_index'] = selected_index
firing_curve_config['show_id'] = True

visualize_firing_curves(firing_curve_config)


