import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import warnings
import matplotlib.pyplot as plt

from utils import generate_firing_curve_config, visualize_firing_curves
warnings.filterwarnings('ignore')


trace_path = './data_alter/gonogo/all_infered_results_filtered.mat'
stim_path = './data_alter/gonogo/stimuli.csv'

trace = h5py.File(trace_path)
stim = pd.read_csv(stim_path)

f_c = np.array(trace['valid_C'])
f_c = f_c.reshape(-1, len(f_c))
evt17 = stim.get('EVT17').to_numpy()
evt19 = stim.get('EVT19')
evt19 = evt19[~np.isnan(evt19)].to_numpy()

plt.plot(evt19)
plt.show(block=True)


evt17_min, evt17_max = evt17[0], evt17[-1]
stim_index = []
for item in evt19:
    if item < evt17_min:
        continue
    elif item > evt17_max:
        continue
    else:
        start_index = np.where(item < evt17)[0][0]
        end_index = np.where(item + 2.1 < evt17)[0][0]
        stim_index.extend([start_index, end_index])
stim_index = np.array(stim_index).reshape(-1, 2)

firing_curve_config = generate_firing_curve_config()
firing_curve_config['mat'] = f_c[: 50]
firing_curve_config['stim_kind'] = 'single'
firing_curve_config['stim_index'] = stim_index
firing_curve_config['show_part'] = 0
firing_curve_config['axis'] = False
firing_curve_config['raw_index'] = np.arange(50)
firing_curve_config['show_id'] = True
visualize_firing_curves(firing_curve_config)

length = stim_index[:, 1] - stim_index[:, 0]
