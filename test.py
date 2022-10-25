import numpy as np
import matplotlib.pyplot as plt

from base_data_two_photo import f_trial1, trial1_stim_index, f_dff, trial_stim_index
from utils import generate_firing_curve_config, visualize_firing_curves

config = generate_firing_curve_config()
index = range(198, 208)
config['mat'] = f_dff[index]
config['stim_kind'] = 'single'
config['stim_index'] = trial_stim_index
visualize_firing_curves(config)
