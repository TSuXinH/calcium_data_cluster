import numpy as np
import matplotlib.pyplot as plt

from base_data_two_photo import f_trial1, trial1_stim_index
from utils import generate_firing_curve_config, visualize_firing_curves

config = generate_firing_curve_config()
config['mat'] = f_trial1[490]
config['stim_kind'] = 'multi'
config['multi_stim_index'] = trial1_stim_index
visualize_firing_curves(config)
