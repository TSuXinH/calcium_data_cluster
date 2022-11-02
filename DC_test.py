from DeepCluster import DeepClusterNet, CustomDataset, Train
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import normalize, set_seed, generate_firing_curve_config, generate_cluster_config, visualize_cluster, visualize_firing_curves
from base_data_two_photo import trial1_stim_index, f_trial1


set_seed(16)
args = EasyDict()
args.input_dim = f_trial1.shape[-1]
args.hidden_dim = 128
args.latent_dim = 3
args.clus_num = 20
args.fixed_epoch = 25
args.bs = 64
args.lr = 1e-3
args.max_epoch = 500
args.device = 'cuda'
args.trans = normalize
args.path = './ckpt/DC_test.pth'
args.test_clus_num = args.clus_num // 2

sel_thr = 10
f_test_sum = np.sum(f_trial1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1[selected_index]
print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))
f_norm = torch.FloatTensor(normalize(f_selected))

state = torch.load(args.path)
model = DeepClusterNet(args)
model.load_state_dict(state['model'])
print(state['epoch'])

model.eval()
lat, _ = model(f_norm)
lat = lat.detach().cpu().numpy()

clus = KMeans(args.test_clus_num)
clus_res = clus.fit_predict(lat)

firing_curve_config = generate_firing_curve_config()
cluster_config = generate_cluster_config()

firing_curve_config['mat'] = f_selected
firing_curve_config['stim_kind'] = 'multi'
firing_curve_config['multi_stim_index'] = trial1_stim_index
firing_curve_config['show_part'] = 0
firing_curve_config['axis'] = False
firing_curve_config['raw_index'] = selected_index
firing_curve_config['show_id'] = True
cluster_config['sample_config'] = firing_curve_config
cluster_config['dim'] = 3
visualize_cluster(args.test_clus_num, dim_rdc_res=lat, clus_res=clus_res, config=cluster_config)
