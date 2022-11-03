from DeepCluster import DeepClusterNet, CustomDataset, Train
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import normalize, set_seed, generate_firing_curve_config, generate_cluster_config, visualize_cluster, visualize_firing_curves, z_score, bin_curve
from base_data_two_photo import trial1_stim_index, f_trial1


set_seed(16)

sel_thr = 10
f_test_sum = np.sum(f_trial1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1[selected_index]
print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))
f_selected_binned = bin_curve(f_selected, trial1_stim_index)
f_selected_cat = np.concatenate([f_selected, f_selected_binned], axis=-1)
f_selected_fft = np.fft.fftshift(np.fft.fft(f_selected))
f_selected_fft = np.abs(f_selected_fft)
f_selected_fft = normalize(f_selected_fft)
f_selected_cat1 = np.concatenate([normalize(f_selected), f_selected_fft], axis=-1)
f_selected_cat_all = np.concatenate([normalize(f_selected), f_selected_fft, normalize(f_selected_binned)], axis=-1)

f_train = f_selected_fft

args = EasyDict()
args.mode = 'linear'
args.conv_ker = 11
args.stride = 9
args.input_dim = f_train.shape[-1]
args.hidden_dim = 256
args.latent_dim = 50
args.clus_num = 100
args.fixed_epoch = 25
args.bs = 128
args.lr = 1e-3
args.max_epoch = 500
args.device = 'cuda'
args.trans = z_score
args.path = './ckpt/DC_test.pth'
args.final_path = './ckpt/final_test.pth'
args.test_clus_num = 9

f_norm = torch.FloatTensor(args.trans(f_train))

state = torch.load(args.path)
model = DeepClusterNet(args)
model.load_state_dict(state['model'])
print('stop epoch: ', state['epoch'])

model.eval()
lat, _ = model(f_norm)
lat = lat.detach().cpu().numpy()

clus = KMeans(args.test_clus_num)
clus_res = clus.fit_predict(lat)

if args.latent_dim > 3:
    # rdc = PCA(n_components=3)
    rdc = TSNE(n_components=3)
    lat = rdc.fit_transform(lat)

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
