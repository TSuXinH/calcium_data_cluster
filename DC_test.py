from DeepCluster import DeepClusterNet
from sklearn.decomposition import PCA
import torch
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import normalize, set_seed, generate_firing_curve_config, generate_cluster_config, visualize_cluster, visualize_firing_curves, z_score, bin_curve, Aug, generate_contrast
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

dic = {
    'stim_index': trial1_stim_index,
    'mode': ['pos', 'neg'],
    'pos_ratio': 2,
}
aug = Aug([generate_contrast], -1, **dic)
f_train = aug(f_selected)

path = './ckpt/DC_test.pth'

state = torch.load(path)
args = state['args']
model = DeepClusterNet(args)
model.load_state_dict(state['model'])
print('stop epoch: ', state['epoch'])
print('current args: ', state['args'])


args.test_clus_num = 12
f_norm = torch.FloatTensor(args.trans(f_train))

model.eval()
lat, _ = model(f_norm)
lat = lat.detach().cpu().numpy()

clus = KMeans(args.test_clus_num)
clus_res = clus.fit_predict(lat)

if args.latent_dim > 3:
    rdc = PCA(n_components=3)
    # rdc = TSNE(n_components=3)
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
