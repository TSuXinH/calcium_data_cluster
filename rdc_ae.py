import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

from base_data_two_photo import trial1_stim_index, f_trial1
from autoencoder import CustomDataset, train, test, AETest1, cal_err, AutoEncoder
from utils import Aug, z_score, normalize, visualize_firing_curves, visualize_cluster, generate_firing_curve_config, generate_cluster_config, set_seed, generate_contrast


def show_reconstruction(f_dff, recon, idx, trans):
    raw = f_dff[idx]
    raw = trans(raw)
    plt.plot(raw, 'g', label='raw')
    plt.plot(recon[idx], 'r', label='recovered')
    plt.legend()
    plt.show(block=True)


set_seed(16, True)

sel_thr = 10
f_test_sum = np.sum(f_trial1, axis=-1)
selected_index = np.where(f_test_sum > sel_thr)[0]
f_selected = f_trial1[selected_index]
print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

args = edict
# network structure
args.trans = z_score
dic = {
    'stim_index': trial1_stim_index,
    'mode': ['pos', 'neg'],
    'pos_ratio': 5,
}
aug = Aug([generate_contrast], -1, **dic)
f_train = aug(f_selected)

args.aug = None
args.input_dim = f_train.shape[-1]
args.latent_dim = 3
args.hidden_dim = [256, 64]
args.drop_ratio = .15
# network hyperparameters
args.lr = 1e-4
args.wd = 5e-4
args.max_epoch = 1000
args.batch_size = 128
# other configurations
args.device = 'cuda'
args.tb_path = ''
args.pred_dim = 2
args.contrast = False
args.save_path = './ckpt/test_lae.pth'

model = AETest1(args).to(args.device)
criterion = nn.MSELoss()
crit2 = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
dataset = CustomDataset(f_train, args)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

print(args.input_dim)
print('training')
loss_list = train(model, loader, criterion, optimizer, args, crit2)
print('training done')
plt.plot(loss_list)
plt.title('non-linear autoencoder loss-epoch')
plt.show(block=True)
print('testing')
recon_res, rdc_res = test(model, test_loader, args)
print('testing done')

x = 20
show_reconstruction(f_train, recon_res, x, trans=args.trans)
f_pro = args.trans(f_train)
print(cal_err(recon_res, f_pro))

clusters = 15
clus_method = KMeans(n_clusters=clusters)
res_clus = clus_method.fit_predict(rdc_res)


if rdc_res.shape[-1] <= 3:
    res_rdc_dim = rdc_res
else:
    dim_rdc = PCA(n_components=3)
    res_rdc_dim = dim_rdc.fit_transform(rdc_res)

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
visualize_cluster(clusters, res_rdc_dim, res_clus, cluster_config)
