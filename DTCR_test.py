import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

from utils import visualize_sampled_spikes, visualize_2d_cluster
from DTCR import DTCR, z_score, test, normalize
from DTCR_train import DTCRDataset
from base_data_two_photo import *


def show_reconstruction(f_dff, recon, idx, trans):
    raw = f_dff[idx]
    raw = trans(raw)
    plt.plot(raw, 'g')
    plt.plot(recon[idx], 'r')
    plt.show(block=True)


""" This dict is used for recording all the parameters of the network and training process. """
args = EasyDict()
args.input_size = f_dff.shape[-1]
args.hidden_size = 64
args.conv_ker = 9
args.conv_stride = 8
args.up_sample_chn = 16
args.conv_size = int((args.input_size - args.conv_ker) / args.conv_stride) + 1
# conv_size1 = int((args.input_size - args.conv_ker) / args.conv_stride)
# conv_size2 = int((conv_size1 - args.conv_ker) / args.conv_stride)
# args.conv_size = conv_size2
args.num_rnn_layer = 2
args.bidirectional = True
args.use_cls = False
args.dropout = .3
args.cls_hid_size = 64
args.lr = 1e-3
args.bs = 64
args.max_epoch = 250
args.lam = .5
args.generate_fake = False
args.alpha = .5
args.device = 'cuda'
args.with_label = False
args.ckpt_path = './rnn/test.pth'
args.tensorboard = False

test_dataset = DTCRDataset(f_dff, args, normalize)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

state_dict = torch.load(args.ckpt_path)
model = DTCR(args)
model.load_state_dict(state_dict['model'])
print(state_dict['epoch'])

recon, lat = test(model, test_loader, args)

x = np.random.randint(0, len(f_dff))
show_reconstruction(f_dff, recon, x, trans=normalize)

model_gmm = GMM(4, max_iter=int(1e3))
res_gmm = model_gmm.fit_predict(lat)


model_tsne = TSNE(n_components=2)
res_tsne = model_tsne.fit_transform(lat)

visualize_2d_cluster(4, res_tsne, res_gmm)
visualize_sampled_spikes(f_dff, res_gmm, 4)
plt.scatter(res_tsne[:, 0], res_tsne[:, 1])
plt.show()


x = np.random.randint(0, len(f_dff))
piece = f_dff[x]
plt.subplot(311)
plt.plot(piece)
plt.subplot(312)
plt.plot(z_score(piece))
plt.subplot(313)
plt.plot(normalize(piece))
plt.show()
