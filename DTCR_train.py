from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

from base_data_two_photo import f_trial1, trial1_stim_index
from utils import visualize_cluster, visualize_firing_curves, generate_firing_curve_config, generate_cluster_config, set_seed, normalize
from DTCR import DTCR, DTCRDataset, Train
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    """ This dict is used for recording all the parameters of the network and training process. """
    set_seed(16, True)

    args = EasyDict()
    args.input_size = f_trial1.shape[-1]
    args.hidden_size = 64
    args.conv_ker = 9
    args.conv_stride = 8
    args.up_sample_chn = 16
    args.conv_size = int((args.input_size - args.conv_ker) / args.conv_stride) + 1
    args.deconv_ker = args.input_size - (args.conv_size - 1) * args.conv_stride

    args.num_rnn_layer = 2
    args.bidirectional = True
    args.use_cls = False
    args.dropout = .3
    args.cls_hid_size = 64
    args.lr = 1e-3
    args.bs = 64
    args.max_epoch = 1000
    args.lam = .5
    args.generate_fake = False
    args.alpha = .5
    args.device = 'cuda'
    args.with_label = False
    args.ckpt_path = './ckpt/test_rnn.pth'
    args.tensorboard = False

    model = DTCR(args)
    cr_recon = nn.MSELoss()
    cr_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    sel_thr = 10
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))

    dataset = DTCRDataset(f_selected, args, normalize)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    print('start training.')
    train = Train(model, loader, cr_recon, optimizer, args)
    loss_list = train.train()

    plt.plot(loss_list)
    plt.show(block=True)
