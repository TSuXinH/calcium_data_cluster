from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

from base_data_two_photo import *
from DTCR import DTCR, DTCRDataset, z_score, normalize, Train


if __name__ == '__main__':
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
    args.max_epoch = 100
    args.lam = .5
    args.generate_fake = True
    args.alpha = .5
    args.device = 'cuda'
    args.with_label = False
    args.ckpt_path = './rnn/test.pth'
    args.tensorboard = False

    model = DTCR(args)
    cr_recon = nn.MSELoss()
    cr_cls = nn.CrossEntropyLoss()
    # cr_kmeans =
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = DTCRDataset(f_dff, args, normalize)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
    # x, y, z = dataset.__getitem__(0)
    # w = torch.cat([x.reshape(1, -1), y.reshape(1, -1)])
    # w = w.unsqueeze(-1)  # shape: [bs, t, 1]
    # model(w)

    print('start training.')
    train = Train(model, loader, cr_recon, optimizer, args)
    loss_list = train.train()

    plt.plot(loss_list)
    plt.show(block=True)
