from DeepCluster import DeepClusterNet, CustomDataset, Train
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import normalize, set_seed
from base_data_two_photo import trial1_stim_index, f_trial1


if __name__ == '__main__':
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

    pre_clus = KMeans(args.clus_num)
    clus_label = pre_clus.fit_predict(normalize(f_selected))
    trainset = CustomDataset(f_selected, clus_label, normalize)
    train_loader = DataLoader(trainset, shuffle=True, batch_size=args.bs)

    model = DeepClusterNet(args)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    print(model)

    train_proc = Train(model, train_loader, optim, crit, args)
    acc_list, loss_list = train_proc.train()

    plt.plot(acc_list, c='g', label='acc')
    plt.plot(loss_list, c='r', label='loss')
    plt.legend()
    plt.show(block=True)

