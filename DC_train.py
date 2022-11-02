from DeepCluster import DeepClusterNet, CustomDataset, Train
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import warnings
warnings.filterwarnings('ignore')

from utils import normalize, set_seed, z_score
from base_data_two_photo import trial1_stim_index, f_trial1


def bin_curve(mat, stim_index):
    res = np.zeros(shape=(len(mat), stim_index.shape[0] * stim_index.shape[1] * 2 - 1))
    new_index = stim_index.transpose((1, 0, 2)).reshape(-1, 2)
    for idx in range(len(new_index)):
        if idx == len(new_index)-1:
            res[:, 2*idx-1] = np.mean(mat[:, new_index[idx][0]: new_index[idx][1]], axis=-1)
        else:
            res[:, 2*idx-1] = np.mean(mat[:, new_index[idx][0]: new_index[idx][1]], axis=-1)
            res[:, 2*idx] = np.mean(mat[:, new_index[idx][1]: new_index[idx+1][0]], axis=-1)
    return res


if __name__ == '__main__':
    set_seed(16)

    sel_thr = 10
    f_test_sum = np.sum(f_trial1, axis=-1)
    selected_index = np.where(f_test_sum > sel_thr)[0]
    f_selected = f_trial1[selected_index]
    print('selected threshold: {}, selected index length: {}'.format(sel_thr, len(selected_index)))
    f_selected_binned = bin_curve(f_selected, trial1_stim_index)
    f_selected_cat = np.concatenate([f_selected, f_selected_binned], axis=-1)
    print('cat size', f_selected.shape)

    f_train = f_selected

    args = EasyDict()
    args.mode = 'conv'
    args.conv_ker = 11
    args.stride = 9
    args.input_dim = f_train.shape[-1]
    args.hidden_dim = 256
    args.latent_dim = 80
    args.clus_num = 15
    args.fixed_epoch = 100
    args.bs = 128
    args.lr = 1e-3
    args.max_epoch = 1000
    args.device = 'cuda'
    args.trans = z_score
    args.path = './ckpt/DC_test.pth'
    args.final_path = './ckpt/final_test.pth'
    args.test_clus_num = 10

    pre_clus = KMeans(args.clus_num)
    clus_label = pre_clus.fit_predict(args.trans(f_train))
    trainset = CustomDataset(f_train, clus_label, args.trans)
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
