import numpy as np
from copy import deepcopy

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from suite2p.extraction import dcnv


class Aug:
    def __init__(self, trans, p=0, **kwargs):
        self.trans = trans
        self.kwargs = kwargs
        self.p = p

    def __call__(self, x):
        for t in self.trans:
            if np.random.rand() < self.p:
                continue
            if t == generate_contrast:
                if 'stim_index' not in self.kwargs or 'pos_ratio' not in self.kwargs or 'mode' not in self.kwargs:
                    raise ValueError('Important key missed. ')
                else:
                    x = t(x, self.kwargs['stim_index'], self.kwargs['mode'], self.kwargs['pos_ratio'])
            else:
                x = t(x)
        return x


def set_seed(seed, do_print=False):
    np.random.seed(seed)
    if do_print:
        print('set seed: {}'.format(seed))


def z_score(data, z_type='mean'):
    if z_type == 'mean':
        if len(data.shape) == 1:
            return (data - np.mean(data)) / np.std(data) if np.std(data).item() != 0 else data
        else:
            divider = np.std(data, axis=-1).reshape(-1, 1)
            divider[divider == 0] = 1
            return (data - np.mean(data, axis=-1).reshape(-1, 1)) / divider
    elif z_type == 'median':
        # todo: finish the improved median z score.
        raise NotImplementedError
    else:
        raise NotImplementedError


def normalize(data):
    if len(data.shape) == 1:
        res = (data - np.min(data)) / (np.max(data) - np.min(data)) if np.max(data) != np.min(data) else data
    else:
        divider = (np.max(data, axis=-1) - np.min(data, axis=-1)).reshape(-1, 1)
        divider[divider == 0] = 1
        res = (data - np.min(data, axis=-1).reshape(-1, 1)) / divider
    return res


def show_config(config):
    is_sampled = False
    for item in config.keys():
        if item == 'mat':
            print('mat :  ', config[item].shape)
        elif item == 'sample_config':
            is_sampled = True if config['sample_config'] is not None else False
        else:
            print(item, ': ', config[item])
    if is_sampled:
        show_config(config['sample_config'])


def get_cluster_index(clus_res, clus_num):
    clus_len_list = []
    res = []
    for item in range(clus_num):
        clus_len_list.append(len(np.where(clus_res == item)[0]))
    sort_index = np.argsort(clus_len_list)
    print('sort index', sort_index)
    for item in sort_index:
        res.append(np.where(clus_res == item)[0])
    return res


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


def cal_pearson_mat(mat):
    res = np.zeros(shape=(len(mat), len(mat)))
    for i in range(len(mat)):
        for j in range(len(mat)):
            res[i][j] = np.corrcoef(mat[i], mat[j])[0][1]
    return res


def cal_distance_mat(mat):
    res = np.zeros(shape=(len(mat), len(mat)))
    for i in range(len(mat)):
        for j in range(len(mat)):
            res[i][j] = np.sqrt(np.sum((mat[i] - mat[j]) ** 2))
    return res


def down_up_sample(mat, rate):
    raw_len = mat.shape[-1]
    length = raw_len - raw_len % rate
    mat = mat[:length].reshape(-1, rate).mean(axis=-1)
    mat = np.interp(np.arange(raw_len), np.arange(mat.shape[-1]), mat)
    return mat


def flip(mat):
    """ Mat is default normalized """
    return 1 - mat


def generate_contrast(mat, stim_index, mode=None, pos_ratio=1.):
    """
    Generates contrast patterns in stimulus state and resting state for machines to better extract the features.
    mat: shape: [N, T]
    stim_index: shape: [k, n, 2]
    """
    if mode is None:
        mode = ['add']
    res = deepcopy(mat)
    if len(res.shape) == 1:
        res = res.reshape(1, -1)
    stim_kind = len(stim_index)
    transposed_stim_index = stim_index.transpose((1, 0, 2)).reshape(-1, 2)
    # todo: simplify the for loop.
    for idx in range(len(transposed_stim_index)):
        if mode is not None:
            if 'add' in mode:
                tmp = idx % stim_kind
                res[:, transposed_stim_index[idx][0]: transposed_stim_index[idx][1]] += pos_ratio * (tmp + 1) / stim_kind
                if idx != len(transposed_stim_index)-1:
                    res[:, transposed_stim_index[idx][1]: transposed_stim_index[idx + 1][0]] += -1 / stim_kind * pos_ratio
            if 'neg' in mode:
                if idx != len(transposed_stim_index) - 1:
                    res[:, transposed_stim_index[idx][1]: transposed_stim_index[idx + 1][0]] *= -1
            if 'pos' in mode:
                res[:, transposed_stim_index[idx][0]: transposed_stim_index[idx][1]] *= 1 * pos_ratio
        else:
            raise NotImplementedError
    return res if len(res) > 1 else res.reshape(-1)


def direct_interpolation(mat, stim_index, standard_len):
    k, t, b = stim_index.shape
    res = np.zeros(shape=(len(mat), (k * t * 2 - 1) * standard_len))
    raw_index = np.transpose(stim_index, (1, 0, 2)).reshape(-1, 2)
    res_index = np.arange(0, k * t * standard_len * 2, standard_len).reshape((t, k, 2)).transpose(1, 0, 2)
    for idx in range(len(raw_index)):
        if idx != len(raw_index) - 1:
            if raw_index[idx][1] - raw_index[idx][0] == standard_len - 1:
                res[:, 2 * idx * standard_len: (2 * idx + 1) * standard_len - 1] = mat[:, raw_index[idx][0]: raw_index[idx][1]]
                res[:, (2 * idx + 1) * standard_len - 1] = (mat[:, raw_index[idx][1]] + mat[:, raw_index[idx][1]-1]) / 2
            else:
                res[:, 2 * idx * standard_len: (2 * idx + 1) * standard_len] = mat[:, raw_index[idx][0]: raw_index[idx][1]]
            if raw_index[idx + 1][0] - raw_index[idx][1] == standard_len - 1:
                res[:, (2 * idx + 1) * standard_len: (2 * idx + 2) * standard_len - 1] = mat[:, raw_index[idx][1]: raw_index[idx + 1][0]]
                res[:, (2 * idx + 2) * standard_len - 1] = (mat[:, raw_index[idx+1][0]] + mat[:, raw_index[idx+1][0]-1]) / 2
            else:
                res[:, (2 * idx + 1) * standard_len: (2*idx+2) * standard_len] = mat[:, raw_index[idx][1]: raw_index[idx+1][0]]
        else:
            if raw_index[idx][1] - raw_index[idx][0] == standard_len - 1:
                res[:, 2 * idx * standard_len: (2 * idx + 1) * standard_len - 1] = mat[:, raw_index[idx][0]: raw_index[idx][1]]
                res[:, -1] = mat[:, -1]
            else:
                res[:, 2*idx*standard_len: (2*idx+1)*standard_len] = mat[:, raw_index[idx][0]: raw_index[idx][1]]
    return res, res_index


def cal_mean_corr(*interpolations):
    length = len(interpolations)
    res_len = length * (length - 1) / 2
    res = np.zeros(shape=(len(interpolations[0]), res_len))
    for ii in range(length):
        for jj in range(ii+1, length):
            tmp = np.corrcoef(interpolations[ii], interpolations[jj], rowvar=True)

            # todo: finish it
            pass


def generate_spike(f_mat, tau=1., fs=10, neu_coef=0, baseline='maximin', sig_baseline=10, win_baseline=1, bs=128):
    ops = {
        'tau': tau,
        'fs': fs,
        'neucoeff': neu_coef,
        'baseline': baseline,
        'sig_baseline': sig_baseline,
        'win_baseline': win_baseline,
        'batch_size': bs
    }
    Fc = dcnv.preprocess(
        F=f_mat,
        baseline=ops['baseline'],
        sig_baseline=ops['sig_baseline'],
        win_baseline=ops['win_baseline'],
        fs=ops['fs'],
    )
    res = dcnv.oasis(Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
    return res
