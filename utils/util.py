import numpy as np


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
