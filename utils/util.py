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
            is_sampled = True
        else:
            print(item, ': ', config[item])
    if is_sampled:
        show_config(config['sample_config'])
