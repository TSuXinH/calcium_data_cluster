import numpy as np
from copy import deepcopy
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.collections as collections


# this dictionary is useful for visualization
color_map = {
    0: 'C0',
    1: 'green',
    2: 'blue',
    3: 'purple',
    4: 'navy',
    5: 'black',
    6: 'maroon',
    7: 'orange',
    8: 'purple',
    9: 'navy',
    10: 'red',
}


def generate_firing_curve_config():
    firing_curve_config = {
        'mat': 0,
        'axis': False,
        'stim_kind': '',
        'title': '',
        'color': 'C0',  # matplotlib plot default color.
        'stim_index': 0,
        'multi_stim_index': 0,
        'single_stim_color': color_map[10],
        'color_map': color_map,
        'trans': None,
        'show_part': 0,
        'last_thr': 10,
        'alpha': .4,
        'line_width': 1
    }
    return firing_curve_config


def generate_cluster_config():
    cluster_config = {
        'dim': 2,
        'title': '',
        'color_map': color_map,
        'sample_config': None,
        's': 8
    }
    return cluster_config


def visualize_firing_curves(config):
    """
    Visualize neuron firing curves:
    `config` is a dictionary which contains:
        `mat`, `axis`, `stim_kind`, `title`, `color`, `stim_index`, `multi_stim_index`,
        `single_stim_color`, `color_map`, 'trans', `show_part`, `last_thr`
    `stim_index` should contain a matrix whose shape is [N, 2].
    `multi_stim_index` should contain a matrix whose shape is [k, N, 2], where k is the stimulus kind.
    """
    def split(_len, _piece_len):
        _index = np.arange(_len)
        loop = int(_len / _piece_len) + 1
        last = _len % _piece_len
        loop_num = loop if last > config['last_thr'] else loop-1
        for idx in range(loop_num):
            if idx == loop-1:
                yield _index[_piece_len * idx:]
            else:
                yield _index[_piece_len * idx: _piece_len * (idx + 1)]

    def plot(_config):
        if _config['stim_kind'] == 'without':
            visualize_firing_curves_wo_stim(_config)
        elif _config['stim_kind'] == 'single':
            visualize_firing_curves_single_stim(_config)
        elif _config['stim_kind'] == 'multi':
            return visualize_firing_curves_multi_stim(_config)
        else:
            raise NotImplementedError('There is no such stimulus. ')

    assert config['show_part'] == 0 or config['show_part'] >= 50, 'No need of `show_part`. '
    if len(config['mat'].shape) == 1:
        config['mat'] = config['mat'].reshape(1, -1)
    if config['show_part'] == 0 or len(config['mat']) < config['show_part']:
        plot(config)
    else:
        tmp_config = deepcopy(config)
        for index in split(len(config['mat']), config['show_part']):
            tmp_config['mat'] = config['mat'][index]
            plot(tmp_config)


def visualize_firing_curves_wo_stim(config):
    """ Visualize the firing curve of all the neurons without stimuli. """
    length = len(config['mat'])
    for idx in range(length):
        plt.subplot(length, 1, idx + 1)
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        plt.plot(piece, color=config['color'], linewidth=config['line_width'])
        plt.axis(config['axis'])
    plt.title(config['title'], fontsize='large')
    plt.show(block=True)


def visualize_firing_curves_single_stim(config):
    """ Visualize the firing curve of all the neurons with single stimulus. """
    stim_index = config['stim_index']
    t = np.arange(config['mat'].shape[-1])
    stim_fake = np.zeros(shape=(config['mat'].shape[-1],))
    fig, ax = plt.subplots(len(config['mat']), 1)
    if config['axis'] is False:
        plt.subplots_adjust(hspace=-.1)
    for idx in range(len(stim_index)):
        stim_fake[stim_index[idx][0]: stim_index[idx][1]] = 1
    for idx in range(len(config['mat'])):
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        ax[idx].plot(piece, color=config['color'], linewidth=config['line_width'])
        collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0), ymax=max(np.max(piece), 1), where=stim_fake > 0, facecolor=config['single_stim_color'], alpha=config['alpha'])
        ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'], fontsize='large')
    plt.show(block=True)


def visualize_firing_curves_multi_stim(config):
    stim_index = config['multi_stim_index']
    stim_indicator = np.ones(config['mat'].shape[-1]) * -1
    t = np.arange(config['mat'].shape[-1])
    for idx in range(len(stim_index)):
        for idx_inner in range(stim_index.shape[1]):
            stim_indicator[stim_index[idx, idx_inner, 0]: stim_index[idx, idx_inner, 1]] = idx
    fig, ax = plt.subplots(len(config['mat']), 1)
    if config['axis'] is False:
        plt.subplots_adjust(hspace=-.1)
    color_len = len(config['color_map'])
    for idx in range(len(config['mat'])):
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        ax[idx].plot(piece, color=config['color'], linewidth=config['line_width'])
        for idx_inner in range(len(stim_index)):
            collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0), ymax=max(np.max(piece), 1), where=stim_indicator == idx_inner, facecolor=config['color_map'][color_len - idx_inner - 1], alpha=config['alpha'])
            ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'])
    plt.show(block=True)


def visualize_cluster(clus_num, dim_rdc_res, clus_res, config):
    """
    Visualize cluster result:
    `config` is a dictionary which may contain:
        `dim`, `title`, `color_map`, `sample_config`
    """
    if config['dim'] == 2:
        return visualize_2d_cluster(clus_num, dim_rdc_res, clus_res, config)
    elif config['dim'] == 3:
        return visualize_3d_cluster(clus_num, dim_rdc_res, clus_res, config)
    else:
        raise NotImplementedError


def visualize_2d_cluster(clus_num, dim_rdc_res, clus_res, config):
    clus_num_array = np.zeros(shape=(clus_num, ))
    for item in range(clus_num):
        clus_num_array[item] = len(np.where(clus_res == item)[0])
    sort_idx = np.argsort(clus_num_array)
    for idx, item in enumerate(sort_idx):
        index = np.where(clus_res == item)[0]
        print('cluster num: {}'.format(len(index)))
        tmp_result = dim_rdc_res[index]
        plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=config['color_map'][idx], label='cluster {}'.format(idx), s=config['s'])
        plt.legend()
    plt.suptitle(config['title'])
    plt.show(block=True)
    if config['sample_config'] is not None:
        tmp_config = deepcopy(config['sample_config'])
        for idx, item in enumerate(sort_idx):
            index = np.where(clus_res == item)[0]
            tmp_config['mat'] = deepcopy(config['sample_config']['mat'][index])
            tmp_config['color'] = config['color_map'][idx]
            tmp_config['title'] = 'cluster: {}'.format(idx)
            visualize_firing_curves(tmp_config)


def visualize_3d_cluster(clus_num, dim_rdc_res, clus_res, config):
    clus_num_array = np.zeros(shape=(clus_num, ))
    for item in range(clus_num):
        clus_num_array[item] = len(np.where(clus_res == item)[0])
    sort_idx = np.argsort(clus_num_array)
    ax = plt.subplot(111, projection='3d')
    for idx, item in enumerate(sort_idx):
        index = np.where(clus_res == item)[0]
        print('cluster num: {}'.format(len(index)))
        tmp_result = dim_rdc_res[index]
        ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=config['color_map'][idx], label='cluster {}'.format(idx), s=config['s'])
        ax.legend()
    plt.suptitle(config['title'])
    plt.show(block=True)
    if config['sample_config'] is not None:
        tmp_config = deepcopy(config['sample_config'])
        for idx, item in enumerate(sort_idx):
            index = np.where(clus_res == item)[0]
            tmp_config['mat'] = deepcopy(config['sample_config']['mat'][index])
            tmp_config['color'] = config['color_map'][idx]
            tmp_config['title'] = 'cluster: {}'.format(idx)
            visualize_firing_curves(tmp_config)
