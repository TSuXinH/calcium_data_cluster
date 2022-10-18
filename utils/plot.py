import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.collections as collections


# this dictionary is useful for visualization
color_map = {
    0: 'b',
    1: 'r',
    2: 'g',
    3: 'c',
    4: 'k',
    5: 'm',
    6: 'y',
    7: 'pink',
}


def generate_firing_curve_config():
    firing_curve_config = {
        'mat': 0,
        'axis': False,
        'stim_kind': 'wo_stim',
        'title': '',
        'stim_index': 0,
        'multi_stim_index': 0,
        'single_stim_color': color_map[1],
        'color_map': color_map,
        'trans': None,
    }
    return firing_curve_config


def generate_cluster_config():
    cluster_config = {
        'dim': 2,
        'title': '',
        'color_map': color_map,
        'sample_dict': False,
    }
    return cluster_config


def visualize_firing_curves(config):
    """
    Visualize neuron firing curves:
    `config` is a dictionary which contains:
        `mat`, `axis`, `stim_kind`, `title`, `stim_index`, `multi_stim_index`,
        `single_stim_color`, `color_map`, 'trans'
    `stim_index` should contain a matrix whose shape is [N, 2].
    `multi_stim_index` should contain a matrix whose shape is [k, N, 2], where k is the stimulus kind.
    """
    if len(config['mat'].shape) == 1:
        config['mat'] = config['mat'].reshape(1, -1)
    if 'axis' not in config:
        raise ValueError('key `axis` is not covered. ')
    if 'title' not in config:
        raise ValueError('key `title` is not covered. ')
    if 'stim_kind' in config:
        if config['stim_kind'] == 'wo_stim':
            visualize_firing_curves_wo_stim(config)
        elif config['stim_kind'] == 'single_stim':
            visualize_firing_curves_single_stim(config)
        elif config['stim_kind'] == 'multi_stim':
            return visualize_firing_curves_multi_stim(config)
        else:
            raise NotImplementedError
    else:
        raise ValueError('key `stim_kind` is not covered. ')


def visualize_firing_curves_wo_stim(config):
    """ Visualize the firing curve of all the neurons without stimuli. """
    length = len(config['mat'])
    for idx in range(length):
        plt.subplot(length, 1, idx + 1)
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        plt.plot(piece)
        plt.axis(config['axis'])
    plt.title(config['title'])
    plt.show(block=True)


def visualize_firing_curves_single_stim(config):
    """
    Visualize the firing curve of all the neurons with single stimulus.
    The configuration dictionary should at least contain `stim_index`.
    """
    if 'stim_index' not in config:
        return ValueError('key `stim_index` is not covered.')
    stim_index = config['stim_index']
    if 'shift' in config:
        stim_index += config['shift']
    t = np.arange(config['mat'].shape[-1])
    stim_fake = np.zeros(shape=(config['mat'].shape[-1],))
    fig, ax = plt.subplots(len(config['mat']), 1)
    for idx in range(len(stim_index)):
        stim_fake[stim_index[idx][0]: stim_index[idx][1]] = 1
    for idx in range(len(config['mat'])):
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        ax[idx].plot(piece)
        collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0), ymax=max(np.max(piece), 1), where=stim_fake > 0, facecolor=config['single_stim_color'], alpha=0.2)
        ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'])
    plt.show(block=True)


def visualize_firing_curves_multi_stim(config):
    if 'multi_stim_index' not in config:
        return ValueError('key `multi_stim_index` is not covered.')
    stim_index = config['multi_stim_index']
    if 'shift' in config:
        stim_index += config['shift']
    stim_indicator = np.ones(config['mat'].shape[-1]) * -1
    t = np.arange(config['mat'].shape[-1])
    for idx in range(len(stim_index)):
        for idx_inner in range(stim_index.shape[1]):
            stim_indicator[stim_index[idx, idx_inner, 0]: stim_index[idx, idx_inner, 1]] = idx
    fig, ax = plt.subplots(len(config['mat']), 1)
    for idx in range(len(config['mat'])):
        piece = config['trans'](config['mat'][idx]) if config['trans'] is not None else config['mat'][idx]
        ax[idx].plot(piece)
        for idx_inner in range(len(stim_index)):
            collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0), ymax=max(np.max(piece), 1), where=stim_indicator == idx_inner, facecolor=config['color_map'][idx_inner], alpha=0.2)
            ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'])
    plt.show(block=True)
    return stim_indicator, stim_index


def visualize_cluster(clus_num, dim_rdc_res, clus_res, config):
    """
    Visualize cluster result:
    `config` is a dictionary which may contain:
        `dim`, `title`, `color_map`
    """
    if config['dim'] == 2:
        return visualize_2d_cluster(clus_num, dim_rdc_res, clus_res, config)
    elif config['dim'] == 3:
        return visualize_3d_cluster(clus_num, dim_rdc_res, clus_res, config)
    else:
        raise NotImplementedError


def visualize_2d_cluster(clus_num, dim_rdc_res, clus_res, config):
    for item in range(clus_num):
        index = np.where(clus_res == item)[0]
        tmp_result = dim_rdc_res[index]
        color = config['color_map'][item]
        plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
        plt.title(config['title'])
    plt.legend()
    plt.show(block=True)
    if config['sample_config'] is not None:
        for item in range(clus_num):
            index = np.where(clus_res == item)[0]
            tmp_res = config['sample_config']['mat'][index]
            tmp_config = deepcopy(config['sample_config'])
            tmp_config['mat'] = tmp_res
            tmp_config['single_stim_color'] = config['color_map'][item]
            visualize_firing_curves(tmp_config)


def visualize_3d_cluster(clus_num, dim_rdc_res, clus_res, config):
    ax = plt.subplot(111, projection='3d')
    for item in range(clus_num):
        index = np.where(clus_res == item)[0]
        tmp_result = dim_rdc_res[index]
        color = config['color_map'][item]
        ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=color, label='cluster {}'.format(item))
        ax.set_title(config['title'])
    plt.legend()
    plt.show(block=True)

