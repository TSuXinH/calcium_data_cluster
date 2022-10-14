import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import cufflinks

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

config_dict = {
    'axis': False,
    'stim_kind': 'wo_stim',
    'title': '',
    'stim_index': 0,
    'single_stim_color': color_map[1],
    'multi_stim_color_map': color_map
}


def visualize_firing_curves(firing_mat, config, trans=None):
    if len(firing_mat.shape) == 1:
        firing_mat = firing_mat.reshape(1, -1)
    if 'axis' not in config:
        raise ValueError('key `axis` is not covered. ')
    if 'title' not in config:
        raise ValueError('key `title` is not covered. ')
    if 'stim_kind' in config_dict:
        if config['stim_kind'] == 'wo_stim':
            visualize_firing_curves_wo_stim(firing_mat, config, trans)
        elif config['stim_kind'] == 'single_stim':
            visualize_firing_curves_single_stim(firing_mat, config, trans)
        elif config['stim_kind'] == 'multi_stim':
            visualize_firing_curves_multi_stim(firing_mat, config, trans)
        else:
            raise NotImplementedError
    else:
        raise ValueError('key `stim_kind` is not covered. ')


def visualize_firing_curves_wo_stim(firing_mat, config, trans=None):
    """ Visualize the firing curve of all the neurons without stimuli. """
    length = len(firing_mat)
    for idx in range(length):
        plt.subplot(length, 1, idx + 1)
        piece = trans(firing_mat[idx]) if trans is not None else firing_mat[idx]
        plt.plot(piece)
        plt.axis(config['axis'])
    plt.title(config['title'])
    plt.show(block=True)


def visualize_firing_curves_single_stim(firing_mat, config, trans=None):
    """
    Visualize the firing curve of all the neurons with single stimulus.
    The configuration dictionary should at least contain `stim_index`.
    """
    if 'stim_index' not in config:
        return ValueError('key `stim_index` is not covered.')
    stim_index = config['stim_index']
    if 'shift' in config:
        stim_index -= config['shift']
    if len(firing_mat.shape) == 1:
        firing_mat = firing_mat.reshape(1, -1)
    t = np.arange(firing_mat.shape[-1])
    stim_fake = np.zeros(shape=(firing_mat.shape[-1],))
    fig, ax = plt.subplots(len(firing_mat), 1)
    for idx in range(len(stim_index)):
        stim_fake[stim_index[idx][0]: stim_index[idx][1]] = 1
    for idx in range(len(firing_mat)):
        piece = trans(firing_mat[idx]) if trans is not None else firing_mat[idx]
        ax[idx].plot(piece)
        collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0), ymax=max(np.max(piece), 1), where=stim_fake > 0, facecolor=config['single_stim_color'], alpha=0.2)
        ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'])
    plt.show(block=True)


def visualize_firing_curves_multi_stim(firing_mat, config, trans=None):
    pass


def visualize_2d_cluster(clus_num, res, kmeans_result, title=''):
    for item in range(clus_num):
        index = np.where(kmeans_result == item)[0]
        tmp_result = res[index]
        color = color_map[item]
        plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
    if title:
        plt.title(title)
    plt.legend()
    plt.show(block=True)


def visualize_3d_cluster(cluster, res, kmeans_result, title=''):
    ax = plt.subplot(111, projection='3d')
    for item in range(cluster):
        index = np.where(kmeans_result == item)[0]
        tmp_result = res[index]
        color = color_map[item]
        ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=color, label='cluster {}'.format(item))
    if title:
        ax.set_title(title)
    plt.legend()
    plt.show(block=True)
