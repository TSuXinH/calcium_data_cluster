import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import numpy as np
from copy import deepcopy
import warnings
import pandas as pd
import matplotlib as mpl
from seaborn.matrix import Grid, _matrix_mask, gridspec, _convert_colors, dendrogram, despine, heatmap


# this dictionary is useful for visualization
color_map = {
    0: 'C0',
    1: 'green',
    2: 'blue',
    3: 'purple',
    4: 'olive',
    5: 'black',
    6: 'maroon',
    7: 'navy',
    8: 'purple',
    9: 'orange',
    10: 'red',
    11: 'C1',
    12: 'C2',
    13: 'C3',
    14: 'C4',
    15: 'C5',
    16: 'C6',
    17: 'C7',
    18: 'C8',
    19: 'C9',
}


def generate_firing_curve_config():
    """
    mat: the input trace matrix to be shown
    axis: decide whether to show the axis, if False, the axis will not be shown, neither the coordinate numbers
    stim_kind: choose between `without`, `single` or 'multi`
    stim_index: show the stimulus background
    :return: a parameter dictionary
    """
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
        'show_part': 0,
        'alpha': .35,
        'line_width': 1.5,
        'show_id': False,
        'raw_index': 0,
        'font_size': 10,
        'use_heatmap': False,
        'h_clus': False,
        'dist': 'euclidean',
        'method': None,
        'norm': None,
    }
    return firing_curve_config


def generate_cluster_config():
    cluster_config = {
        'dim': 2,
        'title': '',
        'color_map': color_map,
        'sample_config': None,
        's': 8,
        'single_color': False,
        'font_size': 10,
        'sort': False,
    }
    return cluster_config


def visualize_firing_curves(config):
    """
    Visualize neuron firing curves:
    `config` is a dictionary which contains:
        `mat`, `axis`, `stim_kind`, `title`, `color`, `stim_index`, `multi_stim_index`,
        `single_stim_color`, `color_map`, `show_part`
    `stim_index` should contain a matrix whose shape is [N, 2].
    `multi_stim_index` should contain a matrix whose shape is [k, N, 2], where k is the stimulus kind.
    """

    def split(_len, _piece_len):
        _index = np.arange(_len)
        last = _len % _piece_len
        loop = int(_len / _piece_len) + 1 if last != 0 else int(_len / _piece_len)
        if loop > last // 5 + 1 and last != 0:
            added = loop // last
            extra = loop % last
            prev_idx = 0
            for idx in range(loop - 1):
                if idx == loop - 2:
                    yield _index[prev_idx:]
                else:
                    if idx < last:
                        yield _index[prev_idx: prev_idx + _piece_len + added]
                        prev_idx += (_piece_len + added)
                    elif idx == last:
                        yield _index[prev_idx: prev_idx + _piece_len + extra]
                        prev_idx += (_piece_len + extra)
                    else:
                        yield _index[prev_idx: prev_idx + _piece_len]
                        prev_idx += _piece_len
        else:
            for idx in range(loop):
                if idx == loop - 1:
                    yield _index[idx * _piece_len:]
                else:
                    yield _index[idx * _piece_len: (idx + 1) * _piece_len]

    def plot(_config):
        if _config['stim_kind'] == 'without':
            visualize_firing_curves_wo_stim(_config)
        elif _config['stim_kind'] == 'single':
            visualize_firing_curves_single_stim(_config)
        elif _config['stim_kind'] == 'multi':
            visualize_firing_curves_multi_stim(_config)
        else:
            raise NotImplementedError('There is no such stimulus. ')

    if len(config['mat'].shape) == 1:
        config['mat'] = config['mat'].reshape(1, -1)
    if config['show_part'] == 0 or len(config['mat']) <= config['show_part']:
        plot(config)
    else:
        tmp_config = deepcopy(config)
        for index in split(len(config['mat']), config['show_part']):
            tmp_config['mat'] = config['mat'][index]
            tmp_config['raw_index'] = config['raw_index'][index]
            plot(tmp_config)


def visualize_firing_curves_wo_stim(config):
    """ Visualize the firing curve of all the neurons without stimuli. """
    if config['use_heatmap']:
        visualize_heatmap_wo_stim(config)
        return
    length = len(config['mat'])
    fig, ax = plt.subplots(length, 1)
    for idx in range(length):
        piece = config['mat'][idx]
        ax[idx].plot(piece, color=config['color'], linewidth=config['line_width'])
        ax[idx].axis(config['axis'])
        if config['show_id']:
            ax[idx].text(config['mat'].shape[1] + 10, 0, config['raw_index'][idx], fontsize=8)
            # print(config['raw_index'][idx])
            # disp_x, disp_y = ax[idx].transAxes.transform((0, 0))
            # ax[idx].annotate(config['raw_index'][idx], (disp_x, disp_y), xycoords='figure pixels', textcoords='offset pixels')
    plt.suptitle(config['title'], fontsize=config['font_size'], y=.9)
    plt.show(block=True)


def visualize_firing_curves_single_stim(config):
    """ Visualize the firing curve of all the neurons with single stimulus. """
    if config['use_heatmap']:
        visualize_heatmap_single_stim(config)
        return
    stim_index = config['stim_index']
    t = np.arange(config['mat'].shape[-1])
    stim_fake = np.zeros(shape=(config['mat'].shape[-1],))
    fig, ax = plt.subplots(len(config['mat']), 1)
    ax = [ax] if isinstance(ax, np.ndarray) is False else ax
    if config['axis'] is False:
        plt.subplots_adjust(hspace=-.1)
    for idx in range(len(stim_index)):
        stim_fake[stim_index[idx][0]: stim_index[idx][1]] = 1
    for idx in range(len(config['mat'])):
        piece = config['mat'][idx]
        ax[idx].plot(piece, color=config['color'], linewidth=config['line_width'])
        collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0),
                                                                 ymax=max(np.max(piece), 1), where=stim_fake > 0,
                                                                 facecolor=config['single_stim_color'],
                                                                 alpha=config['alpha'])
        ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
        if config['show_id']:
            ax[idx].text(config['mat'].shape[1] + 10, 0, config['raw_index'][idx], fontsize=8)
    plt.suptitle(config['title'], fontsize=config['font_size'], y=.9)
    plt.show(block=True)


def visualize_firing_curves_multi_stim(config):
    if config['use_heatmap']:
        visualize_heatmap_multi_stim(config)
        return
    stim_index = config['multi_stim_index']
    stim_indicator = np.ones(config['mat'].shape[-1]) * -1
    t = np.arange(config['mat'].shape[-1])
    for idx in range(len(stim_index)):
        for idx_inner in range(stim_index.shape[1]):
            stim_indicator[stim_index[idx, idx_inner, 0]: stim_index[idx, idx_inner, 1]] = idx
    fig, ax = plt.subplots(len(config['mat']), 1)
    ax = [ax] if isinstance(ax, np.ndarray) is False else ax
    if config['axis'] is False:
        plt.subplots_adjust(hspace=-.1)
    color_len = len(config['color_map'])
    for idx in range(len(config['mat'])):
        piece = config['mat'][idx]
        ax[idx].plot(piece, color=config['color'], linewidth=config['line_width'])
        if config['show_id']:
            ax[idx].text(config['mat'].shape[1] + 10, 0, config['raw_index'][idx], fontsize=8)
        for idx_inner in range(len(stim_index)):
            collection = collections.BrokenBarHCollection.span_where(t, ymin=min(np.min(piece), 0),
                                                                     ymax=max(np.max(piece), 1),
                                                                     where=stim_indicator == idx_inner,
                                                                     facecolor=config['color_map'][
                                                                         color_len - idx_inner - 1],
                                                                     alpha=config['alpha'])
            ax[idx].add_collection(collection)
        ax[idx].axis(config['axis'])
    plt.suptitle(config['title'], fontsize=config['font_size'], y=.9)
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
    clus_num_array = np.zeros(shape=(clus_num,))
    for item in range(clus_num):
        clus_num_array[item] = len(np.where(clus_res == item)[0])
    if config['sort']:
        sort_idx = np.argsort(clus_num_array)
    else:
        sort_idx = np.arange(len(clus_num_array))
    for idx, item in enumerate(sort_idx):
        index = np.where(clus_res == item)[0]
        print('cluster num: {}'.format(len(index)))
        tmp_result = dim_rdc_res[index]
        if config['single_color']:
            plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=config['color_map'][0], s=config['s'])
        else:
            plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=config['color_map'][idx], label='cluster {}'.format(idx),
                        s=config['s'])
            plt.legend()
    plt.title(config['title'], fontsize=config['font_size'])
    plt.show(block=True)
    if config['sample_config'] is not None:
        tmp_config = deepcopy(config['sample_config'])
        for idx, item in enumerate(sort_idx):
            index = np.where(clus_res == item)[0]
            tmp_config['mat'] = deepcopy(config['sample_config']['mat'][index])
            tmp_config['color'] = config['color_map'][idx] if config['single_color'] is False else config['color_map'][0]
            tmp_config['title'] = 'cluster: {}, number: {}'.format(idx, len(index))
            tmp_config['raw_index'] = config['sample_config']['raw_index'][index]
            visualize_firing_curves(tmp_config)


def visualize_3d_cluster(clus_num, dim_rdc_res, clus_res, config):
    clus_num_array = np.zeros(shape=(clus_num,))
    for item in range(clus_num):
        clus_num_array[item] = len(np.where(clus_res == item)[0])
    if config['sort']:
        sort_idx = np.argsort(clus_num_array)
    else:
        sort_idx = np.arange(len(clus_num_array))
    ax = plt.subplot(111, projection='3d')
    for idx, item in enumerate(sort_idx):
        index = np.where(clus_res == item)[0]
        print('cluster num: {}'.format(len(index)))
        tmp_result = dim_rdc_res[index]
        if config['single_color']:
            ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=config['color_map'][0], s=config['s'])
        else:
            ax.scatter(tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2], c=config['color_map'][idx],
                       label='cluster {}'.format(idx), s=config['s'])
            ax.legend()
    ax.set_title(config['title'], fontsize=config['font_size'])
    plt.show(block=True)
    if config['sample_config'] is not None:
        tmp_config = deepcopy(config['sample_config'])
        for idx, item in enumerate(sort_idx):
            index = np.where(clus_res == item)[0]
            tmp_mat = deepcopy(config['sample_config']['mat'][index])
            # index_tmp_mat = np.argsort(np.sum(tmp_mat, axis=-1))
            # tmp_mat = tmp_mat[index_tmp_mat]
            tmp_config['mat'] = tmp_mat
            tmp_config['color'] = config['color_map'][idx] if config['single_color'] is False else config['color_map'][0]
            tmp_config['title'] = 'cluster: {}, number: {}'.format(idx, len(index))
            tmp_config['raw_index'] = config['sample_config']['raw_index'][index]
            visualize_firing_curves(tmp_config)


def plot_ss_ch(ss_list, ch_list, start_index):
    ax = plt.subplot(111)
    l1 = ax.plot(ss_list, label='ss score', c='r')
    ax_alter = ax.twinx()
    l2 = ax_alter.plot(ch_list, label='ch_score', c='g')
    ax.set_xticks(list(range(len(ss_list))), list(range(start_index, start_index + len(ss_list))))
    lin = l1 + l2
    labels = [_l.get_label() for _l in lin]
    ax.legend(lin, labels, loc='upper right')
    ax.set_title('silhouette and calinski harabasz score')
    plt.show(block=True)


def visualize_heatmap_single_stim(config):
    """ Visualize heatmap of all the neurons with single stimulus. """
    fig, ax = plt.subplots()
    mat = config['mat']
    if config['h_clus']:
        if config['norm'] == 'zscore':
            cluster_map(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1,
                        col_cluster=False, cmap='Greys', standard_scale=None, z_score=0, config=config)
        elif config['norm'] == 'standard':
            cluster_map(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1,
                        col_cluster=False, cmap='Greys', standard_scale=0, z_score=None, config=config)
        else:
            raise NotImplementedError
    else:
        sns.heatmap(mat, annot=False, vmin=0, vmax=1, cmap='Greys', ax=ax)
        stim_index = config['stim_index']
        for ii in range(stim_index.shape[0]):
            ax.axvspan(xmin=stim_index[ii][0], xmax=stim_index[ii][1],
                       facecolor=config['single_stim_color'], alpha=config['alpha'])
    if config['title'] is not None:
        plt.title(config['title'])
    plt.show(block=True)


def visualize_heatmap_multi_stim(config):
    """ Visualize heatmap of all the neurons with multi stimuli. """
    fig, ax = plt.subplots()
    mat = config['mat']
    if config['h_clus']:
        if config['norm'] == 'zscore':
            cluster_map(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1,
                        col_cluster=False, cmap='Greys', standard_scale=None, z_score=0, config=config)
        elif config['norm'] == 'standard':
            cluster_map(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1,
                        col_cluster=False, cmap='Greys', standard_scale=0, z_score=None, config=config)
        elif config['norm'] is None:
            cluster_map(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1,
                        col_cluster=False, cmap='Greys', standard_scale=None, z_score=None, config=config)
        else:
            raise NotImplementedError
    else:
        sns.heatmap(mat, annot=False, vmin=0, vmax=1, cmap='Greys', ax=ax)
        stim_index = config['multi_stim_index']
        color_len = len(stim_index)
        for ii in range(stim_index.shape[0]):
            for jj in range(stim_index.shape[1]):
                ax.axvspan(xmin=stim_index[ii][jj][0], xmax=stim_index[ii][jj][1],
                           facecolor=config['color_map'][color_len - ii - 1], alpha=config['alpha'])
    if config['title'] is not None:
        plt.title(config['title'])
    plt.show(block=True)


def visualize_heatmap_wo_stim(config):
    fig, ax = plt.subplots()
    mat = config['mat']
    if config['h_clus']:
        if config['norm'] == 'zscore':
            sns.clustermap(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1, ax=ax,
                           col_cluster=False, cmap='Greys', standard_scale=None, z_score=0)
        elif config['norm'] == 'standard':
            sns.clustermap(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1, ax=ax,
                           col_cluster=False, cmap='Greys', standard_scale=0, z_score=None)
        elif config['norm'] is None:
            sns.clustermap(config['mat'], method=config['method'], metric=config['dist'], vmin=0, vmax=1, ax=ax,
                           col_cluster=False, cmap='Greys', standard_scale=None, z_score=None)
        else:
            raise NotImplementedError
    else:
        sns.heatmap(mat, annot=False, vmin=np.min(mat), vmax=np.max(mat), cmap='Greys', ax=ax)
    plt.show(block=True)


def cluster_map(data, *,
                pivot_kws=None, method='average', metric='euclidean',
                z_score=None, standard_scale=None, figsize=(10, 10),
                cbar_kws=None, row_cluster=True, col_cluster=True,
                row_linkage=None, col_linkage=None,
                row_colors=None, col_colors=None, mask=None,
                dendrogram_ratio=.2, colors_ratio=0.03,
                cbar_pos=(.02, .8, .05, .18), tree_kws=None, config=None,
                **kwargs):
    plotter = ClusterGrid(data, pivot_kws=pivot_kws, figsize=figsize,
                          row_colors=row_colors, col_colors=col_colors,
                          z_score=z_score, standard_scale=standard_scale,
                          mask=mask, dendrogram_ratio=dendrogram_ratio,
                          colors_ratio=colors_ratio, cbar_pos=cbar_pos)

    return plotter.plot(metric=metric, method=method,
                        colorbar_kws=cbar_kws,
                        row_cluster=row_cluster, col_cluster=col_cluster,
                        row_linkage=row_linkage, col_linkage=col_linkage,
                        tree_kws=tree_kws, config=config, **kwargs)


class ClusterGrid(Grid):

    def __init__(self, data, pivot_kws=None, z_score=None, standard_scale=None,
                 figsize=None, row_colors=None, col_colors=None, mask=None,
                 dendrogram_ratio=None, colors_ratio=None, cbar_pos=None):
        """Grid object for organizing clustered heatmap input on to axes"""

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)

        self.data2d = self.format_data(self.data, pivot_kws, z_score,
                                       standard_scale)

        self.mask = _matrix_mask(self.data2d, mask)

        self._figure = plt.figure(figsize=figsize)

        self.row_colors, self.row_color_labels = \
            self._preprocess_colors(data, row_colors, axis=0)
        self.col_colors, self.col_color_labels = \
            self._preprocess_colors(data, col_colors, axis=1)

        try:
            row_dendrogram_ratio, col_dendrogram_ratio = dendrogram_ratio
        except TypeError:
            row_dendrogram_ratio = col_dendrogram_ratio = dendrogram_ratio

        try:
            row_colors_ratio, col_colors_ratio = colors_ratio
        except TypeError:
            row_colors_ratio = col_colors_ratio = colors_ratio

        width_ratios = self.dim_ratios(self.row_colors,
                                       row_dendrogram_ratio,
                                       row_colors_ratio)
        height_ratios = self.dim_ratios(self.col_colors,
                                        col_dendrogram_ratio,
                                        col_colors_ratio)

        nrows = 2 if self.col_colors is None else 3
        ncols = 2 if self.row_colors is None else 3

        self.gs = gridspec.GridSpec(nrows, ncols,
                                    width_ratios=width_ratios,
                                    height_ratios=height_ratios)

        self.ax_row_dendrogram = self._figure.add_subplot(self.gs[-1, 0])
        self.ax_col_dendrogram = self._figure.add_subplot(self.gs[0, -1])
        self.ax_row_dendrogram.set_axis_off()
        self.ax_col_dendrogram.set_axis_off()

        self.ax_row_colors = None
        self.ax_col_colors = None

        if self.row_colors is not None:
            self.ax_row_colors = self._figure.add_subplot(
                self.gs[-1, 1])
        if self.col_colors is not None:
            self.ax_col_colors = self._figure.add_subplot(
                self.gs[1, -1])

        self.ax_heatmap = self._figure.add_subplot(self.gs[-1, -1])
        if cbar_pos is None:
            self.ax_cbar = self.cax = None
        else:
            # Initialize the colorbar axes in the gridspec so that tight_layout
            # works. We will move it where it belongs later. This is a hack.
            self.ax_cbar = self._figure.add_subplot(self.gs[0, 0])
            self.cax = self.ax_cbar  # Backwards compatibility
        self.cbar_pos = cbar_pos

        self.dendrogram_row = None
        self.dendrogram_col = None

    def _preprocess_colors(self, data, colors, axis):
        """Preprocess {row/col}_colors to extract labels and convert colors."""
        labels = None

        if colors is not None:
            if isinstance(colors, (pd.DataFrame, pd.Series)):

                # If data is unindexed, raise
                if (not hasattr(data, "index") and axis == 0) or (
                    not hasattr(data, "columns") and axis == 1
                ):
                    axis_name = "col" if axis else "row"
                    msg = (f"{axis_name}_colors indices can't be matched with data "
                           f"indices. Provide {axis_name}_colors as a non-indexed "
                           "datatype, e.g. by using `.to_numpy()``")
                    raise TypeError(msg)

                # Ensure colors match data indices
                if axis == 0:
                    colors = colors.reindex(data.index)
                else:
                    colors = colors.reindex(data.columns)

                # Replace na's with white color
                # TODO We should set these to transparent instead
                colors = colors.astype(object).fillna('white')

                # Extract color values and labels from frame/series
                if isinstance(colors, pd.DataFrame):
                    labels = list(colors.columns)
                    colors = colors.T.values
                else:
                    if colors.name is None:
                        labels = [""]
                    else:
                        labels = [colors.name]
                    colors = colors.values

            colors = _convert_colors(colors)

        return colors, labels

    def format_data(self, data, pivot_kws, z_score=None,
                    standard_scale=None):
        """Extract variables from data or use directly."""

        # Either the data is already in 2d matrix format, or need to do a pivot
        if pivot_kws is not None:
            data2d = data.pivot(**pivot_kws)
        else:
            data2d = data

        if z_score is not None and standard_scale is not None:
            raise ValueError(
                'Cannot perform both z-scoring and standard-scaling on data')

        if z_score is not None:
            data2d = self.z_score(data2d, z_score)
        if standard_scale is not None:
            data2d = self.standard_scale(data2d, standard_scale)
        return data2d

    @staticmethod
    def z_score(data2d, axis=1):
        """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()

        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def standard_scale(data2d, axis=1):
        """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        # Normalize these values to range from 0 to 1
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
            standardized.max() - standardized.min())

        if axis == 1:
            return standardized
        else:
            return standardized.T

    def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
        """Get the proportions of the figure taken up by each axes."""
        ratios = [dendrogram_ratio]

        if colors is not None:
            # Colors are encoded as rgb, so ther is an extra dimention
            if np.ndim(colors) > 2:
                n_colors = len(colors)
            else:
                n_colors = 1

            ratios += [n_colors * colors_ratio]

        # Add the ratio for the heatmap itself
        ratios.append(1 - sum(ratios))

        return ratios

    @staticmethod
    def color_list_to_matrix_and_cmap(colors, ind, axis=0):
        """Turns a list of colors into a numpy matrix and matplotlib colormap

        These arguments can now be plotted using heatmap(matrix, cmap)
        and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        axis : int
            Which axis this is labeling

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each indexes into the cmap
        cmap : matplotlib.colors.ListedColormap

        """
        try:
            mpl.colors.to_rgb(colors[0])
        except ValueError:
            # We have a 2D color structure
            m, n = len(colors), len(colors[0])
            if not all(len(c) == n for c in colors[1:]):
                raise ValueError("Multiple side color vectors must have same size")
        else:
            # We have one vector of colors
            m, n = 1, len(colors)
            colors = [colors]

        # Map from unique colors to colormap index value
        unique_colors = {}
        matrix = np.zeros((m, n), int)
        for i, inner in enumerate(colors):
            for j, color in enumerate(inner):
                idx = unique_colors.setdefault(color, len(unique_colors))
                matrix[i, j] = idx

        # Reorder for clustering and transpose for axis
        matrix = matrix[:, ind]
        if axis == 0:
            matrix = matrix.T

        cmap = mpl.colors.ListedColormap(list(unique_colors))
        return matrix, cmap

    def plot_dendrograms(self, row_cluster, col_cluster, metric, method,
                         row_linkage, col_linkage, tree_kws):
        # Plot the row dendrogram
        if row_cluster:
            self.dendrogram_row = dendrogram(
                self.data2d, metric=metric, method=method, label=False, axis=0,
                ax=self.ax_row_dendrogram, rotate=True, linkage=row_linkage,
                tree_kws=tree_kws
            )
        else:
            self.ax_row_dendrogram.set_xticks([])
            self.ax_row_dendrogram.set_yticks([])
        # PLot the column dendrogram
        if col_cluster:
            self.dendrogram_col = dendrogram(
                self.data2d, metric=metric, method=method, label=False,
                axis=1, ax=self.ax_col_dendrogram, linkage=col_linkage,
                tree_kws=tree_kws
            )
        else:
            self.ax_col_dendrogram.set_xticks([])
            self.ax_col_dendrogram.set_yticks([])
        despine(ax=self.ax_row_dendrogram, bottom=True, left=True)
        despine(ax=self.ax_col_dendrogram, bottom=True, left=True)

    def plot_colors(self, xind, yind, **kws):
        """Plots color labels between the dendrogram and the heatmap

        Parameters
        ----------
        heatmap_kws : dict
            Keyword arguments heatmap

        """
        # Remove any custom colormap and centering
        # TODO this code has consistently caused problems when we
        # have missed kwargs that need to be excluded that it might
        # be better to rewrite *in*clusively.
        kws = kws.copy()
        kws.pop('cmap', None)
        kws.pop('norm', None)
        kws.pop('center', None)
        kws.pop('annot', None)
        kws.pop('vmin', None)
        kws.pop('vmax', None)
        kws.pop('robust', None)
        kws.pop('xticklabels', None)
        kws.pop('yticklabels', None)

        # Plot the row colors
        if self.row_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.row_colors, yind, axis=0)

            # Get row_color labels
            if self.row_color_labels is not None:
                row_color_labels = self.row_color_labels
            else:
                row_color_labels = False

            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_row_colors,
                    xticklabels=row_color_labels, yticklabels=False, **kws)

            # Adjust rotation of labels
            if row_color_labels is not False:
                plt.setp(self.ax_row_colors.get_xticklabels(), rotation=90)
        else:
            despine(self.ax_row_colors, left=True, bottom=True)

        # Plot the column colors
        if self.col_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(
                self.col_colors, xind, axis=1)

            # Get col_color labels
            if self.col_color_labels is not None:
                col_color_labels = self.col_color_labels
            else:
                col_color_labels = False

            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_col_colors,
                    xticklabels=False, yticklabels=col_color_labels, **kws)

            # Adjust rotation of labels, place on right side
            if col_color_labels is not False:
                self.ax_col_colors.yaxis.tick_right()
                plt.setp(self.ax_col_colors.get_yticklabels(), rotation=0)
        else:
            despine(self.ax_col_colors, left=True, bottom=True)

    def plot_matrix(self, colorbar_kws, xind, yind, config, **kws):
        self.data2d = self.data2d.iloc[yind, xind]
        self.mask = self.mask.iloc[yind, xind]

        # Try to reorganize specified tick labels, if provided
        xtl = kws.pop("xticklabels", "auto")
        try:
            xtl = np.asarray(xtl)[xind]
        except (TypeError, IndexError):
            pass
        ytl = kws.pop("yticklabels", "auto")
        try:
            ytl = np.asarray(ytl)[yind]
        except (TypeError, IndexError):
            pass

        # Reorganize the annotations to match the heatmap
        annot = kws.pop("annot", None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
                annot_data = annot_data[yind][:, xind]
            annot = annot_data

        # Setting ax_cbar=None in clustermap call implies no colorbar
        kws.setdefault("cbar", self.ax_cbar is not None)
        heatmap(self.data2d, ax=self.ax_heatmap, cbar_ax=self.ax_cbar,
                cbar_kws=colorbar_kws, mask=self.mask,
                xticklabels=xtl, yticklabels=ytl, annot=annot, **kws)

        ytl = self.ax_heatmap.get_yticklabels()
        ytl_rot = None if not ytl else ytl[0].get_rotation()
        self.ax_heatmap.yaxis.set_ticks_position('right')
        self.ax_heatmap.yaxis.set_label_position('right')
        if ytl_rot is not None:
            ytl = self.ax_heatmap.get_yticklabels()
            plt.setp(ytl, rotation=ytl_rot)

        if config is not None:
            if config['stim_kind'] == 'single':
                stim_index = config['stim_index']
                for ii in range(stim_index.shape[0]):
                    self.ax_heatmap.axvspan(xmin=stim_index[ii][0], xmax=stim_index[ii][1],
                                               facecolor=config['single_stim_color'], alpha=config['alpha'])
            elif config['stim_kind'] == 'multi':
                stim_index = config['multi_stim_index']
                color_len = len(stim_index)
                for ii in range(stim_index.shape[0]):
                    for jj in range(stim_index.shape[1]):
                        self.ax_heatmap.axvspan(xmin=stim_index[ii][jj][0], xmax=stim_index[ii][jj][1],
                                                   facecolor=config['color_map'][color_len - ii - 1],
                                                   alpha=config['alpha'])
            else:
                pass

        tight_params = dict(h_pad=.02, w_pad=.02)
        if self.ax_cbar is None:
            self._figure.tight_layout(**tight_params)
        else:
            # Turn the colorbar axes off for tight layout so that its
            # ticks don't interfere with the rest of the plot layout.
            # Then move it.
            self.ax_cbar.set_axis_off()
            self._figure.tight_layout(**tight_params)
            self.ax_cbar.set_axis_on()
            self.ax_cbar.set_position(self.cbar_pos)

    def plot(self, metric, method, colorbar_kws, row_cluster, col_cluster,
             row_linkage, col_linkage, tree_kws, config, **kws):

        # heatmap square=True sets the aspect ratio on the axes, but that is
        # not compatible with the multi-axes layout of clustergrid
        if kws.get("square", False):
            msg = "``square=True`` ignored in clustermap"
            warnings.warn(msg)
            kws.pop("square")

        colorbar_kws = {} if colorbar_kws is None else colorbar_kws

        self.plot_dendrograms(row_cluster, col_cluster, metric, method,
                              row_linkage=row_linkage, col_linkage=col_linkage,
                              tree_kws=tree_kws)
        try:
            xind = self.dendrogram_col.reordered_ind
        except AttributeError:
            xind = np.arange(self.data2d.shape[1])
        try:
            yind = self.dendrogram_row.reordered_ind
        except AttributeError:
            yind = np.arange(self.data2d.shape[0])

        self.plot_colors(xind, yind, **kws)
        self.plot_matrix(colorbar_kws, xind, yind, config, **kws)
        return self
