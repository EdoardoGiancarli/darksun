"""
IROS output plotting.
"""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from bloodmoon.mask import CodedMaskCamera
from darksun.data import Log

__all__ = []


RCPARAMS = {
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
PLOTPARAMS = {
    # figure
    'size': 6,
    'spacing': 1.25,
    'dpi': 200,
    # plot
    'title_pad': 8,
    'title_fs': 14,
    'label_fs': 12,
    'label_fw': 'bold',
    'alpha': 0.75,
    'grid_color': 'lightgray',
    'grid_ls': '-',
    'grid_lw': 0.2,
    'grid_alpha': 0.7,
    'ticks_w': 2,
    'ticks_len_major': 7,
    'ticks_len_minor': 4,
    'ticks_scilim': (-3, 4),
    # colorbar
    'cbar_origin': 'lower',
    'cbar_shrink': 0.75,
    'cbar_pad': 0.05,
    'cbar_ticks_ls': 11,
    'cbar_label_fs': 12,
    'cbar_label_fw': 'bold',
    'cbar_scilim': (-3, 3),
    # text
    'txt_body_fs': 6,
    'txt_title_fs': 12,
    'txt_fw': 'bold',
    'txt_color': 'black',
}
mpl.rcParams.update(RCPARAMS)


def _config_subplots(nplots: int) -> None:
    """Creates and configures subplots."""
    size = PLOTPARAMS['size']
    fig_size = (size * nplots + 1, size) if nplots > 1 else (size, size)
    fig, axes = plt.subplots(1, nplots, figsize=fig_size)
    fig.tight_layout()
    fig.dpi = PLOTPARAMS['dpi']
    fig.subplots_adjust(wspace=PLOTPARAMS['spacing'] / size)
    return fig, (axes if nplots > 1 else [axes])

def _config_labels(ax: Axes, xlabel: str | None, ylabel: str | None, title: str) -> None:
    """Configures labels and titles."""
    ax.set_xlabel(
        label=xlabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
    )
    ax.set_ylabel(
        label=ylabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
    )
    ax.set_title(
        label=title, fontsize=PLOTPARAMS['title_fs'],
        pad=PLOTPARAMS['title_pad'], fontweight=PLOTPARAMS['label_fw'],
    )

def _config_ticks(ax: Axes) -> None:
    """Configures grid and tick properties."""
    ax.grid(
        visible=True, color=PLOTPARAMS['lightgray'], linestyle=PLOTPARAMS['grid_ls'],
        linewidth=PLOTPARAMS['grid_lw'], alpha=PLOTPARAMS['grid_alpha'],
    )
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(
        which='both', direction='in', width=PLOTPARAMS['ticks_w'],
        length=PLOTPARAMS['ticks_len_major'] if 'major' else PLOTPARAMS['ticks_len_minor'],
    )
    ax.ticklabel_format(scilimits=PLOTPARAMS['ticks_scilim'])


def biplot(
    *args, **kwargs,
) -> None:
    """
    
    """
    raise NotImplementedError


def distr_plot(
    arr: NDArray,
    title: str,
    bins: int | Sequence[int | float] = 50,
    xlabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    pdf_distr: tuple[str, NDArray] | None = None,
    **hist_kwargs: Any,
) -> None:
    """
    Plots the histogram of the values inside the input array.
    Array can be N-dim, it will be flattened for the plot.

    Args:
        arr (NDArray):
            Input array, could be N-dim.
        title (str):
            Plot title.
        bins (int | Sequence[int | float], optional (default=`50`)):
            Histogram bins.
        label (str | None, optional (default=`None`)):
            Label for x-axis and for legend.
        xlim (float | tuple[float, float], optional (default=`None`)):
            Edge values for the distribution.
        pdf_distr (tuple[str, NDArray] | None, optional (default=`None`)):
            Input PDF distribution for comparison.
        hist_kwargs (Any)
    """
    arr_ = arr.copy()
    if np.ndim(arr_) > 1: arr_ = arr_.reshape(-1)

    if xlim is not None:
        arr_ = arr_[(xlim[0] < arr_ < xlim[1])]

    fig, ax = _config_subplots(1)
    _config_labels(ax, xlabel, 'density', title)
    _config_ticks(ax)
    ax.hist(
        arr_, bins=bins, density=True, alpha=PLOTPARAMS['alpha'], **hist_kwargs,
    )
    if pdf_distr:
        ax.plot(
            np.arange(len(pdf_distr[1])), pdf_distr[1], color='OrangeRed', label=f'{pdf_distr[0]}',
        )
        ax.legend(loc='best')
    plt.show()


def image_plot(
    arr: NDArray,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    cbarlabel: str = None,
    cbarloc: str = 'right',
    **kwargs: Any,
) -> None:
    """
    Displays a 2D array as an image.

    Args:
        arr (NDArray): 2D array to display.
        title (str, optional (default=`None`)): Title for the plot.
        xlabel (str, optional (default=`None`)): Label for the x-axis.
        ylabel (str, optional (default=`None`)): Label for the y-axis.
        cbarlabel (str, optional (default=`None`)): Label for the colorbar.
        cbarloc (str, optional (default=`'right'`)): Location of the colorbar.
        **kwargs (Any): Additional keyword arguments passed to `matplotlib.pyplot.imshow`.
    """
    fig, ax = _config_subplots(1)
    _config_labels(ax, xlabel, ylabel, title)
    _config_ticks(ax)
    img = ax.imshow(arr, origin=PLOTPARAMS['cbar_origin'], **kwargs)
    cbar = fig.colorbar(
        img, ax=ax, location=cbarloc, shrink=PLOTPARAMS['cbar_shrink'], pad=PLOTPARAMS['cbar_pad'],
    )
    cbar.ax.tick_params(labelsize=PLOTPARAMS['cbar_ticks_ls'])
    cbar.formatter.set_powerlimits(PLOTPARAMS['cbar_scilim'])
    if cbarlabel: cbar.set_label(
        cbarlabel, fontsize=PLOTPARAMS['cbar_label_fs'], fontweight=PLOTPARAMS['cbar_label_fw'],
    )
    plt.show()


def slices_plot(
    *args, **kwargs,
) -> None:
    """
    
    """
    raise NotImplementedError


def sky_plot(
    *args, **kwargs,
) -> None:
    """
    
    """
    raise NotImplementedError


def skyfield_map(
    log: Log,
    camera: CodedMaskCamera,
    show_IDs: bool = True,
    show_coords: bool = True,
    show_errbox: bool = True,
) -> None:
    """
    Displays the reconstructed sources in the sky-grid, optionally
    specifying the IDs and the RA/Dec coordinates.

    Args:
        log (Log):
            Reconstructed sources database.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with binning info.
        show_IDs (bool, optional (default=`True`)):
            If `True`, the source IDs are shown in the plot.
        show_coords (bool, optional (default=`True`)):
            If `True`, the RA/Dec coords are shown in the plot.
        show_errbox (bool, optional (default=`True`)):
            If `True`, the source pos errorboxes are displayed.
    """
    skyx, skyy = camera.bins_sky
    SETUP = {
        'x': 'shift_x',
        'err_x': 'dshift_x',
        'y': 'shift_y',
        'err_y': 'dshift_y',
        'xlabel': 'x bins [mm]',
        'ylabel': 'y bins [mm]',
        'title': f'{log.name} SkyMap-Grid',

        'ancor': (skyx[0], skyy[0]),
        'width': skyx[-1] - skyx[0],
        'height': skyy[-1] - skyy[0],
        'color': 'k',
        'txt_color': 'white',
        'errbox_color': 'OrangeRed',

        'xline': ((skyx[0], skyx[-1]), (0, 0)),
        'yline': ((0, 0), (skyy[0], skyy[-1])),

        'xlim': (skyx[0], skyx[-1]),
        'ylim': (skyy[0], skyy[-1]),
    }

    fig, ax = _config_subplots(1)
    _config_labels(ax, SETUP['xlabel'], SETUP['ylabel'], SETUP['title'])
    _config_ticks(ax)
    ax.set_xlim(SETUP['xlim'])
    ax.set_ylim(SETUP['ylim'])

    # plot bkg and axis lines
    ax.add_patch(
        patches.Rectangle(
            xy=SETUP['ancor'], width=SETUP['width'], height=SETUP['height'],
            linewidth=0.1, edgecolor='k', facecolor=SETUP['color'],
        )
    )
    for line in ('xline', 'yline'):
        ax.plot(*SETUP[line], color='OrangeRed', linestyle='-', linewidth=0.6, alpha=0.25)
    
    # plot sources on grid
    ax.scatter(
        log.log[SETUP['x']], log.log[SETUP['y']], color='LawnGreen',
        marker='+', alpha=PLOTPARAMS['alpha'], s=15, label='sources',
    )
    if any((show_IDs, show_coords, show_errbox)):
        for name, sx, dsx, sy, dsy, ra, dec in zip(
            log.log['ID'],
            log.log[SETUP['x']],
            log.log[SETUP['err_x']],
            log.log[SETUP['y']],
            log.log[SETUP['err_y']],
            log.log['ra'],
            log.log['dec'],
        ):
            if show_coords:
                ax.text(
                    sx - 18, sy + 5, f'RA: {ra:.4f}\nDEC: {dec:.4f}', color=SETUP['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
            if show_IDs:
                ax.text(
                    sx - 5, sy - 5, name, color=SETUP['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
            if show_errbox:
                ax.add_patch(
                    patches.Rectangle(
                        xy=(sx - dsx, sy - dsy), width=2*dsx, height=2*dsy,
                        linewidth=0.1, edgecolor=SETUP['errbox_color'], facecolor=None,
                    )
                )

    ax.legend(loc='best')
    plt.show()


# end