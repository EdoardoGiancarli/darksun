"""
IROS output plotting.
"""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

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
        arr (np.array):
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


def enhance_slices(
    *args, **kwargs,
) -> None:
    """
    
    """
    raise NotImplementedError


def image_plot(
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
    *args, **kwargs,
) -> None:
    """
    the one with the sources on the binning grid in placeholder (LCMs)
    """
    raise NotImplementedError


# end