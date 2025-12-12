"""
IROS output plotting.
"""

from typing import Any, Sequence
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from bloodmoon.coords import shift2angle
from bloodmoon.mask import CodedMaskCamera

from .types import Tag
from .data import Log
from .images import crop

__all__ = [
    "DSPlot", "map4plot", "plot", "distr_plot",
    "map4image", "image_plot", "slices_plot",
    "skyfield_map",
]


RCPARAMS = {
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.formatter.limits': (-3, 3),
    'savefig.bbox': 'tight',
    'savefig.dpi': 350,
}
mpl.rcParams.update(RCPARAMS)

PLOTPARAMS = {
    # figure
    'size': 6,
    'wspacing_plot': 1.25,
    'hspacing_plot': 1.25,
    'wspacing_image': 1.25,
    'hspacing_image': 0.5,
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
    # colorbar
    'cbar_origin': 'lower',
    'cbar_shrink': 0.75,
    'cbar_pad_right': 0.03,
    'cbar_pad_bottom': 0.1,
    'cbar_ticks_ls': 11,
    'cbar_label_fs': 12,
    'cbar_label_fw': 'bold',
    'cbar_scilim': (-3, 3),
    # text
    'txt_body_fs': 6,
    'txt_title_fs': 12,
    'txt_fw': 'bold',
    'txt_color': 'black',
    # line/scatter/bar plot styles
    'scatter_size': 50,
    'scatter_lw': 1.5,
}


class DSPlot:
    """
    Plot configuration for the `darksun` package.
    This class initializes the Figure of the plot with the
    style setup in the `darksun.show.py` module.

    Args:
        ncolumns (int):
            Number of subplot columns for the Plot.
        nrows (int):
            Number of subplot rows for the Plot.
    
    Examples:
        >>> # suppose to create a basic plot
        >>> dsp = DSPlot(ncolumns=1, nrows=1)
        >>> fig, ax = dsp.config_subplots()
        >>> dsp.config_labels(ax, title, None, None)
        >>> dsp.config_ticks(ax)
        >>> ...                              # plot func here
        >>> plt.show()
    """
    def __init__(
        self,
        ncolumns: int,
        nrows: int,
    ) -> None:
        self.ncols = ncolumns
        self.nrows = nrows
    
    def config_subplots(
        self,
        *,
        size: int | float = PLOTPARAMS['size'],
        dpi: int | float = PLOTPARAMS['dpi'],
        ptype: str = 'plot',
        **kwargs: Any,
    ) -> tuple[Figure, Axes | NDArray]:
        """
        Creates and configures subplots.
        Default keys are specified in the setup map in `darksun.show.py`.

        Args:
            size (int | float, optional (default=PLOTPARAMS['size'])):
                Figure size for one plot. If multiple rows/cols, the size
                is automathically adjusted to the number of subplots.
            dpi (int | float, optional (default=PLOTPARAMS['dpi'])):
                Figure DPI value.
            ptype (str, optional (default='plot')):
                Subplots plot type, can be 'plot' for plots or 'image'
                for images plotting (e.g., see `plot()` and `image_plot()`
                in the `darksun.show.py` module).
            kwargs (Any):
                Additional keywords passed to `plt.subplot()`.

        Returns:
            output (tuple[Figure, Axes | NDArray]):
                Output from `plt.subplots()`, i.e. a Figure obj and a
                Axes or array of Axes based on chosen cols and rows number.

        Raises:
            ValueError: If `ptype` is not 'plot' or 'image'.
        """
        if ptype not in ('plot', 'image'):
            raise ValueError(
                f"Invalid 'ptype'={ptype}. Must be 'plot' for plots or 'image' for images."
            )
        height, width = map(
            lambda x: int(size * x + 1) if x > 1 else size,
            (self.nrows, self.ncols),
        )
        fig, axs = plt.subplots(
            self.nrows, self.ncols, figsize=(width, height), **kwargs,
        )
        fig.dpi = dpi
        fig.tight_layout()
        fig.subplots_adjust(
            wspace=PLOTPARAMS[f'wspacing_{ptype}'] / size,
            hspace=PLOTPARAMS[f'hspacing_{ptype}'] / size,
        )
        return fig, axs

    @staticmethod
    def config_labels(
        ax: Axes,
        title: str | None,
        xlabel: str | None,
        ylabel: str | None,
    ) -> None:
        """
        Configures labels and titles for the specified Axes.

        Args:
            ax (Axes): Plot Axes object to configure.
            title (str | None): Plot title.
            xlabel (str | None): Label for the x-axis.
            ylabel (str | None): Label for the y-axis.
        """
        ax.set_title(
            label=title or '', fontsize=PLOTPARAMS['title_fs'],
            pad=PLOTPARAMS['title_pad'], fontweight=PLOTPARAMS['label_fw'],
        )
        ax.set_xlabel(
            xlabel=xlabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
        )
        ax.set_ylabel(
            ylabel=ylabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
        )

    @staticmethod
    def config_ticks(
        ax: Axes,
        xticks: ArrayLike | None = None,
        yticks: ArrayLike | None = None,
    ) -> None:
        """
        Configures grid and tick properties for the specified Axes.

        Args:
            ax (Axes): Plot Axes object to configure.
            xticks (ArrayLike | None): Ticks values for the x axis.
            yticks (ArrayLike | None): Ticks values for the y axis.
        """
        ax.grid(
            visible=True, color=PLOTPARAMS['grid_color'], linestyle=PLOTPARAMS['grid_ls'],
            linewidth=PLOTPARAMS['grid_lw'], alpha=PLOTPARAMS['grid_alpha'],
        )
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(
            which='both', direction='in', width=PLOTPARAMS['ticks_w'],
            length=PLOTPARAMS['ticks_len_major'] if 'major' else PLOTPARAMS['ticks_len_minor'],
        )
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

    @staticmethod
    def config_axscale(
        ax: Axes,
        xscale: str,
        yscale: str,
    ) -> None:
        """
        Configures axes scales for the specified Axes.

        Args:
            ax (Axes): Plot Axes object to configure.
            xscale (str): Plot scale for the x-axis.
            yscale (str): Plot scale for the y-axis.
        """
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    @staticmethod
    def config_axlim(
        ax: Axes,
        xlim: tuple[Any, Any],
        ylim: tuple[Any, Any],
    ) -> None:
        """
        Configures axes values limits for the specified Axes.

        Args:
            ax (Axes): Plot Axes object to configure.
            xlim (tuple[Any, Any]): Value limits for the x-axis.
            ylim (tuple[Any, Any]): Value limits for the y-axis.
        """
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

"""                               
                                             █████████████                     
                                        ███████  ░░░░  ███████                 
                                      ████       ░░░░       ████               
                                    ████░░     ░░░░░░░░     ░░████             
                                    ██  ░░░░░░░░░░░░░░░░░░░░░░  ██             
                                  ████    ░░░░░        ░░░░░    ████           
                                  ████    ░░░░░        ░░░░░    ████           
                                  ██      ░░░             ░░      ██           
                                  ██      ░░░             ░░      ██           
                                  ██    ░░░░░             ░░░░    ██           
                                  ██░░░░░░░░░░░        ░░░░░░░░░░░██           
                                  ██░░░░██████████████████████░░░░██           
                                  ████████     ██    ██     ████████           
                                  ████████     ██    ██     ████████           
                                    ████       ██    ██       ████             
                          ██████████████                      ██               
                       ███████      █████████████████████████████              
                       ███████      █████████████████████████████              
                       ███          █████████████████████████████████████     
                       ███          ██████    ██████   █████           ▒████   
                       ███████      ██████    ██████   █████    ██████   ▒███ 
                          ████      ██████    ██████   █████    ██████   ▒███ 
                          ████      ██████    ██████   █████    ██████   ▒███ 
                          ████      ██████    ██████   █████           ▒████
                          ████      ██████    ██████   █████    █████████
                    ██████████      ████████          ██████    ██████ 
                    ███                   ████████████████████████████            
                    ███                   ██████            
                    ████████████████████████████                        
"""

def map4plot(
    arrs: NDArray | Sequence[NDArray],
    title: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: str | Sequence[str | None] | None = None,
    x: NDArray | Sequence[NDArray] | None = None,
    style: str | Sequence[str] | None = None,
    color: str | tuple[str, str] | Sequence[str | tuple[str, str]] | None = None,
    xlim: tuple[Any, Any] = (None, None),
    ylim: tuple[Any, Any] = (None, None),
    xscale: str = 'linear',
    yscale: str = 'linear',
    tags: Tag | Sequence[Tag] | None = None,
) -> dict[str, Any]:
    """
    Configures a dictionary with the specified info for plotting.
    This method can be used to generate a map to give as input to `darksun.plot()`.

    For parameters like `labels`, `style`, and `color`, it is possible to provide a single value to
    apply to all data series or a sequence of values to style each series individually.
    The function processes these inputs and returns them in a structured dictionary,
    ready for a plotting utility.

    Args:
        arrs (NDArray | Sequence[NDArray]):
            The data to be plotted. Can be a single NumPy array or a sequence of arrays.
        title (str):
            The main title for the plot.
        xlabel (str | None, optional (default=`None`)):
            The label for the plot's x-axis.
        ylabel (str | None, optional (default=`None`)):
            The label for the plot's y-axis.
        labels (str | Sequence[str | None] | None, optional (default=`None`)):
            The labels for the data series. If a single string is provided, it's applied to all
            series. If a sequence of strings is given, each series gets its corresponding label.
        x (NDArray | Sequence[NDArray] | None, optional (default=`None`)):
            The x-coordinates for the data points. If not provided, it defaults to `np.arange(n)`
            for each array. If a single array is provided, it's used for all data series. A
            sequence of arrays can be used to specify x-values for each data series.
        style (str | Sequence[str] | None, optional (default=`None`)):
            The plotting style ('plot', 'scatter', 'stairs'). A single string applies the same
            style to all series. A sequence of strings applies a different style to each.
            If `None`, the style is initialised to 'plot' for all array entries.
        color (str | tuple[str, str] | Sequence[str | tuple[str, str]] | None, optional (default=`None`)):
            The color for the data series. A single color string (e.g., 'blue') applies to
            all series. A sequence of color strings styles each series individually.
            For 'scatter' plots, it is possible to insert a tuple to specify the facecolor
            and the edgecolor of the points (e.g., ('SkyBlue', 'DodgerBlue')).
        xlim (tuple[Any, Any], optional (default=`tuple(None, None)`)):
            A tuple `(min, max)` setting the limits for the x-axis.
            This setting applies to the entire plot.
        ylim (tuple[Any, Any], optional (default=`tuple(None, None)`)):
            A tuple `(min, max)` setting the limits for the y-axis.
            This setting applies to the entire plot.
        xscale (str, optional (default=`linear`)):
            The scale for the x-axis (e.g., 'linear', 'log').
            Applies to the entire plot.
        yscale (str, optional (default=`linear`)):
            The scale for the y-axis (e.g., 'linear', 'log').
            Applies to the entire plot.
        tags (Tag | Sequence[Tag] | None, optional (default=`None`)):
            Single `Tag` or sequence of `Tag`s objects (e.g., to mark the
            scatter points with the ID of the sources).
    
    Returns:
        output (dict[str, Any]): Map with the info for the plot.
    
    Example:
        >>> # Example 1
        >>> import numpy as np
        >>> arr1 = np.array([1, 2, 3])
        >>> # using default values
        >>> params = map4plot(
        ...     arrs=[arr1, arr2],
        ...     title="My Plot",
        ... )
        >>> print(params['title'])
        >>> "My Plot"
        >>> print(params['labels'])
        >>> (None,)
        ...
        >>> # Example 2
        >>> import numpy as np
        >>> arrs = (np.array([1, 2, 3]), np.array([3, 2, 1]))
        >>> # using single values for style and label
        >>> params = map4plot(
        ...     arrs=arrs,
        ...     title="My Plot",
        ...     labels="Series",
        ...     style="scatter"
        ... )
        >>> print(params['title'])
        >>> "My Plot"
        >>> print(params['labels'])
        >>> ("Series", "Series")
        >>> print(params['color'])
        >>> (None, None)
    """
    arrs_ = (arrs,) if isinstance(arrs, np.ndarray) else tuple(arrs)
    N_PLOTS = len(arrs_)
    tags_ = (tags,) if isinstance(tags, Tag) else tags

    def setup(
        x: Any,
        dtype: Any,
        *,
        default: Any = None,
        special: Any = None,
    ) -> tuple[Any]:
        """Setup variables for plotting."""
        if x is None:
            return (default,) * N_PLOTS if special is None else special
        elif isinstance(x, dtype):
            return (x,) * N_PLOTS
        return tuple(x)
    
    dmap = {
        'arrs': arrs_,
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'labels': setup(labels, str),
        'x': setup(
            x, np.ndarray, special=tuple(np.arange(len(arr)) for arr in arrs_),
        ),
        'style': setup(style, str, default='plot'),
        'color': setup(color, str),
        'xlim': xlim,
        'ylim': ylim,
        'xscale': xscale,
        'yscale': yscale,
        'tags': tags_,
    }
    return dmap


def plot(
    dmaps: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    ncols: int = 1,
    nrows: int = 1,
    save_to: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Displays a figure with the specified subplots by taking the info stored in the
    dictionaries in input. The maps must have the structure described in `map4plot()`.
    Each subplot displays a plot with the sequences in the respective dmap.

    Args:
        dmaps (dict[str, Any] | Sequence[dict[str, Any]]):
            Dictionary with the info for the plots.
        ncols (int, optional (default=`1`)):
            Number of columns to insert in the plot.
        nrows (int, optional (default=`1`)):
            Number of rows to insert in the plot.
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.
        **kwargs (Any):
            Additional arguments passed to plot func (e.g., `plt.plot()`).
    
    Raises:
        ValueError: If plot style different from 'plot', 'scatter' or 'stairs'.
    
    Example:
        >>> # build maps from `map4plot()`
        >>> dmap1 = map4plot(
        ...     ...,
        ... )
        >>> dmap2 = map4plot(
        ...     ...,
        ... )
        >>> # plot maps
        >>> plot(dmap1)                           # single plot
        >>> plot(dmaps=(dmap1, dmap2), ncols=2)   # double plot on two cols
        >>> plot(dmaps=(dmap1, dmap2), nrows=2)   # double plot on two rows
        >>> plot(dmaps=(...), ncols=2, nrows=2)   # plots on two cols and rows
    """
    dsp = DSPlot(ncolumns=ncols, nrows=nrows)
    fig, axs = dsp.config_subplots()
    dmaps_ = (dmaps,) if isinstance(dmaps, dict) else dmaps
    axs_ = (
        (axs,) if isinstance(axs, Axes)
        else axs.flatten() if (isinstance(axs, np.ndarray) and np.ndim(axs) > 1)
        else axs
    )
    if (len(dmaps_) > np.prod((nrows, ncols))):
        warnings.warn(
            "To display all the input dmaps, select adequate values of 'ncols' or 'nrows'."
        )

    for ax, dmap in zip(axs_, dmaps_):
        dsp.config_labels(ax, dmap['title'], dmap['xlabel'], dmap['ylabel'])
        dsp.config_ticks(ax)
        dsp.config_axscale(ax, dmap['xscale'], dmap['yscale'])
        
        for idx, arr in enumerate(dmap['arrs']):
            match dmap['style'][idx]:
                case 'plot':
                    ax.plot(
                        dmap['x'][idx], arr, c=dmap['color'][idx], alpha=PLOTPARAMS['alpha'],
                        label=dmap['labels'][idx], **kwargs,
                    )
                case 'scatter':
                    c = dmap['color'][idx]
                    fcolor, ecolor = c if isinstance(c, tuple) else (c, c)
                    ax.scatter(
                        dmap['x'][idx], arr, c=fcolor, edgecolors=ecolor, s=PLOTPARAMS['scatter_size'],
                        alpha=PLOTPARAMS['alpha'], linewidths=PLOTPARAMS['scatter_lw'],
                        label=dmap['labels'][idx], **kwargs,
                    )
                case 'stairs':
                    ax.stairs(
                        arr, edges=dmap['x'][idx], edgecolor=dmap['color'][idx],
                        alpha=PLOTPARAMS['alpha'], label=dmap['labels'][idx], **kwargs,
                    )
                case _:
                    raise ValueError(
                        f"Invalid plot style '{dmap['style'][idx]}'. Must be 'plot', 'scatter' or 'stairs'."
                    )
            
        if dmap['tags'] is not None:
            for (name, y, x) in dmap['tags']:
                ax.text(
                    x, y, name, color=PLOTPARAMS['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
        dsp.config_axlim(ax, dmap['xlim'], dmap['ylim'])
        if any(dmap['labels']): ax.legend(loc='best')
    
    if save_to is not None: fig.savefig(save_to)
    plt.show()


def distr_plot(
    arr: NDArray,
    title: str,
    *,
    bins: int | Sequence[int | float] = 50,
    xlabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    pdf_distr: tuple[str, NDArray] | None = None,
    save_to: str | Path | None = None,
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
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.
        hist_kwargs (Any): Keyword arguments passed to `matplotlib.pyplot.hist`.
    """
    if np.ndim(arr_) > 1: arr_ = arr.reshape(-1)

    dsp = DSPlot(1, 1)
    fig, ax = dsp.config_subplots()
    dsp.config_labels(ax, title, xlabel, 'density')
    dsp.config_ticks(ax)
    dsp.config_axscale(ax, 'linear', 'linear')
    ax.hist(
        arr_, bins=bins, density=True, alpha=PLOTPARAMS['alpha'], **hist_kwargs,
    )
    if pdf_distr:
        ax.plot(
            np.arange(len(pdf_distr[1])), pdf_distr[1], color='OrangeRed', label=f'{pdf_distr[0]}',
        )
        ax.legend(loc='best')
    dsp.config_axlim(ax, xlim, (None, None))
    if save_to is not None: fig.savefig(save_to)
    plt.show()


def map4image(
    img: NDArray,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbarlabel: str = None,
    cbarloc: str = 'right',
    tags: Tag | Sequence[Tag] | None = None,
    img_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Configures a dictionary with the specified info for plotting a
    2D array as image. This method can be used to generate a map to
    give as input to `darksun.image_plot()`.

    Args:
        img (NDArray):
            2D array to display.
        title (str, optional (default=`None`)):
            Title for the plot.
        xlabel (str, optional (default=`None`)):
            Label for the x-axis.
        ylabel (str, optional (default=`None`)):
            Label for the y-axis.
        cbarlabel (str, optional (default=`None`)):
            Label for the colorbar.
        cbarloc (str, optional (default=`'right'`)):
            Location of the colorbar.
        tags (Tag | Sequence[Tag] | None, optional (default=`None`)):
            Single `Tag` or sequence of `Tag`s objects (e.g., to
            mark the position of the sources, with respective ID).
        img_kwargs (dict[str, Any], optional (default=`None`)):
            Keyword arguments passed to `matplotlib.pyplot.imshow`.
    """
    dmap = {
        'img': img,
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'cbarlabel': cbarlabel,
        'cbarloc': cbarloc,
        'img_kwargs': img_kwargs,
        'tags': (tags,) if isinstance(tags, Tag) else tags,
    }
    return dmap


def image_plot(
    dmaps: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    ncols: int = 1,
    nrows: int = 1,
    save_to: str | Path | None = None,
) -> None:
    """
    Displays a figure with the specified subplots by taking the info stored in the
    dictionaries in input. The maps must have the structure described in `map4image()`.
    Each subplot displays a 2D array as image.

    Args:
        dmaps (dict[str, Any] | Sequence[dict[str, Any]]):
            Dictionary with the info for the image plots.
        ncols (int, optional (default=`1`)):
            Number of columns to insert in the plot.
        nrows (int, optional (default=`1`)):
            Number of rows to insert in the plot.
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.ù
    
    Example:
        >>> # build maps from `map4image()`
        >>> dmap1 = map4image(
        ...     ...,
        ... )
        >>> dmap2 = map4image(
        ...     ...,
        ... )
        >>> # plot maps
        >>> image_plot(dmap1)                           # single img
        >>> image_plot(dmaps=(dmap1, dmap2), ncols=2)   # double img on two cols
        >>> image_plot(dmaps=(dmap1, dmap2), nrows=2)   # double plot on two rows
        >>> image_plot(dmaps=(...), ncols=2, nrows=2)   # imgs on two cols and rows
    """
    dsp = DSPlot(ncols, nrows)
    fig, axs = dsp.config_subplots(dpi=110, ptype='image')
    dmaps_ = (dmaps,) if isinstance(dmaps, dict) else dmaps
    axs_ = (
        (axs,) if isinstance(axs, Axes)
        else axs.flatten() if (isinstance(axs, np.ndarray) and np.ndim(axs) > 1)
        else axs
    )

    for ax, dmap in zip(axs_, dmaps_):
        dsp.config_labels(ax, dmap['title'], dmap['xlabel'], dmap['ylabel'])
        dsp.config_ticks(ax)
        kwargs_ = dmap['img_kwargs'] or {}
        img = ax.imshow(dmap['img'], origin=PLOTPARAMS['cbar_origin'], **kwargs_)
        cbar = fig.colorbar(
            img, ax=ax, location=dmap['cbarloc'], shrink=PLOTPARAMS['cbar_shrink'],
            pad=PLOTPARAMS[f'cbar_pad_{dmap['cbarloc']}'],
        )
        cbar.ax.tick_params(labelsize=PLOTPARAMS['cbar_ticks_ls'])
        cbar.formatter.set_powerlimits(PLOTPARAMS['cbar_scilim'])
        if dmap['cbarlabel']:
            cbar.set_label(
                dmap['cbarlabel'], fontsize=PLOTPARAMS['cbar_label_fs'], fontweight=PLOTPARAMS['cbar_label_fw'],
            )
        if dmap['tags'] is not None:
            for (name, y, x) in dmap['tags']:
                ax.scatter(
                    x, y, s=12, facecolors='None', edgecolors='white',
                    linewidths=1.0, alpha=0.9,
                )
                ax.text(
                    x + 125, y, name, color='white', rotation=45,
                    fontsize=PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
        
    if save_to is not None: fig.savefig(save_to)
    plt.show()


def slices_plot(
    sky: NDArray | Sequence[NDArray],
    pos: tuple[int, int],
    crp: tuple[int, int],
    *,
    source: str | None = None,
    ylabel: str | None = None,
    ylim_xslice: tuple[Any, Any] | None = None,
    ylim_yslice: tuple[Any, Any] | None = None,
    labels: str | Sequence[str | None] | None = None,
    cameraID: str | None = None,
    save_to: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Plots horizontal and vertical slices of 2D data arrays centered at a given position.

    This function extracts 1D slices along the x- and y-axes from one or more 2D arrays 
    centered at `pos` with size `2 * crp + 1`. The resulting slices are plotted using 
    a dual-panel plot, with optional customization for axis labels, y-axis limits, 
    and plot annotations.

    Parameters:
        sky (NDArray | Sequence[NDArray]):
            Single 2D array or sequence of 2D arrays representing image-like data.
        pos (tuple[int, int]):
            The (y, x) center coordinates around which to extract the slices.
        crp (tuple[int, int]):
            Size of the cropping along (y, x).
        source (str | None, optional (default=`None`)):
            Optional source label to include in plot titles.
        ylabel (str | None, optional (default=`None`)):
            Label for the y-axis of both slice plots.
        ylim_xslice (tuple[Any, Any] | None, optional (default=`None`)):
            Y-axis limits for the x-axis slice plot.
        ylim_yslice (tuple[Any, Any] | None, optional (default=`None`)):
            Y-axis limits for the y-axis slice plot.
        labels (str | Sequence[str | None] | None, optional (default=`None`)):
            Labels for each 2D array slice, used in the legend.
        cameraID (str | None, optional (default=`None`)):
            ID for the LEM-X coded-mask camera unit.
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.
        **kwargs (Any):
            Additional keyword arguments for the `plot` function.
    """
    colors_ = (
        'OrangeRed', 'DodgerBlue', 'm', 'Lawngreen',
    )

    def phase(x: NDArray) -> NDArray:
        """Centers x-axis values around zero."""
        return np.arange(len(x) + 1) - len(x) // 2 - 0.5
    
    cropped = tuple(
        crop(s, pos, crp, strict=False) for s in (
            (sky,) if isinstance(sky, np.ndarray) else sky
        )
    )
    xslice, yslice = zip(
        *tuple((c[crp[0], :], c[:, crp[1]]) for c in cropped)
    )
    name, cam = map(
        lambda x: x.upper() if x else '',
        (source, cameraID),
    )

    dmapx = map4plot(
        arrs=xslice,
        title=f"{name} X-axis Slice {cam}",
        xlabel='x [px]',
        ylabel=ylabel or 'counts [ph]',
        labels=labels,
        x=map(phase, xslice),
        style='stairs',
        color=colors_,
        ylim=ylim_xslice or (None, None),
    )
    dmapy = map4plot(
        arrs=yslice,
        title=f"{name} Y-axis Slice {cam}",
        xlabel='y [px]',
        ylabel=ylabel or 'counts [ph]',
        labels=labels,
        x=map(phase, yslice),
        style='stairs',
        color=colors_,
        ylim=ylim_yslice or (None, None),
    )
    plot(
        dmaps=(dmapx, dmapy),
        ncols=2,
        save_to=save_to,
        **kwargs,
    )


def skyfield_map(
    log: Log,
    camera: CodedMaskCamera,
    *,
    show_IDs: bool = True,
    show_coords: bool = False,
    show_errbox: bool = False,
    save_to: str | Path | None = None,
) -> None:
    """
    Displays the reconstructed sources in the sky-grid, optionally
    specifying the IDs, the RA/Dec coordinates and the coords errors.

    The input `log` container must have at least the sources camera
    local-frame sky-angular coords (in [deg]).

    Args:
        log (Log):
            Reconstructed sources database.
        camera (CodedMaskCamera):
            CodedMaskCamera instance with binning info.
        show_IDs (bool, optional (default=`True`)):
            If `True`, the source IDs are shown in the plot
            (the log must have the IDs entries stored inside).
        show_coords (bool, optional (default=`False`)):
            If `True`, the RA/Dec coords are shown in the plot
            (the log must have the coords entries stored inside).
        show_errbox (bool, optional (default=`False`)):
            If `True`, the source pos errorboxes are displayed
            (the log must have the angular coords errors stored inside).
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.
    """
    _skyx, _skyy = camera.bins_sky
    skyx, skyy = map(
        lambda x: shift2angle(camera, x),
        (_skyx, _skyy),
    )
    SETUP = {
        'x': 'angle_x',
        'err_x': 'dangle_x',
        'y': 'angle_y',
        'err_y': 'dangle_y',
        'xlabel': '$\\theta_{fine}$ [deg]',
        'ylabel': '$\\theta_{coarse}$ [deg]',
        'title': f'{log.name} Local Frame SkyMap Grid',

        'ancor': (skyx[0], skyy[0]),
        'width': skyx[-1] - skyx[0],
        'height': skyy[-1] - skyy[0],
        'color': '#120E16',
        'txt_color': 'white',
        'rot': 35,
        'errbox_color': 'Orange',
        'alpha': 0.5,

        'xline': ((skyx[0], skyx[-1]), (0, 0)),
        'yline': ((0, 0), (skyy[0], skyy[-1])),

        'xlim': (skyx[0], skyx[-1]),
        'ylim': (skyy[0], skyy[-1]),
    }

    dsp = DSPlot(1, 1)
    fig, ax = dsp.config_subplots(size=8, dpi=100)
    dsp.config_labels(ax, SETUP['title'], SETUP['xlabel'], SETUP['ylabel'])
    dsp.config_ticks(ax)

    # plot bkg and axis lines
    ax.add_patch(
        Rectangle(
            xy=SETUP['ancor'], width=SETUP['width'], height=SETUP['height'],
            linewidth=0.1, edgecolor=SETUP['color'], facecolor=SETUP['color'],
        )
    )
    for line in ('xline', 'yline'):
        ax.plot(*SETUP[line], color='OrangeRed', linestyle='-', linewidth=0.6, alpha=0.25)
    
    # plot sources on grid
    ax.scatter(
        log.log[SETUP['x']], log.log[SETUP['y']], c=None,
        edgecolor='LawnGreen', alpha=SETUP['alpha'], s=10,
    )
    if any((show_IDs, show_coords, show_errbox)):
            if show_IDs:
                for name, x, y in zip(
                    log.log['ID'],
                    log.log[SETUP['x']],
                    log.log[SETUP['y']],
                ):
                    ax.text(
                        x + 0.475, y + 0.385, name, color=SETUP['txt_color'], rotation=SETUP['rot'],
                        fontsize=0.95*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                    )
            if show_coords:
                for x, y, ra, dec in zip(
                    log.log[SETUP['x']],
                    log.log[SETUP['y']],
                    log.log['ra'],
                    log.log['dec'],
                ):
                    ax.text(
                        x - 5, y - 0.5, f'RA: {ra:.4f}\nDEC: {dec:.4f}', color=SETUP['txt_color'],
                        rotation=SETUP['rot'], fontsize=0.95*PLOTPARAMS['txt_body_fs'],
                        fontweight=PLOTPARAMS['txt_fw'],
                    )
            if show_errbox:
                for x, dx, y, dy in zip(
                    log.log[SETUP['x']],
                    log.log[SETUP['err_x']],
                    log.log[SETUP['y']],
                    log.log[SETUP['err_y']],
                ):
                    ax.add_patch(
                        Rectangle(
                            xy=(x - dx, y - dy), width=2 * dx, height=2 * dy,
                            linewidth=0.2, edgecolor=SETUP['errbox_color'], facecolor=None,
                        )
                    )

    dsp.config_axlim(ax, SETUP['xlim'], SETUP['ylim'])
    if save_to is not None: fig.savefig(save_to)
    plt.show()


# end