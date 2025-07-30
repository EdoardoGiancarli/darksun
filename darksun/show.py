"""
IROS output plotting.
"""

from typing import Any, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from bloodmoon.mask import CodedMaskCamera
from bloodmoon.optim import model_sky

from .types import Tag
from .data import Log
from .images import crop
from .images import upscale

__all__ = []


RCPARAMS = {
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
mpl.rcParams.update(RCPARAMS)

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
    # line/scatter/bar plot styles
    'scatter_size': 50,
    'scatter_lw': 1.5,
}

def _config_subplots(nplots: int) -> tuple[Figure, Axes | list[Axes]]:
    """Creates and configures subplots."""
    size = PLOTPARAMS['size']
    fig_size = (size * nplots + 1, size) if nplots > 1 else (size, size)
    fig, axs = plt.subplots(1, nplots, figsize=fig_size)
    fig.tight_layout()
    fig.dpi = PLOTPARAMS['dpi']
    fig.subplots_adjust(wspace=PLOTPARAMS['spacing'] / size)
    return fig, (axs if nplots > 1 else [axs])

def _config_labels(ax: Axes, xlabel: str | None, ylabel: str | None, title: str) -> None:
    """Configures labels and titles."""
    ax.set_xlabel(
        xlabel=xlabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
    )
    ax.set_ylabel(
        ylabel=ylabel or '', fontsize=PLOTPARAMS['label_fs'], fontweight=PLOTPARAMS['label_fw'],
    )
    ax.set_title(
        label=title, fontsize=PLOTPARAMS['title_fs'],
        pad=PLOTPARAMS['title_pad'], fontweight=PLOTPARAMS['label_fw'],
    )

def _config_ticks(ax: Axes) -> None:
    """Configures grid and tick properties."""
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
    ax.ticklabel_format(scilimits=PLOTPARAMS['ticks_scilim'])

def _config_view(
    ax: Axes,
    xlim: tuple[Any, Any],
    ylim: tuple[Any, Any],
    xscale: str,
    yscale: str,
) -> None:
    """Configures axes limits and scales."""
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xscale(xscale); ax.set_yscale(yscale)

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

def map4biplot(
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
    This method can be used to generate a map to give as input to `biplot`.

    For parameters like `labels`, `style`, and `color`, you can provide a single value to
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
            The plotting style ('plot', 'scatter', 'bar'). A single string applies the same
            style to all series. A sequence of strings applies a different style to each.
            If `None`, the style is initialised to 'plot' for all array entries.
        color (str | tuple[str, str] | Sequence[str | tuple[str, str]] | None, optional (default=`None`)):
            The color for the data series. A single color string (e.g., 'blue') applies to
            all series. A sequence of color strings styles each series individually.
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
        >>> params = map4biplot(
        ...     arrs=[arr1, arr2],
        ...     title="My Plot",
        ... )
        >>> print(params['title'])
        >>> My Plot
        >>> print(params['labels'])
        >>> (None,)
        ...
        >>> # Example 2
        >>> import numpy as np
        >>> arrs = (np.array([1, 2, 3]), np.array([3, 2, 1]))
        >>> # using single values for style and label
        >>> params = map4biplot(
        ...     arrs=arrs,
        ...     title="My Plot",
        ...     labels="Series",
        ...     style="scatter"
        ... )
        >>> print(params['title'])
        >>> My Plot
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


def biplot(
    dmap_A: dict[str, Any],
    dmap_B: dict[str, Any],
    save_to: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Displays a figure with two subplots by taking the info stored
    in the two dictionaries in input. The two maps must have the
    structure described in `map4biplot()`.

    Args:
        dmap_A (dict[str, Any]):
            Dictionary with the info for the first subplot.
        dmap_B (dict[str, Any]):
            Dictionary with the info for the second subplot.
        save_to (str | Path | None, optional (default=`None`)):
            Path to save the figure.
        **kwargs (Any):
            Additional arguments passed to plot func (e.g., `plt.plot()`).
    
    Raises:
        ValueError: If plot style different from 'plot', 'scatter' or 'bar'.
    
    Example:
        >>> # built maps from `map4biplot()`
        >>> dmap1 = map4biplot(
        ...     ...,
        ... )
        >>> dmap2 = map4biplot(
        ...     ...,
        ... )
        >>> # plot maps
        >>> biplot(dmap1, dmap2)
    """
    fig, axs = _config_subplots(2)
    for ax, dmap in zip(axs, (dmap_A, dmap_B)):
        _config_labels(ax, dmap['xlabel'], dmap['ylabel'], dmap['title'])
        _config_ticks(ax)
        
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
                case 'bar':
                    raise NotImplementedError
                case _:
                    raise ValueError(
                        f"Invalid plot style '{dmap['style'][idx]}'. Must be 'plot', 'scatter' or 'bar'."
                    )
            
        if dmap['tags'] is not None:
            for (name, y, x) in dmap['tags']:
                ax.text(
                    x, y, name, color=PLOTPARAMS['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
        _config_view(
            ax, dmap['xlim'], dmap['ylim'], dmap['xscale'], dmap['yscale'],
        )
        if any(dmap['labels']): ax.legend(loc='best')
    
    if save_to is not None: plt.savefig(save_to)
    plt.show()


def distr_plot(
    arr: NDArray,
    title: str,
    *,
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
        hist_kwargs (Any): Keyword arguments passed to `matplotlib.pyplot.hist`.
    """
    arr_ = arr.copy()
    if np.ndim(arr_) > 1: arr_ = arr_.reshape(-1)

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
    _config_view(ax, xlim, (None, None), 'linear', 'linear')
    plt.show()


def image_plot(
    arr: NDArray,
    *,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    cbarlabel: str = None,
    cbarloc: str = 'right',
    **img_kwargs: Any,
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
        **img_kwargs (Any): Keyword arguments passed to `matplotlib.pyplot.imshow`.
    """
    fig, ax = _config_subplots(1)
    _config_labels(ax, xlabel, ylabel, title)
    _config_ticks(ax)
    img = ax.imshow(arr, origin=PLOTPARAMS['cbar_origin'], **img_kwargs)
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
            ID for the Wide Field Monitor coded-mask camera.
        **kwargs (Any):
            Additional keyword arguments for the `biplot` function.
    """
    def phase(x: NDArray) -> NDArray:
        """Centers x-axis values around zero."""
        return np.arange(len(x)) - len(x) // 2
    
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
    biplot(
        dmap_A=map4biplot(
            arrs=xslice,
            title=f"{name} X-axis Slice {cam}",
            xlabel='x [px]',
            ylabel=ylabel or 'counts [ph]',
            labels=labels,
            x=map(phase, xslice),
            ylim=ylim_xslice or (None, None),
        ),
        dmap_B=map4biplot(
            arrs=yslice,
            title=f"{name} Y-axis Slice {cam}",
            xlabel='y [px]',
            ylabel=ylabel or 'counts [ph]',
            labels=labels,
            x=map(phase, yslice),
            ylim=ylim_yslice or (None, None),
        ),
        **kwargs,
    )


def reconstruction_plot(
    true_sky: NDArray,
    log: Log,
    crp: tuple[int, int],
    camera: CodedMaskCamera,
    *,
    vignetting: bool = True,
    psfy: bool = True,
) -> None:
    """
    Displays the IROS reconstruction effect wrt the original decoded sky.

    Specifically, it shows:
        - IROS reconstructed source slices wrt the true sky
        - source slices residues
        - source profiles residues heatmap

    Args:
        true_sky (NDArray):
            True observed decoded sky.
        log (Log):
            Reconstructed sources database.
        crp (tuple[int, int]):
            Size of the cropping along (y, x).
        camera (CodedMaskCamera):
            CodedMaskCamera instance used for imaging and reconstruction.
        vignetting (bool, optional (default=`True`)):
            Simulates vignetting effects.
        psfy (bool, optional (default=`True`)):
            Simulates detector reconstruction effects.
    
    Notes:
        - A copy of `true_sky` is used to avoid memory overwrite.
    """
    true_sky_ = true_sky.copy()
    UPX, UPY = camera.upscale_f

    for source, sx, sy, f, x, y in zip(
        log.log['ID'],
        log.log['shift_x'],
        log.log['shift_y'],
        log.log['fluence'],
        log.log['x'],
        log.log['y'],
    ):
        POS = (y, x)

        # simulate source
        modeled = model_sky(
            camera=camera,
            shift_x=sx,
            shift_y=sy,
            fluence=f,
            vignetting=vignetting,
            psfy=psfy,
        ).astype(np.int32)

        # plot IROS vs True skies slices (profiles and residues)
        slices_plot(
            sky=(true_sky_, modeled),
            pos=POS,
            crp=crp,
            source=source,
            labels=('True', 'IROS'),
            cameraID=log.name,
        )
        slices_plot(
            sky=true_sky_ - modeled,
            pos=POS,
            crp=crp,
            source=source,
            ylabel='residues [ph]',
            cameraID=log.name,
        )

        # plot IROS vs True residues heatmaps
        CUT = (
            int(camera.specs['slit_deltay'] * UPY / camera.specs['mask_deltay']) + 1,
            int(camera.specs['slit_deltax'] * UPX / camera.specs['mask_deltax']) + 1,
        )
        F_UPY, F_UPX = 2, 5

        true_crp, modeled_crp = map(
            lambda x: crop(x, pos=POS, crp=CUT, strict=False),
            (true_sky_, modeled),
        )
        image_plot(
            arr=upscale(true_crp - modeled_crp, F_UPY, F_UPX) * np.prod((F_UPY, F_UPX)),
            title=f'True - IROS Residues {log.name}',
            cbarlabel='counts',
            cmap='bwr',
            extent=(-CUT[1] * F_UPX, CUT[1] * F_UPX, -CUT[0] * F_UPY, CUT[0] * F_UPY),
        )

        # remove source from True sky
        true_sky_ -= modeled


def sky_plot(
    *args, **kwargs,
) -> None:
    """
    
    """
    raise NotImplementedError


def skyfield_map(
    log: Log,
    camera: CodedMaskCamera,
    *,
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
        Rectangle(
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
        for name, x, dx, y, dy, ra, dec in zip(
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
                    x - 18, y + 5, f'RA: {ra:.4f}\nDEC: {dec:.4f}', color=SETUP['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
            if show_IDs:
                ax.text(
                    x - 5, y - 5, name, color=SETUP['txt_color'],
                    fontsize=0.9*PLOTPARAMS['txt_body_fs'], fontweight=PLOTPARAMS['txt_fw'],
                )
            if show_errbox:
                ax.add_patch(
                    Rectangle(
                        xy=(x - dx, y - dy), width=2 * dx, height=2 * dy,
                        linewidth=0.1, edgecolor=SETUP['errbox_color'], facecolor=None,
                    )
                )

    ax.legend(loc='best')
    plt.show()


# end