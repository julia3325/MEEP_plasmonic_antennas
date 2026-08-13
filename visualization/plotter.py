# import matplotlib
# matplotlib.use("Agg")
# 
import numpy as np
import os, h5py, meep, sys, gc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Global settings for plotting

## Font
from matplotlib import font_manager
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_path = os.path.join(
    BASE_DIR,
    "fonts",
    "LiberationSerif-Regular.ttf"
)
font_manager.fontManager.addfont(font_path)
rcParams['font.family'] = 'Liberation Serif'
rcParams['font.size'] = 14

## Lines
rcParams['lines.solid_joinstyle'] = 'miter'  # other options: 'round' or 'bevel'
rcParams['lines.antialiased'] = True  # turning on/off of antialiasing for sharper edges
rcParams['lines.linewidth'] = 1.25

## Legend
rcParams['legend.loc'] = 'upper left'
rcParams['legend.frameon'] = False

## Ticks
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True

rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True

## Resolution
rcParams['figure.dpi'] = 150

## Global color
rcParams['image.cmap'] = "viridis"

## Colors
### cmaps
cm_inferno = get_cmap("inferno")
cm_viridis = get_cmap("viridis")
cm_seismic = get_cmap("seismic")
cm_jet = get_cmap("jet")
cm_tab10 = get_cmap("tab10")
cm_rdbu = get_cmap("RdBu")
### Palettes from color-hex.com/
c_google = ['#008744', '#0057e7', '#d62d20', '#ffa700'] # G, B, R, Y # https://www.color-hex.com/color-palette/1872
c_twilight = ['#363b74', '#673888', '#ef4f91', '#c79dd7', '#4d1b7b'] # https://www.color-hex.com/color-palette/809


# Get array of colors from cmap
def cm2c(cmap, c_numb, step=6):
    """
    Convert a colormap to a list of discrete colors.
    This function samples colors from a given colormap at regular intervals
    and returns them as a list of color values.
    Args:
        cmap: A matplotlib colormap object to sample colors from.
        c_numb (int): The number of colors to generate from the colormap.
        step (int, optional): The step size for sampling the colormap. 
            If c_numb is greater than step, step is adjusted to c_numb.
            Defaults to 6.
    Returns:
        list: A list of color tuples sampled from the colormap at regular intervals.
    """
    if c_numb > step:
        step = c_numb
    
    colors_arr = []
    for i in range(c_numb):
        colors_arr.append(cmap(i / step))
    
    return colors_arr
    
def line_plotter(xdata, ydata, ax=None, xlabel=r"x [-]", ylabel=r"y [-]", color="black",
                linestyle="-", xlim=None, ylim=None, equal_aspect=False, title=None, label=None, show=False,
                save_path=None, save_name=None):
    """
    Plot a line graph with customizable axes, limits, and styling.
    Parameters
    ----------
    xdata : array-like
        X-axis data points.
    ydata : array-like
        Y-axis data points.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, a new figure and axes are created.
        Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is "x [-]".
    ylabel : str, optional
        Label for the y-axis. Default is "y [-]".
    color : str, optional
        Line color. Default is "black".
    linestyle : str, optional
        Line style (e.g., "-", "--", "-.", ":"). Default is "-".
    xlim : tuple, list, or array-like, optional
        X-axis limits as [min, max]. If None, limits are set to data range.
        Default is None.
    ylim : tuple, list, or array-like, optional
        Y-axis limits as [min, max]. If None, axis auto-scales.
        Default is None.
    equal_aspect : bool, optional
        If True, sets equal aspect ratio for the plot. Default is False.
    title : str, optional
        Title for the plot. If None, no title is set. Default is None.
    label : str, optional
        Label for the line (used in legend). Default is None.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Raises
    ------
    ValueError
        If xlim or ylim are not in the correct format (list, dict, tuple, or numpy array).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    
    if equal_aspect:
        ax.set_aspect('equal')

    if xlim is not None:
        if isinstance(xlim, (list, dict, tuple, np.ndarray)):
            ax.set_xlim(xlim[0], xlim[1])
        else:
            print("\n\nWrong format of 'xlim'!\n")
    else:
        ax.set_xlim(min(xdata), max(xdata))

    if ylim is not None:
        if isinstance(ylim, (list, dict, tuple, np.ndarray)):
            ax.set_ylim(ylim[0], ylim[1])
        else:
            print("\n\nWrong format of 'ylim'!\n")

    ax.plot(xdata, ydata, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    # --------------------------------------------------
    # Save animation
    # --------------------------------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches="tight", format="png")
        
    if show:
        plt.show()

    return ax

def multi_line_plotter_same_axes(xdata_list, ydata_list, colors=None, linestyles=None, labels=None, 
                                  xlabel=r"x [-]", ylabel=r"y [-]",
                                  xlim=None, ylim=None, equal_aspect=False, title=None,
                                  legend=True, show=False, grid=False,
                                  save_path=None, save_name=None, IMG_CLOSE=True):
    """
    Plot multiple lines on the same axes with customizable styling.
    This function creates a single matplotlib figure with multiple line plots overlaid
    on the same axes. It supports customization of colors, line styles, labels, axis
    limits, and other plot properties.
    Parameters
    ----------
    xdata_list : list of array-like
        List of x-axis data arrays for each curve.
    xdata_list : list of array-like
        List of y-axis data arrays for each curve. Must have the same length as xdata_list.
    colors : list of str, optional
        List of color specifications for each line. If None, defaults to "black".
    linestyles : list of str, optional
        List of line style specifications for each line (e.g., "-", "--", "-.").
        If None, defaults to "-" (solid line).
    labels : list of str, optional
        List of labels for each line to be displayed in the legend. If None, no labels
        are shown.
    xlabel : str, optional
        Label for the x-axis. Default is "x [-]".
    ylabel : str, optional
        Label for the y-axis. Default is "y [-]".
    xlim : tuple of float, optional
        Limits for the x-axis as (min, max). If None, limits are determined automatically.
    ylim : tuple of float, optional
        Limits for the y-axis as (min, max). If None, limits are determined automatically.
    equal_aspect : bool, optional
        If True, sets equal aspect ratio for x and y axes. Default is False.
    title : str, optional
        Title for the plot. If None, no title is displayed. Default is None.
    legend : bool, optional
        If True and labels are provided, display a legend. Default is True.
    Returns
    -------
    None
        Displays the plot using plt.show().
    """    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    num_curves = len(xdata_list)
    
    for i in range(num_curves):
        color = colors[i] if colors is not None and i < len(colors) else "black"
        linestyle = linestyles[i] if linestyles is not None and i < len(linestyles) else "-"
        label = labels[i] if labels is not None and i < len(labels) else None

        line_plotter(xdata_list[i], ydata_list[i], ax=ax,
                       xlabel=xlabel, ylabel=ylabel,
                       color=color, linestyle=linestyle,
                       xlim=xlim, ylim=ylim,
                       equal_aspect=equal_aspect, title=title)
        
        # Dodaj label tylko, jeśli jest
        if label is not None:
            ax.plot(xdata_list[i], ydata_list[i], label=label, color=color, linestyle=linestyle)

    if legend and labels is not None:
        ax.legend()

    plt.tight_layout()
    if grid:
        plt.grid(True)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches="tight", format="png")
    if show:
        plt.show()

    plt.close(fig)
    gc.collect()

def animate_field_from_h5(
    h5_filename,
    load_h5data_path=None,
    save_name=None,
    save_path=None,
    dataset_name=None,
    interval=50,
    cmap="inferno",
    vmin=None,
    vmax=None,
    transpose_xy=False,
    IMG_CLOSE=False,
    xzeros=0,
    yzeros=None,
):
    """
    Animate time-dependent field data stored in an HDF5 file.

    FIXED DATA FORMAT
    -----------------
    The HDF5 dataset is assumed to have the layout:
        data[x, y, time]

    No axis reordering or transposition of the data array is performed.
    Any transpose is applied ONLY at the visualization level.

    Parameters
    ----------
    h5_filename : str
        Name of the HDF5 file containing field data.
        If `load_h5data_path` is provided, this is treated as a filename
        relative to that directory.

    dataset_name : str or None, optional
        Name of the dataset inside the HDF5 file.
        If None, the first dataset found in the file is used.

    interval : int, optional
        Delay between animation frames in milliseconds.

    cmap : str, optional
        Matplotlib colormap used for rendering the field.

    vmin, vmax : float or None, optional
        Color scale limits.
        If None, they are computed from the cropped data.

    transpose_xy : bool, optional
        If True, transpose X and Y axes for display purposes ONLY.
        This does NOT affect the underlying data array.

    save_path : str or None, optional
        Directory where the animation file should be saved.
        If None, the animation is not saved to disk.

    save_name : str or None, optional
        Name of the output animation file (e.g. "field.mp4").
        Used only if `save_path` is not None.
        If None and `save_path` is given, defaults to "<h5_filename>.mp4".

    load_h5data_path : str or None, optional
        Directory in which the HDF5 file is searched.
        If None, `h5_filename` is interpreted as a full or relative path.

    IMG_CLOSE : bool, optional
        If True, the matplotlib window is not displayed (useful for batch runs).

    xzeros : int, optional
        Number of grid points to crop from left and right boundaries (PML removal).

    yzeros : int or None, optional
        Number of grid points to crop from bottom and top boundaries.
        If None, defaults to `xzeros`.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The generated animation object.
    """
    
    # --------------------------------------------------
    # Resolve HDF5 file path
    # --------------------------------------------------
    if load_h5data_path is not None:
        h5_path = os.path.join(load_h5data_path, h5_filename)
    else:
        h5_path = h5_filename

    # print("\nOpen file\n")

    # --------------------------------------------------
    # OPEN FILE (NO "with"!)
    # --------------------------------------------------
    f = h5py.File(h5_path, "r")

    if dataset_name is None:
        dataset_name = list(f.keys())[0]

    data = f[dataset_name]

    if data.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {data.shape}")

    Nx, Ny, Nt = data.shape
    print("\n",data.shape,"\n")

    # --------------------------------------------------
    # Default yzeros
    # --------------------------------------------------
    if yzeros is None:
        yzeros = xzeros

    xzeros = max(0, min(xzeros, Nx // 2))
    yzeros = max(0, min(yzeros, Ny // 2))

    print("\nCrop\n")
    # --------------------------------------------------
    # Crop
    # --------------------------------------------------
    xs = slice(xzeros, Nx - xzeros)
    ys = slice(yzeros, Ny - yzeros)

    Nx = Nx - 2 * xzeros
    Ny = Ny - 2 * yzeros

    print("\nColor scale\n")
    # --------------------------------------------------
    # Color scale
    # --------------------------------------------------
    if vmin is None or vmax is None:
        sample = data[xs, ys, ::max(1, Nt // 20)]
        if vmin is None:
            vmin = np.min(sample)
        if vmax is None:
            vmax = np.max(sample)

    # --------------------------------------------------
    # Plot setup
    # --------------------------------------------------
    fig, ax = plt.subplots()

    frame0 = data[xs, ys, 0]
    if transpose_xy:
        frame0 = frame0.T

    img = ax.imshow(
        frame0,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    plt.colorbar(img, ax=ax)

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------
    def update(t):
        frame = data[xs, ys, t]

        if transpose_xy:
            frame = frame.T

        img.set_array(frame)
        ax.set_title(f"frame {t}/{Nt-1}")
        # print("\nFRAME: ", t, "\n")
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval,
        blit=True,
    )

    # --------------------------------------------------
    # Save animation
    # --------------------------------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        if save_name is None:
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            save_name = f"{base}.mp4"

        save_fullpath = os.path.join(save_path, save_name)
        anim.save(save_fullpath, writer="ffmpeg")

    # --------------------------------------------------
    # Show / close
    # --------------------------------------------------
    if not IMG_CLOSE:
        plt.show()

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
        
    plt.close(fig)
    f.close()
    del anim
    del img
    del frame0
    gc.collect()
    
    return 0

def animate_field_from_h5_physical(
    h5_filename,
    load_h5data_path=None,
    save_name=None,
    save_path=None,
    dataset_name=None,
    interval=50,
    cmap="inferno",
    vmin=None,
    vmax=None,
    transpose_xy=False,
    IMG_CLOSE=False,

    # --- physical axis definition ---
    x_phys_range=None,   # (xmin, xmax) e.g. (-2000, 2000)
    y_phys_range=None,   # (ymin, ymax)

    # --- PML / border crop ---
    xzeros=0,
    yzeros=None,

    # --- zoom (fraction of current size) ---
    x_zoom=1.0,
    y_zoom=1.0,

    # --- artifact killing ---
    mask_left=0,
    mask_right=0,
    mask_bottom=0,
    mask_top=0,

    # --- structure overlay ---
    structure=None,

    # --- labels ---
    title="Field enhancement |E|²",
    xlabel="X [nm]",
    ylabel="Y [nm]",
):
    """
    Animate 2D field data stored as data[x, y, time] with:
    - physical axes
    - centered zoom
    - optional structure overlay
    """

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    if x_phys_range is None or y_phys_range is None:
        raise ValueError("Provide x_phys_range and y_phys_range")

    h5_path = (
        os.path.join(load_h5data_path, h5_filename)
        if load_h5data_path is not None
        else h5_filename
    )

    # ---------------------------
    # OPEN FILE (no "with")
    # ---------------------------
    f = h5py.File(h5_path, "r")

    if dataset_name is None:
        dataset_name = list(f.keys())[0]

    dset = f[dataset_name]

    if dset.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {dset.shape}")

    Nx0, Ny0, Nt = dset.shape

    if yzeros is None:
        yzeros = xzeros

    # ---------------------------
    # CROPPING
    # ---------------------------
    xzeros = min(xzeros, Nx0 // 2)
    yzeros = min(yzeros, Ny0 // 2)

    xs0 = slice(xzeros, Nx0 - xzeros)
    ys0 = slice(yzeros, Ny0 - yzeros)

    Nx_c = Nx0 - 2 * xzeros
    Ny_c = Ny0 - 2 * yzeros

    # ---------------------------
    # ZOOM
    # ---------------------------
    cx, cy = Nx_c // 2, Ny_c // 2

    hx = int(0.5 * x_zoom * Nx_c)
    hy = int(0.5 * y_zoom * Ny_c)

    x1, x2 = cx - hx, cx + hx
    y1, y2 = cy - hy, cy + hy

    xs = slice(xzeros + x1, xzeros + x2)
    ys = slice(yzeros + y1, yzeros + y2)

    # ---------------------------
    # PHYSICAL AXES
    # ---------------------------
    x_min0, x_max0 = x_phys_range
    y_min0, y_max0 = y_phys_range

    x_full = np.linspace(x_min0, x_max0, Nx_c)
    y_full = np.linspace(y_min0, y_max0, Ny_c)

    x_phys = x_full[x1:x2]
    y_phys = y_full[y1:y2]

    extent = [x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]]

    # ---------------------------
    # INITIAL FRAME (ONE LOAD)
    # ---------------------------
    frame0 = dset[xs, ys, 0].copy()

    # masking
    if mask_left > 0:
        frame0[:mask_left, :] = 1.0
    if mask_right > 0:
        frame0[-mask_right:, :] = 1.0
    if mask_bottom > 0:
        frame0[:, :mask_bottom] = 1.0
    if mask_top > 0:
        frame0[:, -mask_top:] = 1.0

    if transpose_xy:
        frame0 = frame0.T

    # ---------------------------
    # COLOR SCALE (sampling)
    # ---------------------------
    if vmin is None or vmax is None:
        vmin, vmax = compute_vmin_vmax_streaming(dset, xs, ys, Nt)

    # ---------------------------
    # PLOT
    # ---------------------------
    fig, ax = plt.subplots()

    img = ax.imshow(
        frame0,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.colorbar(img, ax=ax)

    if structure is not None:
        if structure["type"] == "splitbar":
            draw_splitbar_outline(ax, structure["bars"])

    # ---------------------------
    # UPDATE
    # ---------------------------
    def update(t):
        frame = dset[xs, ys, t]

        # masking
        if mask_left > 0:
            frame[:mask_left, :] = 1.0
        if mask_right > 0:
            frame[-mask_right:, :] = 1.0
        if mask_bottom > 0:
            frame[:, :mask_bottom] = 1.0
        if mask_top > 0:
            frame[:, -mask_top:] = 1.0

        if transpose_xy:
            frame = frame.T

        img.set_array(frame)
        ax.set_title(f"frame {t}/{Nt-1}")

        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval,
        blit=True,
    )

    # --------------------------------------------------
    # Save animation
    # --------------------------------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_name is None:
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            save_name = f"{base}.mp4"
        anim.save(os.path.join(save_path, save_name), writer="ffmpeg")

    if not IMG_CLOSE:
        plt.show()

    # ---------------------------
    # CLEANUP
    # ---------------------------
    plt.close(fig)
    f.close()

    del anim
    del img
    del frame0
    gc.collect()

    return 0

def draw_splitbar_outline(
    ax,
    bars,
    linestyle=":",
    linewidth=1.5,
    alpha=0.5,
    color="white",
    zorder=10,
):
    """
    Draw dashed outlines of split-bar antenna.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.

    bars : list of dict
        Each dict must contain:
            - center : (x, y) in physical units
            - width  : bar length
            - height : bar thickness

    linestyle : str
        Matplotlib linestyle (default '--').

    linewidth : float
        Line width.

    alpha : float
        Transparency.

    color : str
        Line color.

    zorder : int
        Draw order (should be above field image).
    """

    for bar in bars:
        cx, cy = bar["center"]
        w = bar["width"]
        h = bar["height"]

        rect = Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)
        
def plot_field_frame_from_h5_physical(
    h5_filename,
    frame_index,
    load_h5data_path=None,
    dataset_name=None,
    cmap="inferno",
    vmin=None,
    vmax=None,
    transpose_xy=False,

    # --- physical axis definition ---
    x_phys_range=None,   # (xmin, xmax)
    y_phys_range=None,   # (ymin, ymax)

    # --- PML / border crop ---
    xzeros=0,
    yzeros=None,

    # --- zoom (fraction of current size) ---
    x_zoom=1.0,
    y_zoom=1.0,

    # --- artifact killing ---
    mask_left=0,
    mask_right=0,
    mask_bottom=0,
    mask_top=0,

    # --- structure overlay ---
    structure=None,

    # --- ROI for averaging ---
    roi=None,   # dict, e.g. {"type":"rectangle","center":(0,0),"width":20,"height":10}

    # --- plot text ---
    title=None,
    mean_prefix=r"|E|²/|E$_0$|² = ",
    mean_position=(0.02, 0.95),
    mean_color="white",
    mean_fontsize=12,

    # --- debug ---
    draw_roi=True,

    # --- misc ---
    IMG_CLOSE=False,

    # --- save ---
    save_path=None,
    save_name=None,

    # --- labels ---
    xlabel="X [nm]",
    ylabel="Y [nm]",
):
    """
    Plot a single 2D field frame with:
    - physical axes
    - crop & zoom
    - optional structure overlay
    - ROI-based averaging in physical coordinates
    """
    # ---------------------------
    # Sanity
    # ---------------------------
    if x_phys_range is None or y_phys_range is None:
        raise ValueError("Provide x_phys_range and y_phys_range")

    h5_path = (
        os.path.join(load_h5data_path, h5_filename)
        if load_h5data_path is not None
        else h5_filename
    )

    # ---------------------------
    # OPEN FILE
    # ---------------------------
    f = h5py.File(h5_path, "r")

    if dataset_name is None:
        dataset_name = list(f.keys())[0]

    dset = f[dataset_name]

    if dset.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {dset.shape}")

    Nx0, Ny0, Nt = dset.shape

    if frame_index < 0 or frame_index >= Nt:
        raise ValueError(f"frame_index must be in [0, {Nt-1}]")

    if yzeros is None:
        yzeros = xzeros

    # ---------------------------
    # CROPPING
    # ---------------------------
    xzeros = min(xzeros, Nx0 // 2)
    yzeros = min(yzeros, Ny0 // 2)

    Nx_c = Nx0 - 2 * xzeros
    Ny_c = Ny0 - 2 * yzeros

    # ---------------------------
    # ZOOM
    # ---------------------------
    cx, cy = Nx_c // 2, Ny_c // 2
    hx = int(0.5 * x_zoom * Nx_c)
    hy = int(0.5 * y_zoom * Ny_c)

    x1, x2 = cx - hx, cx + hx
    y1, y2 = cy - hy, cy + hy

    xs = slice(xzeros + x1, xzeros + x2)
    ys = slice(yzeros + y1, yzeros + y2)

    # ---------------------------
    # PHYSICAL AXES
    # ---------------------------
    x_min0, x_max0 = x_phys_range
    y_min0, y_max0 = y_phys_range

    x_full = np.linspace(x_min0, x_max0, Nx_c)
    y_full = np.linspace(y_min0, y_max0, Ny_c)

    x_phys = x_full[x1:x2]
    y_phys = y_full[y1:y2]

    extent = [x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]]

    # ---------------------------
    # LOAD SINGLE FRAME
    # ---------------------------
    frame_raw = dset[xs, ys, frame_index].copy()

    # ---------------------------
    # MASKING
    # ---------------------------
    if mask_left > 0:
        frame_raw[:mask_left, :] = 1.0
    if mask_right > 0:
        frame_raw[-mask_right:, :] = 1.0
    if mask_bottom > 0:
        frame_raw[:, :mask_bottom] = 1.0
    if mask_top > 0:
        frame_raw[:, -mask_top:] = 1.0

    # ---------------------------
    # PLOT ARRAY
    # ---------------------------
    frame_plot = frame_raw.T if transpose_xy else frame_raw

    # ---------------------------
    # ROI mean
    # ---------------------------
    mean_val = np.nan
    roi_mask = None

    if roi is not None:
        if roi["type"] == "rectangle":
            roi_mask = roi_mask_from_rectangle(
                x_phys,
                y_phys,
                center=roi["center"],
                width=roi["width"],
                height=roi["height"],
            )
            mean_val = np.mean(frame_raw[roi_mask])

    # ---------------------------
    # SAVE MEAN/MAX
    # ---------------------------
    max_val = np.max(frame_raw)

    if save_path is not None:

        plane = dataset_name

        dat_name = f"fef_{plane}.dat"

        dat_path = os.path.join(
            save_path,
            dat_name
        )

        # append mode
        with open(dat_path, "a") as f:

            # header tylko raz
            if os.path.getsize(dat_path) == 0:

                f.write(f"# {plane}\n")
                f.write("# meanval maxval\n")

            f.write(
                f"{mean_val:.6g} {max_val:.6g}\n"
            )

    # ---------------------------
    # COLOR SCALE
    # ---------------------------
    if vmin is None:
        vmin = np.min(frame_raw)
    if vmax is None:
        vmax = np.max(frame_raw)

    # ---------------------------
    # PLOT
    # ---------------------------
    fig, ax = plt.subplots()

    img = ax.imshow(
        frame_plot,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    plt.colorbar(img, ax=ax)

    # ---------------------------
    # STRUCTURE
    # ---------------------------
    if structure is not None:
        if structure["type"] == "splitbar":
            draw_splitbar_outline(ax, structure["bars"])

    # ---------------------------
    # ROI overlay
    # ---------------------------
    if roi is not None and draw_roi:
        cx, cy = roi["center"]
        w = roi["width"]
        h = roi["height"]

        roi_rect = Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor="cyan",
            linestyle=":",
            linewidth=0.8,
            zorder=11,
        )
        ax.add_patch(roi_rect)

    # ---------------------------
    # TEXT
    # ---------------------------
    if roi is not None:
        text = f"{mean_prefix}{mean_val:.3g}"
        ax.text(
            mean_position[0],
            mean_position[1],
            text,
            transform=ax.transAxes,
            color=mean_color,
            fontsize=mean_fontsize,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )

    # ---------------------------
    # SAVE
    # ---------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_name is None:
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            save_name = f"{base}.png"
        plt.savefig(
            os.path.join(save_path, save_name),
            dpi=300,
            bbox_inches="tight",
        )

    # ---------------------------
    # SAVE RAW FRAME
    # ---------------------------
    if save_path is not None:

        base = os.path.splitext(
            os.path.basename(h5_filename)
        )[0]

        raw_name = (
            f"{base}_frame_{frame_index}.npz"
        )

        np.savez(
            os.path.join(save_path, raw_name),

            field=frame_plot,

            x=x_phys,

            y=y_phys,
        )

        print(
            "Saved raw frame:",
            os.path.join(save_path, raw_name)
        )

    # ---------------------------
    # CLEANUP
    # ---------------------------
    if not IMG_CLOSE:
        plt.show()

    plt.close(fig)
    f.close()

    del frame_raw
    gc.collect()

    return mean_val

def roi_mask_from_rectangle(x_phys, y_phys, center, width, height):
    """
    Create boolean mask for a rectangular ROI in physical coordinates.

    Parameters
    ----------
    x_phys, y_phys : 1D numpy arrays
        Physical coordinates corresponding to data grid (after crop & zoom).

    center : tuple (cx, cy)
        Center of ROI in physical units.

    width, height : float
        ROI size in physical units.

    Returns
    -------
    mask : 2D boolean numpy array [Nx, Ny]
    """

    cx, cy = center
    half_w = width / 2
    half_h = height / 2

    X, Y = np.meshgrid(x_phys, y_phys, indexing="ij")

    mask = (
        (X >= cx - half_w) & (X <= cx + half_w) &
        (Y >= cy - half_h) & (Y <= cy + half_h)
    )

    return mask


def show_data_img(datas_arr, abs_bool, norm_bool, cmap_arr, alphas, name_to_save=None, IMG_CLOSE=False, Title=None, disable_ticks=True, log10_scale=False):
    """
    Displays a series of images from a given array of data.

    Parameters:
    datas_arr (list of np.ndarray): A list of 2D arrays containing the data to be visualized.
    norm_bool (list of bool): A list of boolean values indicating whether to normalize each corresponding data array.
    cmap_arr (list of str): A list of colormap names to be used for each corresponding data array.
    alphas (list of float): A list of alpha values for transparency for each corresponding data array.

    The function iterates through the provided data arrays, normalizes them if specified, 
    and displays each image using matplotlib's imshow function with the specified colormap 
    and transparency settings. The x and y axis ticks are turned off for a cleaner visualization.
    """
    for idx, data in enumerate(datas_arr):
        if abs_bool[idx]:
            data = np.abs(data) # complex -> real
        if norm_bool[idx]:
            max_data = np.max(data)
            data = data / max_data # normalization
        if log10_scale:
            data = np.log10(data + 1e-12) # log scale with small offset to avoid log(0)
        plt.imshow(data.transpose(), interpolation="spline36", cmap=cmap_arr[idx], alpha=alphas[idx])
        plt.colorbar(shrink=0.6)  # Show color scale
    if disable_ticks:
        plt.xticks([])  # Turn off x-axis numbers
        plt.yticks([])  # Turn off y-axis numbers
    if Title is not None:
        plt.title(Title)
    if name_to_save is not None:
        plt.savefig(f"{name_to_save}.png", dpi=300, bbox_inches="tight", format="png")
    if IMG_CLOSE:
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    else:
        plt.show()

def save_2D_plot(sim, volume, save_name="2Dplot.png", IMG_SAVE=True, path_to_save=None, IMG_CLOSE=False, config=None):
    if meep.am_master():
        sim.plot2D(output_plane=volume,
                eps_parameters={'alpha':0.8, 'cmap':'binary', 'interpolation':'spline36', 'frequency':1/0.2},
                boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
        # sim.plot2D() labels the axes "X"/"Y"/"Z" with no unit. Meep lengths
        # are in units of a = 1 um (xm = 1000 in the experiment modules), so the
        # numbers already ARE micrometres - only the label was missing. Derive
        # the pair from the plane name in save_name; fall back to leaving the
        # Meep labels alone when the name says nothing.
        ax = plt.gca()
        plane = next((p for p in ("XY", "XZ", "YZ", "XY", "YX")
                      if save_name and p in save_name), None)
        if plane:
            ax.set_xlabel(rf"{plane[0]} [$\mu$m]")
            ax.set_ylabel(rf"{plane[1]} [$\mu$m]")

        if config is not None:
            if save_name is not None and any(axis in save_name for axis in ("XZ", "YZ")):
                plt.hlines(config.z_reflection, -config.cell_size[0]/2.0, config.cell_size[0]/2.0, color='blue', linestyle='--', linewidth=1.0)
                plt.hlines(config.z_transmission, -config.cell_size[0]/2.0, config.cell_size[0]/2.0, color='red', linestyle='--', linewidth=1.0)
                plt.text(-config.cell_size[0]/2.1, config.z_reflection*1.1, 'R', color='blue', fontsize=12)
                plt.text(-config.cell_size[0]/2.1, config.z_transmission*0.9, 'T', color='red', fontsize=12)

        if IMG_SAVE:
            plt.savefig(os.path.join(path_to_save, save_name), dpi=300, bbox_inches="tight", format="png")
        if IMG_CLOSE:
            # plt.show(block=False)
            # plt.pause(2)
            plt.close("all")
        else:
            plt.show()
    return 0

def draw_dielectric_constant(sim, config, visvol, sampling_wavelength=None, log10_scale=False):
    """
    Generate dielectric constant maps in XY, XZ and YZ planes.

    For each plane (XY, XZ, YZ):
    - the dielectric constant is extracted using sim.get_array(),
    - a 2D map is plotted and saved to disk,
    - the wavelength is included in the plot title and filename.

    Parameters
    ----------
    sampling_wavelength : float or None
        Wavelength in nm at which ε is sampled.
        If None, the default simulation wavelength is used.

    log10_scale : bool
        If True, apply log10 scaling to the plotted dielectric map.

    Returns
    -------
    int
        Returns 0 after successful execution.
    """
    if meep.am_master():
        sim.run(until=0)  # Run for 0 time to initialize the fields and materials

        if sampling_wavelength is not None:
            wavelength = sampling_wavelength
            sampling_wavelength = sampling_wavelength / 1000  # Convert nm to um
            frequency = 1 / sampling_wavelength
        else:
            wavelength = config.lambda0*1e3  # Convert um to nm for title
            frequency = config.frequency

        # ============================================================
        # Plane configuration
        # ============================================================

        planes = {
            "XY": {
                "volume": visvol.vis_volume["XY"],
                "title": f"Dielectric constant in XY plane\nWavelength {int(wavelength)} nm",
                "save_name": f"dielectric_XY_plane_{int(wavelength)}nm.png",
            },
            "XZ": {
                "volume": visvol.vis_volume["XZ"],
                "title": f"Dielectric constant in XZ plane\nWavelength {int(wavelength)} nm",
                "save_name": f"dielectric_XZ_plane_{int(wavelength)}nm.png",
            },
            "YZ": {
                "volume": visvol.vis_volume["YZ"],
                "title": f"Dielectric constant in YZ plane\nWavelength {int(wavelength)} nm",
                "save_name": f"dielectric_YZ_plane_{int(wavelength)}nm.png",
            },
        }

        # ============================================================
        # Iteration over planes
        # ============================================================

        for plane, cfg in planes.items():
            print(f"Processing dielectric map for {plane} plane")

            eps_data = sim.get_array(
                vol=cfg["volume"],
                frequency=frequency,
                component=meep.Dielectric,
            )

            show_data_img(
                datas_arr=[eps_data],
                abs_bool=[True],
                norm_bool=[True],
                cmap_arr=["binary"],
                alphas=[1.0],
                IMG_CLOSE=config.IMG_CLOSE,
                Title=cfg["title"],
                disable_ticks=False,
                name_to_save=os.path.join(config.path_to_save, cfg["save_name"]),
                log10_scale=log10_scale,
            )
        sim.reset_meep()
    return 0

def animate_raw_fields(
    config,
    mode="BOTH",
    animate_E=True,
    animate_H=False,
    animate_DPWR=False,
    component="X",
    extras=False,
):
    """
    Generate animations for fields map.

    Parameters
    ----------
    mode : str
        "WITH_ANTENNA", "EMPTY", or "BOTH"

    animate_E : bool
        Animate E-field component.

    animate_H : bool
        Animate H-field component.

    animate_DPWR : bool
        Animate power density field.

    component : str
        Field component: "X", "Y", or "Z".
    """
    if meep.am_master():
        valid_modes = ["WITH_ANTENNA", "EMPTY", "BOTH"]
        valid_components = ["X", "Y", "Z"]

        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")

        if component not in valid_components:
            raise ValueError(f"component must be one of {valid_components}")

        comp = component.lower()

        planes = [
            "xyplanar",
            "xyplanarTOP",
            "xzplanar",
            "yzplanar",
        ]

        if extras:
            planes.extend([
                "xyplanar_1",
                "xyplanar_2",
                "xyplanar_3",
                "xyplanar_4",
                "xyplanar_5",
                "xzplanar_1",
                "xzplanar_2",
                "xzplanar_3",
                "xzplanar_4",
                "xzplanar_5",
                "yzplanar_1",
                "yzplanar_2",
                "yzplanar_3",
                "yzplanar_4",
                "yzplanar_5",
            ])

        # ============================================================
        # FUNCTION TO ANIMATE SINGLE FILE
        # ============================================================

        def animate_file(filename):
            print(f"Animating {filename}")
            animate_field_from_h5(
                h5_filename=filename,
                save_name=filename.replace(".h5", ".mp4"),
                load_h5data_path=config.path_to_save,
                save_path=config.animations_folder_path,
                transpose_xy=True,
                cmap="RdBu",
                IMG_CLOSE=config.IMG_CLOSE,
            )

        # ============================================================
        # WITH ANTENNA
        # ============================================================

        if mode in ["WITH_ANTENNA", "BOTH"]:
            print("Animating WITH antenna")

            for plane in planes:

                if animate_E:
                    animate_file(f"{plane}_e{comp}.h5")

                if animate_H:
                    animate_file(f"{plane}_h{comp}.h5")

                if animate_DPWR:
                    animate_file(f"{plane}_dpwr.h5")

        # ============================================================
        # EMPTY
        # ============================================================

        if mode in ["EMPTY", "BOTH"]:
            print("Animating EMPTY structure")

            for plane in planes:

                if animate_E:
                    animate_file(f"{plane}-empty_e{comp}.h5")

                if animate_H:
                    animate_file(f"{plane}-empty_h{comp}.h5")

                if animate_DPWR:
                    animate_file(f"{plane}-empty_dpwr.h5")
    return 0

def plot_signal_amplitude_vs_time_from_h5(
    h5_filename,
    load_h5data_path=None,
    dataset_name=None,
    xzeros=0,
    yzeros=None,
    time_step=1.0,
    mode="BOTH",
    normalize=False,
    save_name=None,
):
    """
    Plotting source amplitude spectrum in time.

    Parameters
    ----------
    mode : str
        "E", "E2", or "BOTH"
    """

    # ------------------------------
    # resolve path
    # ------------------------------
    if load_h5data_path is not None:
        h5_path = os.path.join(load_h5data_path, h5_filename)
    else:
        h5_path = h5_filename

    print("Opening:", h5_path)

    # ------------------------------
    # open file
    # ------------------------------
    with h5py.File(h5_path, "r") as f:

        if dataset_name is None:
            dataset_name = list(f.keys())[0]

        dset = f[dataset_name]

        Nx, Ny, Nt = dset.shape

        print("Dataset shape:", dset.shape)

        if yzeros is None:
            yzeros = xzeros

        mean_E = np.zeros(Nt)
        mean_E2 = np.zeros(Nt)
        mean_E_val2 = np.zeros(Nt)

        # ------------------------------
        # iterate over frames
        # ------------------------------
        for t in range(Nt):

            frame = dset[:, :, t]

            frame = frame[
                xzeros:Nx-xzeros,
                yzeros:Ny-yzeros
            ]

            mean_E[t] = np.mean(frame)
            mean_E2[t] = np.mean(frame**2)
            mean_E_val2[t] = mean_E[t]**2

            if t % 50 == 0:
                print(f"frame {t}/{Nt}")
                
    # ------------------------------
    # time axis
    # ------------------------------
    time = np.arange(Nt) * time_step

    # ------------------------------
    # save data
    # ------------------------------
    if save_name is None:
        save_name = "source_profil"

    data2save = np.column_stack((time, mean_E, mean_E2))

    np.savetxt(
        os.path.join(load_h5data_path, save_name+".dat"),
        data2save,
        header="TIME E E2 meanE^2",
        comments="# ",
        fmt="%.3e"
    )

    # ------------------------------
    # normalization
    # ------------------------------
    if normalize:
        mean_E = mean_E / max(mean_E)
        mean_E2 = mean_E2 / max(mean_E2)

    # ------------------------------
    # plot
    # ------------------------------
    multi_line_plotter_same_axes(
        [time, time, time],
        [mean_E, mean_E2, mean_E_val2],
        colors = ["tab:red", "k", "tab:orange"],
        linestyles = ["-", "-.", ":"],
        labels = ["<E>", "<E²>", "<E>²"],
        xlabel = "Time",
        ylabel = "Amplitude",
        save_path = load_h5data_path,
        save_name = save_name+".png",
    )

    return time, mean_E2, mean_E2

def compute_vmin_vmax_streaming(dset, xs, ys, Nt):
    """
    Compute exact vmin and vmax by iterating over frames (no full load).

    Parameters
    ----------
    dset : h5py.Dataset
        Dataset with shape (x, y, time)

    xs, ys : slice
        Spatial slices (after crop / zoom)

    Nt : int
        Number of time frames

    Returns
    -------
    vmin, vmax : float
    """

    import numpy as np

    vmin = np.inf
    vmax = -np.inf

    for t in range(Nt):
        frame = dset[xs, ys, t]

        fmin = frame.min()
        fmax = frame.max()

        if fmin < vmin:
            vmin = fmin
        if fmax > vmax:
            vmax = fmax

    return vmin, vmax
