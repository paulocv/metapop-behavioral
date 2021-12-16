"""Tools for plot styling - creation of style sheets."""

from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
import numpy as np
import os

from functools import reduce

from toolbox.file_tools import SEP, write_config_string, list_to_csv

_STD_USETEX = False


# ------------------------------------------
# CUSTOM STYLES
# ------------------------------------------

# A nice style for journal plots, with large fonts, minor ticks,
# a latex font much better than default, and others.
# Based on a style from Luiz Alves.
mystyle_01 = {
    # Latex
    "text.usetex": _STD_USETEX,
    "text.latex.preamble": [r"\usepackage[T1]{fontenc}",
                            r"\usepackage{lmodern}",
                            r"\usepackage{amsmath}",
                            r"\usepackage{mathptmx}"
                            ],
    # Axes configuration
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "ytick.right": "on",  # Right and top axis included
    "xtick.top": "on",
    "xtick.labelsize": "25",
    "ytick.labelsize": "25",
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.8,
    "xtick.minor.width": 1.8,
    "xtick.major.size": 14,
    "xtick.minor.size": 7,
    "xtick.major.pad": 10,
    "xtick.minor.pad": 10,
    "ytick.major.width": 1.8,
    "ytick.minor.width": 1.8,
    "ytick.major.size": 14,
    "ytick.minor.size": 7,
    "ytick.major.pad": 10,
    "ytick.minor.pad": 10,
    "axes.labelpad": 15,
    "axes.titlepad": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,  # Includes minor ticks
    "ytick.minor.visible": True,

    # Lines and markers
    "lines.linewidth": 4,
    "lines.markersize": 9,
    "lines.markeredgecolor": "k",  # Includes marker edge
    'errorbar.capsize': 4.0,

    # Legend
    "legend.numpoints": 2,  # Uses two points as a sample
    "legend.fontsize": 20,
    "legend.framealpha": 0.75,

    # Font
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica"],
    "font.size": 20,
    "mathtext.fontset": "cm",  # Corrects the horrible sans font for latex

    # Figure
    "figure.figsize": (9.1, 7.),
}

mystyle_01_docs = "A nice style for journal plots, with large fonts, minor ticks, "
mystyle_01_docs += "a latex font much better than default, and others. "
mystyle_01_docs += "Based on a style from Luiz Alves."


# --------------------
# Redesign of mystyle_01, to look slightly more modern.
# Smaller ticks and a Times-like font.
mystyle_02 = {
    # Latex
    "text.usetex": _STD_USETEX,
    "text.latex.preamble": [r"\usepackage[T1]{fontenc}",
                            r"\usepackage{lmodern}",
                            r"\usepackage{amsmath}",
                            r"\usepackage{mathptmx}"
                            ],

    # Font params
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "axes.labelpad": 15,
    "axes.titlepad": 15,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,

    # Axis config
    "axes.linewidth": 1.8,  # For axis frame, not for plotted lines

    # Axis ticks
    "ytick.right": "on",  # Right and top axis included
    "xtick.top": "on",
    "xtick.major.width": 1.6,
    "xtick.minor.width": 1.6,
    "xtick.major.size": 7,
    "xtick.minor.size": 3.5,
    "xtick.major.pad": 10,
    "xtick.minor.pad": 10,
    "ytick.major.width": 1.6,
    "ytick.minor.width": 1.6,
    "ytick.major.size": 7,
    "ytick.minor.size": 3.5,
    "ytick.major.pad": 10,
    "ytick.minor.pad": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,  # Includes minor ticks
    "ytick.minor.visible": True,

    # Lines and markers
    "lines.linewidth": 4,
    "lines.markersize": 9,
    "lines.markeredgecolor": (0.1, 0.1, 0.1),  # Dark gray for the edges of markers
    "lines.markeredgewidth": 1,
    'errorbar.capsize': 4.0,

    # Font
    "font.family": "serif",
    "font.serif": ["FreeSerif", "Liberation Serif"],
    "mathtext.fontset": "cm",  # Corrects the horrible sans font for latex math

    # Legend
    "legend.numpoints": 2,  # Uses two points as a sample
    "legend.fontsize": 18,
    "legend.title_fontsize": 22,
    "legend.framealpha": 0.75,
}

mystyle_02_docs = "Another nice style for journal plots, with large fonts, minor ticks, "
mystyle_02_docs += "a latex font much better than default, and others. "
mystyle_02_docs += "Redesigned from mystyle_01 to look more modern."


def create_mpl_style(name, style_dict, convert_lists=True, docstring=None):
    """Creates a .mplstyle file in the matplotlib styles folder.

    If using from ipython, you have to restart the kernel for the changes to have effect.
    Then load the style with plt.style.use(name).

    Parameters
    ----------
    name : str
        Name of the style and its file name.
    style_dict : dict
        Dictionary with matplotlib rcParams.
    convert_lists : bool
        Converts lists and tuples to csv, so they are understood by matplotlib.
        Default is True.
    docstring : str
        A comment about the style.
    """

    # Gets the matplotlib config dir and possibly creates the style subdir
    mpl_style_dir = mpl.get_configdir() + SEP + "stylelib"
    if not os.path.exists(mpl_style_dir):
        os.system("mkdir {}".format(mpl_style_dir))

    # Makes some modifications, but preserves the original dict
    style_dict = style_dict.copy()

    # Converts lists to csv, which is how mpl understands
    if convert_lists:
        for key, val in style_dict.items():
            if type(val) in [list, tuple]:
                style_dict[key] = list_to_csv(val)

    # Exports the dict to file
    style_text = write_config_string(style_dict, entry_char="", attribution_char=":")
    with open(mpl_style_dir + SEP + name + ".mplstyle", "w") as fp:
        if docstring is not None:
            fp.write("# " + docstring + "\n")
        fp.write(style_text)


# ------------------------
# FIGURE AND AXIS CREATION
# ------------------------

def make_axes_seq(num_axes, max_cols=3, total_width=9., ax_height=3.):
    """Creates a sequence of num_axes axes in a figure.
    The axes are periodically disposed into rows of at most max_cols elements.
    Exceeding axes in the last row are removed
    """
    # Basic dependent numbers
    num_rows = (num_axes - 1) // max_cols + 1

    # Empty figure initialization and gridspec object (divides space into grids).
    fig = plt.figure(figsize=(total_width, num_rows * ax_height))
    gridspecs = fig.add_gridspec(num_rows, max_cols)

    # Creates the list of axes with the required number of axes
    axes = [fig.add_subplot(gridspecs[i]) for i in range(num_axes)]

    return fig, axes


def stdfigsize(scale=1, nrows=1, ncols=1, xtoy_ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns
    By default: ratio=1.3
    If ratio<0 then ratio = golden ratio
    """
    if xtoy_ratio < 0:
        xtoy_ratio = 1.61803398875

    return 7 * xtoy_ratio * scale * ncols, 7 * scale * nrows


# ---------------------------
# Color, linestyle and other sequences
# ---------------------------


# Aux function: least common multiple.
def _lcm(x, y):
    """Least common multiple via brute (incremental) search."""
    if x > y:
        z = x
    else:
        z = y

    while True:
        if(z % x == 0) and (z % y == 0):
            lcm = z
            break
        z += 1

    return lcm


# Qualitative printer friendly only color seqs
colorbrewer_pf_01 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                     '#a65628', '#f781bf', '#999999']
colorbrewer_pf_02 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
colorbrewer_pf_03 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
                     '#b3de69', '#fccde5']

# Colorblind friendly only color seqs
colorbrewer_cbf_01 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
colorbrewer_cbf_02 = ['#1b9e77', '#d95f02', '#7570b3']
colorbrewer_cbf_03 = ['#66c2a5', '#fc8d62', '#8da0cb']

# Matplotlib modern standard
default_colorlist = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def set_color_cycle(colors):
    """Warning: this will only setup the colors, reseting all other cyclic
    properties.

    To set up a property without affecting the other, implement
    change_prop_cycle function.
    """
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


def set_composite_prop_cycle(**props):
    """Sets the prop cycle to combinations of properties from seqs that
    are not necessarily commensurable.
    Each property is advanced at each new plot, unlike
    set_product_prop_cycle.

    Arguments are keywords as prop_name=prop_list.
    """
    # lcm()?? # Current function is for 2 values only.

    # Gets the multiplication factor for each prop sequence
    lens = [len(vals) for vals in props.values()]
    total_len = reduce((lambda x, y: x * y), lens)  # Product of each element
    mult_facs = [total_len // d for d in lens]

    # Creates the composite cycler
    comp_props = {key: n * list(vals)
                  for n, (key, vals) in zip(mult_facs, props.items())}
    prop_cycle = cycler(**comp_props)

    # Sets it to mpl and returns
    plt.rcParams["axes.prop_cycle"] = prop_cycle
    return prop_cycle


def set_product_prop_cycle(**props):
    """Sets the prop cycle to combinations of properties from seqs that
    are not necessarily commensurable.
    The properties defined at last will cycle first ("faster"), followed
    by the next and so on.

    Arguments are keywords as prop_name=prop_list.
    """
    # Composite prop cycler
    cycle_list = []
    for key, vals in props.items():
        cycle_list.append(cycler(**{key: vals}))

    prop_cycle = reduce((lambda x, y: x * y), cycle_list)  # Product

    # Sets it to mpl and returns
    plt.rcParams["axes.prop_cycle"] = prop_cycle
    return prop_cycle


# ------------------------
# ETC
# ------------------------


def get_available_font_names():
    """Returns a list with fonts currently recognized by matplotlib"""
    return [f.name for f in mpl.font_manager.fontManager.ttflist]


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    By users Eric and thomas from stackoverflow
    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib

    Add an arrow to a line, showing the "direction" of the plot (when it makes sense).

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
