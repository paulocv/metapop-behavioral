"""
Plot features of each node.
THIS IS A CONTENT MODULE, not meant to be directly called.
"""

# %matplotlib notebook
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# import os
import time

# [NXVERSION]
from sim_modules.nx2_version_bkp.metapopulation import *  # nx2
# from sim_modules.metapopulation import *  # nx1

from toolbox.file_tools import *
from toolbox.plot_tools import *
from toolbox.network_gen_tools import load_network_with_data
import warnings
# from toolbox.weighted_distances import *

SAVE_PLOT = False

# Modules:
# Load a single simulation/execution file.
# Load the network file. Load also it from a given sim file.
# Calculates the global metrics, the node population series and the fraction series.
#


# --------------------------------------
# MANUAL INPUTS

# - - - - S E E   M A I N ( ) - - - - - - - - -

# ---------------------------------------------------------------
# BASIC AUX FUNCTIONS
# ---------------------------------------------------------------
# If some are proven useful, could be moved to a toolbox module.

def dict_to_array(d, keylist, dtype=object):
    return np.fromiter((d[key] for key in keylist), dtype=dtype)


def array_to_dict(a, keylist):
    return {key: x for key, x in zip(keylist, a)}


def gen_sim_fname(prefix, state):
    return "{}_{}.out".format(prefix, state)


def gen_exec_fname(prefix, state, i_exec):
    return "{}_{}_exec{:04d}.out".format(prefix, state, i_exec)


def gen_netw_path(prefix):
    return prefix + ".csv", prefix + ".edgl"


def reorder_array(array, old_ids, new_ids):
    """Reorders a numpy array based on two lists of ids.
    The old_ids contains the ids of the elements currently in array.
    new_ids is the desired order. Elements and sizes must match!
    """
    if type(old_ids) not in [list, tuple]:
        old_ids = list(old_ids)  # Needs to have the index method
    permut = [old_ids.index(ni) for ni in new_ids]
    return array[permut]


def global_response_term(rho_i, rho_r, k, lt):
    return (1 - rho_i - lt * rho_r)**k


def distance_shell_layout(g, source):
    """Uses networkx's shell layout to make concentric circles ordered
    by the distance to source node.
    """
    dists = nx.shortest_path_length(g, source)
    max_d = max(dists.values())

    nlist = [[] for _ in range(max_d + 1)]  # List of node lists

    for ni, d in dists.items():
        nlist[d].append(ni)

    return nx.shell_layout(g, nlist)


# -----------------------------------------------------------------------------
# SIMULATION AND METAPOPULATION LOADING
# -----------------------------------------------------------------------------

def load_sim_file(fname, unpack=True, read_header=True, header_size=None):

    # Checks if file is zipped only, unzipping if so.
    zipped = possibly_unzip_file(fname)

    if read_header:
        header_size = count_header_lines(fname)
        input_dict = read_config_file(fname)
    else:
        input_dict = None
        if header_size is None:
            raise ValueError("Hey, function 'load_sim_file' was called with read_header=False but "
                             "parameter header_size, which is mandatory in this case, is None. Please "
                             "inform header_size so np.load_txt knows how many lines to skip. Thx.")

    data = np.loadtxt(fname, skiprows=header_size, unpack=unpack)

    if zipped:
        remove_file(fname)

    return data, input_dict


def load_metapop_from_input_dict(input_dict, path_prefix=""):
    """Loads a network based on the entries of an input dict.
    If the current directory is not the project root, from which network paths
    are given in sim file, you can add a prefix that brings to the root folder.
    """
    try:
        net_path_prefix = input_dict["pop_path"]
        net_node_path, net_edge_path = gen_netw_path(net_path_prefix)
    except KeyError:
        net_node_path = input_dict["pop_node_path"]
        net_edge_path = input_dict["pop_edge_path"]
    return Metapop(load_network_with_data(path_prefix + net_node_path, path_prefix + net_edge_path), nodetype=int)


def get_seed_node(input_dict):
    init_mode = input_dict["init_mode"]
    init_data = input_dict["init_data"]

    if init_mode == "infected_dict":
        seeds = str_to_dict(init_data)
        if len(seeds) > 1:
            warnings.warn("Hey, multiple seed nodes! Will use the ''first'' key from init_data,"
                          " which is not well defined.")
        seed = list(seeds.keys())[0]
    else:
        raise ValueError("Hey, sorry, I still do not recognize init_mode '{}'"
                         "".format(init_mode))
    return seed


def get_nodelists_by_shortestpath(g, source):
    dists = nx.shortest_path_length(g, source)
    max_d = max(dists.values())

    nlist = [[] for _ in range(max_d + 1)]  # List of node lists

    for ni, d in dists.items():
        nlist[d].append(ni)

    return nlist


class SimBunch:
    """A bunch for all the time series and features read from a simulation set of files."""

    __slots__ = ["t_array", "count_series", "pop_series", "frac_series", "a_series", "total_frac_series",
                 "total_a_series", "feat", "nodes", "g"]

    def __init__(self, t_array, count_series, pop_series, frac_series, a_series,
                 total_frac_series, total_a_series, feat, nodes, g):
        self.t_array = t_array
        self.count_series = count_series
        self.pop_series = pop_series
        self.frac_series = frac_series
        self.a_series = a_series
        self.total_frac_series = total_frac_series
        self.total_a_series = total_a_series
        self.feat = feat
        self.nodes = nodes
        self.g = g

    def quickplot(self, states=("I", "R"), plot_a=True, figsize_ax=None):
        return quickplot(self, states, plot_a, figsize_ax=figsize_ax)


def load_and_process_sim_data(sim_prefix, states=None, i_exec=None, read_a_array=True,
                              g=None, nodes=None):
    """
    Reads all data from a simulation set of files. Extracts several features,
    returning all time series (population, counts and fractions) as well as a
    bunch with sim features.

    EDIT: variable states now mandatorily infered from files, but kept for compatibility.
    Shall be deprecated.

    Adapted from previous jupyter code.

    [10/11/2020] - Backed up at "test_programs/bkp_load_and_process_sim_data.py" before
        optimization attempts.

    Returns
    -------
    SimBunch
    """

    # -----------------------
    # Sim data retrieval

    # Bunch of misc outputs
    feat = {}
    # Time series
    count_series = {}  # Signature: sim_dict[state][ni][i_t]

    # Choose between single execution or average simulation
    if i_exec is None:
        fname_func = gen_sim_fname
    else:
        def fname_func(prefix, s):
            return gen_exec_fname(prefix, s, i_exec)

    # Takes data from a sample file (uses "S" file for that)
    sample_fname = fname_func(sim_prefix, "S")  # A file used for initial reading
    zipped = possibly_unzip_file(sample_fname)
    sample_header = read_file_header(sample_fname)
    header_size = len(sample_header) + 1  # Assumed to be the same for all sim files through the rest of the function
    input_dict = read_config_strlist(sample_header)
    model_class = input_dict["model_class"]
    if zipped:
        remove_file(sample_fname)

    # Deduces all model states from the model class. Necessary to calculate node instant populations.
    feat["model_class"] = model_class
    if model_class == "MetapopSEIR":
        states = "SEIR"
    elif model_class == "MetapopSIR":
        states = "SIR"
    else:
        raise ValueError("Hey, include here all states for model class '{}'.".format(
            model_class
        ))

    # x_t0 = time.time()
    # STATECOUNT DATA RETRIEVAL
    # Optimization note: this seems to be an expensive part of the function, and no much can be
    #    done about it. Involves file IO and zip-unzipping.
    for i_state, state in enumerate(states):
        count_series[state], tmp_input_dict = load_sim_file(fname_func(sim_prefix, state),
            read_header=False, header_size=header_size,
        )
    # x_tf = time.time()
    # print()
    # print("[OPT {:0.5f}]".format(x_tf - x_t0), end=" ")

    # ------------------------------
    # SIM PARAMETERS AND BASIC FEATURES RETRIEVAL

    # Simulation parameters, retrieved from the last read file.
    feat["travel_fac"] = float(input_dict["travel_fac"])
    feat["k_reac"] = read_optional_from_dict(input_dict, "sd_reac_exponent", typecast=float)
    feat["lt_coef"] = read_optional_from_dict(input_dict, "sd_long_term_coef", typecast=float)
    feat["uniform_a"] = read_optional_from_dict(input_dict, "uniform_a", typecast=float)
    feat["detec_ratio"] = read_optional_from_dict(input_dict, "detec_ratio", typecast=float, standard_val=1.0)
    # Model parameters
    feat["model_class"] = input_dict["model_class"]
    feat["r0"] = float(input_dict["r0"])
    feat["mu"] = float_as_frac(input_dict["mu"])
    # Simulation params
    feat["num_exec"] = int(input_dict["num_exec"])

    # Initial conditions
    feat["init_mode"] = input_dict["init_mode"]
    feat["init_data"] = input_dict["init_data"]

    # Seed node (assuming unique!)
    feat["seed_node"] = get_seed_node(input_dict)

    # Optional event time lists
    feat["i_t_resets"] = read_optional_from_dict(input_dict, "i_t_resets", typecast=str_to_list)
    feat["i_t_deacts"] = read_optional_from_dict(input_dict, "i_t_deacts", typecast=str_to_list)

    # Simulation result features
    if i_exec is None:
        # Features only found in average simulations
        feat["total_final_frac"] = float(input_dict["avg_final_frac"])
        feat["t_outbreak"] = float(input_dict["avg_t_outbreak"])
        feat["max_t_outbreak"] = int(input_dict["max_t_outbreak"])
    else:
        # Features from single executions
        feat["max_t_outbreak"] = feat["t_outbreak"] = int(input_dict["t_outbreak"])

    tmax = feat["tmax"] = feat["max_t_outbreak"] + 2

    # -------------------------------
    # METAPOPULATION handling
    sim_nodes = list(map(int, read_csv_names(input_dict["nodes"], sep=";")))
    if g is None:
        g = load_metapop_from_input_dict(input_dict)
    if nodes is None:
        nodes = list(g.nodes())
    num_indiv = g.total_pop_size()
    num_nodes = len(nodes)
    feat["num_nodes"] = num_nodes
    x_tf = time.time()

    # Compares the g's list of nodes with that of the simulation files.
    if set(nodes) != set(sim_nodes):
        raise ValueError("Hey, node ids read from a simulation file do not coincide with those "
                         "on the metapopulation.")
    if nodes != sim_nodes:
        # Reorders data to the node order given on network
        for state in states:
            count_series[state] = reorder_array(count_series[state], sim_nodes, nodes)

    # -------------------------------
    # SIM DATA PROCESSING
    # [10/11/2020] Optimization note - this block was highly shortened AND optimized with numpy vectorization.
    # >100x performance gain.

    # # NO SUPPORT for dt != 1 for now. A lot of things have to be changed if it's the case.
    dt = 1.
    t_array = np.arange(0., tmax, dt)

    # INTERMEDIATE OBJ: contiguous array of counts over all states - needed to calculate pop_series
    count_array = np.stack([count_series[state] for state in states], axis=0)

    # Processed data structures
    total_frac_series = {state: np.sum(count_series[state], axis=0) / num_indiv for state in states}
    # ^^  Signature: dict[state][i_t], where the each dict entry is an array.
    pop_series = np.sum(count_array, axis=0)
    # ^^  Signature: array[i, i_t] = population of i-th node at i_t time step
    frac_series = {state: count_series[state] / pop_series for state in states}
    # ^^  Signature: dict[state][i_ni][i_t] = fraction of node population in each state at each time

    # ------------------------------------------
    # A_ARRAY HANDLING

    # a_array, either read from file or calculated on the fly
    if read_a_array:
        # Retrieves a_array
        sim_data, input_dict = load_sim_file(fname_func(sim_prefix, "a"))

        # sim_data is a global a_array
        if sim_data.shape == (tmax, ):
            total_a_series = sim_data
            a_series = np.array([total_a_series]*num_nodes)
            # FUTURE: see if you can find how to generate an array of pointers instead.

        # sim_data is a local, nodewise a_array
        elif sim_data.shape == (num_nodes, tmax):
            a_series = sim_data

            # Global a_array as a weighted average of the local ones
            total_a_series = np.average(a_series, weights=pop_series/num_indiv, axis=0)

        else:
            raise ValueError("Hey, shape of a_array is strange: {}\n"
                             "File: {}".format(sim_data.shape, sim_prefix))

    else:
        # Assumes a global reaction and calculates on the fly
        total_a_series = np.empty(tmax, dtype=float)
        a_series = np.array([total_a_series]*num_nodes)
        for i_t, t in enumerate(t_array):
            total_a_series[i_t] = global_response_term(total_frac_series["I"][i_t], total_frac_series["R"][i_t],
                                                       feat["k_reac"], feat["lt_coef"])

    # -----------------------------
    # GLOBAL FEATURES EXTRACTED FROM THE TIME SERIES
    # [10/11/2020] Note: at calc_execwise_features.py, some of the following feats are being recalculated.
    #     No impact on performance could be observed though (numpy vectorization is used).

    # Time to herd immunity
    feat["herd"] = 1 - 1. / feat["r0"]  # For metapopulations, this definition must be used with careful.
    feat["i_t_herd"] = np.argmax(total_frac_series["R"] > feat["herd"])
    if feat["i_t_herd"] == 0:
        feat["t_herd"] = np.inf
    else:
        feat["t_herd"] = t_array[feat["i_t_herd"]]

    # Peak time and size
    feat["peak_i_t"] = np.argmax(total_frac_series["I"])
    feat["peak_time"] = t_array[feat["peak_i_t"]]
    feat["peak_size"] = total_frac_series["I"][feat["peak_i_t"]]
    # Response term at I peak
    feat["a_at_peak"] = total_a_series[feat["peak_i_t"]]

    # Minimum a (max social isolation)
    feat["minimum_a"] = min(total_a_series)

    # Total social distancing (sum of a). May differ from homix, as it is just a sum.
    feat["a_impulse"] = (1. - total_a_series).sum()

    return SimBunch(t_array, count_series, pop_series, frac_series, a_series,
                    total_frac_series, total_a_series, feat, nodes, g)


def get_series_after_value(sim, value, monitor_states=("I", "R"), return_states=("I", "R"),
                           use_fraction=True):
    """Returns the time series of each node after the number (or fraction)
    of individuals in monitor_states overcomes a threshold value.
    """

    # Sums over all use_states
    if use_fraction:
        series = sim.frac_series
    else:
        series = sim.count_series
    monitor_array = sum((series[state] for state in monitor_states))
    use_array = sum((series[state] for state in return_states))

    # Finds the time when value is reached for each node
    i_t_first = np.argmax(monitor_array > value, axis=1)  # Series entirely above or below 'value' will return 0

    # Collects the array of each node in a list (arrays have different sizes)
    result = []
    for i, (first, node_array) in enumerate(zip(i_t_first, use_array)):
        result.append(node_array[first:])

    return result


def truncate_series(sim, i_t):
    """Truncates all the time series of the SimBunch to index i_t.
    Changes are made in place, so they are irreversible.
    """

    # 1D arrays  a[i_t]
    for name in ["total_a_series", "t_array"]:
        array = getattr(sim, name)
        if array is not None:
            setattr(sim, name, array[:i_t])

    # 2D arryas, first index for the nodes a[i, i_t]:
    for name in ["pop_series", "a_series"]:
        array = getattr(sim, name)
        if array is not None:
            setattr(sim, name, array[:, i_t])

    # Dicts of 1D arrays  a[state][i_t]
    for name in ["total_frac_series"]:
        d = getattr(sim, name)
        if d is not None:
            for key, array in d.items():
                d[key] = array[:i_t]

    # Dicts of 2D arrays  a[state][i][i_t]
    for name in ["frac_series"]:
        d = getattr(sim, name)
        if d is not None:
            for key, array in d.items():
                d[key] = array[:, :i_t]


def quickplot(sim, states=("I", "R"), plot_a=True, plot_labels=True, figsize_ax=None, local_lw=3.0):
    num_plots = len(states)
    figsize = (5.*num_plots, 4.) if figsize_ax is None else (num_plots * figsize_ax[0], figsize_ax[1])
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    axes = [axes] if num_plots == 1 else axes  # Repacks in case of plotting a single state

    for i_plot, state in enumerate(states):
        ax = axes[i_plot]
        # Time series
        ax.plot(sim.t_array, sim.frac_series[state].T)
        # Global series
        ax.plot(sim.t_array, sim.total_frac_series[state], "k", linewidth=local_lw, alpha=0.7)

        # a_array
        if plot_a and sim.total_a_series is not None:
            ax2 = ax.twinx()
            ax2.plot(sim.t_array, sim.total_a_series, color="r", linestyle="-.", linewidth=1.5)
            ax2.plot(ax2.get_xlim(), [1., 1.], "r--", linewidth=0.7)
            ax2.tick_params(axis='y', labelcolor="r")
            if plot_labels:
                ax2.set_ylabel("$a(t)$", color="r")

        # Labels
        if plot_labels:
            ax.set_xlabel("$t$")
            ax.set_ylabel("{} fraction".format(state))
        else:
            ax.set_title(state)  # Just for reference


    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------------------
# NODEWISE FEATURES calc functions
# ------------------------------------------------------------------------------
# Most (if not all) use the SimBunch class.

def calc_outbreak_size(sim, final_state="R"):
    return sim.frac_series[final_state][:, -1]


def calc_final_fracsum(sim, states=("I", "R")):
    return sum(sim.frac_series[state][:, -1] for state in states)


# Delete this dumb sh*t
# def calc_peak_time_and_size(sim, use_state="I"):
#     peak_it = np.argmax(sim.frac_series[use_state], axis=1)
#     peak_time = sim.t_array[peak_it]
#     peak_size = np.max(sim.frac_series[use_state], axis=1)
#     # peak_size = np.fromiter((sim.frac_series[use_state][i][i_t]  # Rejected for overcomplication
#     #                          for i, i_t in enumerate(peak_it)), dtype=float)
#
#     return peak_time, peak_size


def calc_peak_time(sim, use_state="I"):
    return sim.t_array[np.argmax(sim.frac_series[use_state], axis=1)]


def calc_peak_size(sim, use_state="I"):
    return np.max(sim.frac_series[use_state], axis=1)


def calc_peak_time_after_value(sim, value, monitor_states=("I", ), return_states=("I", ),
                               dt=1., use_fraction=True):
    # Sums over all use_states
    if use_fraction:
        series = sim.frac_series
    else:
        series = sim.count_series
    monitor_array = sum((series[state] for state in monitor_states))
    use_array = sum((series[state] for state in return_states))

    # Finds the time when value is reached for each node
    i_t_first = np.argmax(monitor_array > value, axis=1)  # Series entirely above 'value' will return 0

    peak_t = np.empty_like(i_t_first, dtype=float)
    # For each node, takes the sliced time series and calculates peak time.
    for i, (first, node_array) in enumerate(zip(i_t_first, use_array)):
        peak_t[i] = np.argmax(node_array[first:]) * dt

    return peak_t


def calc_time_above_value(sim, value, use_state="I"):
    """Calculates the number of time steps that each node spent above a given state value.
    """
    array = sim.frac_series
    return np.count_nonzero(array[use_state] > value, axis=1)


def calc_node_herd_time_abs(sim, immune_state="R", as_index=False, dt=1.):
    """
    Parameters
    ----------
    sim : SimBunch
    immune_state : str
    as_index : bool
        If True, returns the index (from t_array) at which herd was achieved, and -1 if not achieved.
        If False, returns as time (multiplies by dt), and np.inf if not achieved.
    dt : float
    """
    array = sim.frac_series[immune_state]
    ht = np.argmax(array > sim.feat["herd"], axis=1)

    # Replaces null entries by -1 (for index) or np.inf (for time).
    if as_index:
        ht[ht == 0] = -1
    else:
        ht = ht.astype(np.float) * dt
        ht[ht < dt/2.] = np.inf  # Finds zeros by checking if number is smaller than dt
    return ht


def calc_node_herd_time_rel(sim, value, monitor_states=("I", "R"), immune_state="R",
                            as_index=False, dt=1.0):
    """
    Calculates the nodewise time to herd immunity, relative to the moment at which
    the node population reaches a given fraction of monitored cases.
    The fraction is given as 'value' (no option to use counts here), the monitored
    states are monitor_states, while immune_state is that considered to herd immunity.

    Parameters
    ----------
    sim : SimBunch
    value : float
        Always as a fraction, not count.
    monitor_states : any sequence
        States to monitor and compare with value.
    immune_state : str
    as_index : bool
        If True, returns the index (from t_array) at which herd was achieved, and -1 if not achieved.
        If False, returns as time (multiplies by dt), and np.inf if not achieved.
    dt : float
    """
    # array = sim.frac_series[immune_state]
    ht = np.empty_like(sim.nodes, dtype=int)

    # Gets the relative time series for immune, then finds herd time for each
    array_list = get_series_after_value(sim, value, monitor_states=monitor_states,
                                        return_states=(immune_state, ), use_fraction=True)
    for i, array in enumerate(array_list):
        ht[i] = np.argmax(array > sim.feat["herd"])

    # Replaces null entries by -1 (for index) or np.inf (for time).
    if as_index:
        ht[ht == 0] = -1
    else:
        ht = ht.astype(np.float) * dt
        ht[ht < dt/2.] = np.inf  # Finds zeros by checking if number is smaller than dt
    return ht


def calc_node_a_impulse(sim: SimBunch, dt=1.0):
    """Assumes that sim.a_series, which is the local series, is defined.
    """
    return np.sum(1. - sim.a_series, axis=1) * dt


def calc_node_abar_mean(sim: SimBunch):
    return calc_node_a_impulse(sim) / sim.feat["tmax"]


def calc_node_abar_max(sim: SimBunch):
    return np.max(1. - sim.a_series, axis=1)


def calc_num_ipeaks(sim: SimBunch, eps, smooth_steps=30):
    """Returns the number of infectious frac peaks that are above a given threshold
    in a smoothened time series.
    Not very precise, as plateaus and stochastic fluctuations can generate spurious counts.

    This is probably much slower and memory consumer than the other metrics.
    """
    len_series = sim.feat["tmax"]
    len_diff = len_series - smooth_steps

    # Aux infrastructure containers
    smooth_container = np.empty(len_diff + 1, dtype=float)  # Smoothened series (moving average)
    diff_container = np.empty(len_diff, dtype=float)  # Difference between consecutive smoothened steps
    sprod_container = np.empty(len_diff - 1, dtype=float) # Product between consecutive differences
    smooth_mask = np.ones(smooth_steps)

    res = np.empty(sim.feat["num_nodes"], dtype=int)

    for i_ni, series in enumerate(sim.frac_series["I"]):
        # Smooth (moving average) series
        smooth_container[:] = np.convolve(series, smooth_mask, 'valid') / smooth_steps
        # Difference of consecutive steps ("derivative")
        diff_container[:] = np.convolve(smooth_container, np.array([1, -1], dtype=int), 'valid')
        # Products of consecutive differences
        sprod_container[:] = diff_container[:-1] * diff_container[1:]

        # Criteria to count for local maxima
        maxima = np.argwhere(np.logical_and(
            np.logical_and(
                sprod_container < 0,  # Product is negative - zero crossing
                smooth_container[1:len_diff] > eps),  # Infectious moving average value above threshold
                diff_container[:len_diff - 1] > 0)  # Up->down crossings only
        )

        # Counts the number of detected maxima
        res[i_ni] = len(maxima)

    return res


def calc_num_outb_threshold(sim: SimBunch, start_thres, end_thres):
    """
    Determines how many outbreaks occurred by threshold crossings, where thresholds for "activation" and
    "deactivation" are different.

    Returns, for each node, the number of outbreaks counted this way.
    """
    res = np.empty(sim.feat["num_nodes"], dtype=int)

    for i_ni, series in enumerate(sim.frac_series["I"]):
        # -----------
        # Use convolution to find threshold crossings
        # Array of products between successive differences to the threshold
        start_sprod = (series[:-1] - start_thres) * (series[1:] - start_thres)
        end_sprod = (series[:-1] - end_thres) * (series[1:] - end_thres)

        # List of crossings (in any direction)
        start_cross = np.argwhere(start_sprod < 0)
        end_cross = np.argwhere(end_sprod < 0)

        # Include the possibility of starting inside an outbreak
        if series[0] > start_thres:
            start_cross = [0] + start_cross

        # ---------------------------------
        # Outbreak events registration
        outb = 0
        i_end = 0  # Index for end_cross elements that's updated manually
        t_end = -1  # Time stamp of the last detected outbreak end, to avoid double-counting
        for t_start in start_cross:

            if series[t_start] > start_thres:
                # Up-down cross. Discard
                continue

            if t_start <= t_end:
                # Repeated outbreak start. Means that the previous outbreak did not finish. Discard
                continue

            # Looks for the next end crossing after the current start crossing
            try:
                while end_cross[i_end] <= t_start:  # While t_end < t_start, get next t_end
                    i_end += 1
                t_end = end_cross[i_end]
            except IndexError:
                # This happens if the simulation exceeded tmax, or something like this. Stop here and return current.
                break
            finally:
                outb += 1  # Here it is just counted

            # Appends a tuple with start and end indexes of the current outbreak.
            # outb.append((t_start + 1, t_end + 1))  # (First outbreak t, one past last outbreak t)

        res[i_ni] = outb

    return res


# A single function that calculates the nodewise metric array from its name
def calc_node_metric(sim, metric, monitor_states=("I", "R"), monitor_value=0.01, critical_value=0.03,
                     histher=0.2):
    """TODO: this function can be more elegantly (and perhaps more efficiently) used with
          dicts instead of a big elif. A good enhancement could be a broadcast of the type of each metrics.
          Problem: handle the variable list of arguments.
    """

    d = dict()
    # Metric elif ladder
    if metric == "peak_size":
        d["array"] = calc_peak_size(sim, use_state="I")
        d["y_label"] = "Local peak size"

    elif metric == "r_max":
        d["array"] = calc_peak_size(sim, use_state="R")
        d["y_label"] = "Local outbreak size"

    elif metric == "abs_peak_time":
        d["array"] = calc_peak_time(sim, use_state="I")
        d["y_label"] = "days"

    elif metric == "outb_size":
        d["array"] = calc_outbreak_size(sim, final_state="R")
        d["y_label"] = "Local outbreak size"

    elif metric == "final_fracsum":
        d["array"] = calc_final_fracsum(sim, states=("I", "R"))
        d["y_label"] = "prevalence"

    elif metric == "rel_peak_time":
        d["array"] = calc_peak_time_after_value(sim, monitor_value, monitor_states=monitor_states,
                                                return_states=("I",), use_fraction=True)
        d["y_label"] = "days"

    elif metric == "abs_herd_time":
        d["array"] = calc_node_herd_time_abs(sim)
        d["y_label"] = "days"

    elif metric == "rel_herd_time":
        d["array"] = calc_node_herd_time_rel(sim, monitor_value, monitor_states=monitor_states)
        d["y_label"] = "days"

    elif metric == "time_above_val":
        d["array"] = calc_time_above_value(sim, value=critical_value) * 1.0  # dt = 1.0
        d["y_label"] = "days"

    elif metric == "a_impulse":
        d["array"] = calc_node_a_impulse(sim, dt=1.)
        d["y_label"] = "intensity * days"

    elif metric == "abar_mean":
        d["array"] = calc_node_abar_mean(sim)
        d["y_label"] = "intensity"

    elif metric == "abar_max":
        d["array"] = calc_node_abar_max(sim)
        d["y_label"] = "intensity"

    elif metric == "num_outb_threshold":
        d["array"] = calc_num_outb_threshold(sim, start_thres=critical_value, end_thres=(1. - histher) * critical_value)
        d["y_label"] = "outbreaks"

    else:
        # Metric not implemented
        raise ValueError("Hey, metric {} not implemented".format(metric))
        # d = None

    return d

# -------------------------------------- -  - - - - - - - -- - -  - - - - - - - - - - - - - - -
# --------------------------------------
# MAIN
# --------------------------------------
# --------------------------------------

# --------------------------------------


# # TEST
# def main():
#     # sim_path_prefix = "outputs/new_response_curves/path_n10_trav1e-1/seir_glob_a"  # Path (use b to trav1e-2)
#     path_prefix = "outputs/condit_threshold/strategies/path_n10_thres1E-3_trav1e-1/seir_condit"  # Path
#     # path_prefix = "outputs/condit_threshold/er_k10/k05p00_thres1e-3_trav1e-2/condit"
#     states = "SIR"
#     # read_a = True
#     strat_prefix = "_k20p00_l0p00"  # "_basic"  #
#
#     # ------------------
#     # Gets network from a sample file
#     # g = load_metapop_from_input_dict(read_config_file(path_prefix + "_E.out"))
#     # t_array, count_series, pop_series, frac_series, total_frac_series, total_a_series, feats = \
#     #     load_and_process_sim_data(path_prefix, states, read_a_array=read_a, i_exec=2)
#
#     g = load_metapop_from_input_dict(read_config_file(gen_sim_fname(path_prefix + "_basic", "S")))
#     nodes = list(g.nodes())
#     sim = load_and_process_sim_data(path_prefix + strat_prefix, states, read_a_array=True,
#                                     g=g, nodes=nodes)
#
#     # plt.plot(sim.t_array, sim.frac_series["R"].T)
#
#     # plt.plot(sim.t_array, a.T)
#     series_after_val = get_series_after_value(sim, 0.02, monitor_states=("I", "R"),
#                                               return_states=("R",))
#     for i, ni in enumerate(nodes):
#         plt.plot(series_after_val[i])
#
#     # print(calc_time_above_value(sim, 4E-2))
#
#     # Herd immunity
#     plt.plot([0, max(sim.t_array)], [sim.feat["herd"]]*2, "--", color="orange")
#     # for therd in calc_node_herd_time_abs(sim):
#     for therd in calc_node_herd_time_rel(sim, 0.02, ):
#         plt.plot([therd]*2, [0., sim.feat["herd"]], "--", linewidth=0.5)
#
#     print(calc_peak_time_after_value(sim, 0.01))
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
