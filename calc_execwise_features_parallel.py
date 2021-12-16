"""

Calculates the features for simulations in an executionwise manner, allowing for statistics and more correct
measures.

Produces metric files (.met) as outputs: files that contain the metrics of each execution (lines),
for each node (columns).

Usage
-----
TODO!



"""

# TODO LIST OF THIS PROCEDURE
# v Clean the use interface
# v Optimize reading and calculation of metrics
# o File checkpoint after nodeset aggregation
# o Ready-to-plot file checkpoint
# o Some header data to metric files
# o More functions and less code on main()
# o Reformulate nodeset aggregation (may be hard)
# o PARALLELIZE? CAN I?

# --- Optimization of the metric reading and calculating ----
# - The time spent calculating the metrics was very small inside the function.
#      - eg: 0.008s of a total 1.774s for a given simulation.
#      - 'load_and_process_sim_data' : 1.31s out of 1.72s


import glob  # List of file names using wildcards
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sys
import time
import datetime

from nodewise_features_metapop import load_and_process_sim_data, calc_node_metric, get_nodelists_by_shortestpath, \
    gen_sim_fname, load_metapop_from_input_dict, get_seed_node
from toolbox.file_tools import make_folder, read_file_header, read_config_strlist, read_config_file, \
    HEADER_END, SEP, seconds_to_hhmmss, possibly_unzip_file, remove_file, read_argv_optional
from pathos.pools import ProcessPool

STD_NUM_PROCESSES = 10
num_processes = read_argv_optional(1, dtype=int, default=STD_NUM_PROCESSES)


# ----------------

def calc_simulation_metrics(sim_prefix, num_exec, use_metrics, monitor_states=("I", "R"), monitor_value=0.01,
                            critical_value=0.03, histher=0.1):
    """Loads a simulation dataset (i.e., a set of independent executions with same parameters)
    and calculates, for each, the required epidemic metrics.

    Parameters
    ----------
    sim_prefix : str
        Prefix of the simulation files, which is passed to 'load_and_process_sim_data()'.
        The prefix is used with 'gen_exec_fname()' to create the actual file names.
    num_exec : int or iterable.
        Sequence of execution indexes to be used. If an integer, it uses range(num_exec).
    use_metrics : sequence of str
        Names of the metrics to be calculated, as accepted in 'calc_node_metric'.
    """

    # Either takes all executions from 0 to num_exec-1 or assumes num_exec is an iterable with desired indexes.
    if isinstance(num_exec, int):
        i_exec_list = list(range(num_exec))
    else:
        i_exec_list = list(num_exec)

    # Dummy execution loading, to get some constant features and save time.
    i_exec = i_exec_list[0]
    exec_bunch = load_and_process_sim_data(sim_prefix, i_exec=i_exec)
    g = exec_bunch.g
    nodes = exec_bunch.nodes
    num_nodes = len(nodes)

    # Allocates containers for the calculated metrics
    index = pd.MultiIndex.from_product((i_exec_list, nodes), names=("exec", "node"))
    sim_metrics_df = pd.DataFrame(index=index,  columns=use_metrics, dtype=(float, float))  # Signature: df[i_exec, ni]

    # Loops over execution files.
    # test_dt = 0.
    for i_exec in i_exec_list:
        # test_t0 = time.time()
        exec_bunch = load_and_process_sim_data(sim_prefix, i_exec=i_exec, g=g, nodes=nodes)
        # test_tf = time.time()
        # test_dt += test_tf - test_t0

        # For each metric, calculates in all nodes

        for metric in use_metrics:
            # Metric calculation command
            d = calc_node_metric(exec_bunch, metric, monitor_states=monitor_states, monitor_value=monitor_value,
                                 critical_value=critical_value, histher=histher)

            # Storing of the metrics into the multiindex dataframe
            # sim_metrics_df.xs(i_exec, level="exec")[metric] = d["array"]
            sim_metrics_df.loc[(i_exec, ), metric][:] = d["array"][:]

    # print(" /t: {:0.5f}s/ ".format(test_dt), end="")

    # print(sim_metrics_df.xs(1, level="node"))
    return sim_metrics_df


def export_simulation_metrics(sim_metrics_df, fname, split_files=False, out_format="csv",
                              sep=";", header_end=HEADER_END):
    """
    Export calculated metrics from a simulation, i.e., a set of independent executions.
    Currently no support for making a file header (besides that created by Pandas)

    Parameters
    ----------
    sim_metrics_df : pd.DataFrame
        Data frame of a simulation. Expects the signature:
        multiindex = (i_exec, ni)
        columns = _metric names_
    fname : str
        Output file name. Creates the underlying directory if not existent.
    split_files : bool
        Not implemented. If true, shoud split each metric in one file.
    sep : str
        Separation character for the CSV format.
    """

    # Creates output folder if does not exist. No warning if already exists.
    make_folder(os.path.dirname(fname), silent=True)

    use_metrics = sim_metrics_df.columns  # Column names.
    index_names = sim_metrics_df.index.names  # Names of multiindex levels.

    out_string = ""

    # Writes a header with simulation data (todo: from where?)

    # Exports dataframe into a single file with all metrics
    # Signature: df(exec, node)[metric]
    if out_format == "str":
        out_string += sim_metrics_df.to_string()
    elif out_format == "csv":
        out_string += sim_metrics_df.to_csv(None, sep=sep)#, float_format="{:0.8f}") # float_format has a problem. Dtypes could help
    else:
        raise ValueError("Hey, unrecognized out_format: '{}'".format(out_format))

    # Finally writes content to file.
    with open(fname, "w") as fp:
        fp.write(out_string)


def import_simulation_metrics(fname, input_format="csv", sep=";", header_end=HEADER_END):
    """Imports a dataframe of nodewise/executionwise metrics, as exported by 'export_simulation_metrics'. """

    # File header handling
    try:
        file_header = read_file_header(fname, header_end)
        header_size = len(file_header) + 1
    except EOFError:
        file_header = []  # No header
        header_size = "infer"  # Parameter used by pandas
    input_dict = read_config_strlist(file_header)  # Assumes default entry, attr and comment chars

    # -----------
    # Reads according to each format
    if input_format == "csv":
        df = pd.read_csv(fname, sep, header=header_size, index_col=[0, 1])
    elif input_format == "str":
        raise NotImplementedError("Sorry, no time to implement an import of this unified metric file!")
    else:
        raise ValueError("Hey, unrecognized input_format: '{}'".format(input_format))

    return df, input_dict


def make_strategy_df_from_data(sim_df_list, sim_prefixes):
    """
    Makes a data frame for a whole strategy, i.e., a series of simulations data frames,
    from a list of simulation dataframes.
    """
    strat_df = pd.concat(sim_df_list, axis=0, keys=sim_prefixes)
    strat_df.index.set_names("sim_prefix", level=0, inplace=True)

    return strat_df


def make_strategy_df_from_files():
    """
    Makes a data frame for a whole strategy, i.e., a series of simulations,
    from a list of data files that store calculated metrics (one for each sim).
    """


def make_thebig_df_from_data(strat_df_list, strat_names):
    """Joins strategy data frames into a single df - **The Big DF** -

    Signature of The Big DF:
    df(strategy, sim_prefix, exec, node)[metrics]
    """
    thebig_df = pd.concat(strat_df_list, axis=0, keys=strat_names)
    thebig_df.index.set_names("strategy", level=0, inplace=True)

    return thebig_df


# ----------------------------------------------------
# AGGREGATION AND STATISTICAL ANALYSIS OF THE DATA
# ----------------------------------------------------

def aggregate_metrics_by_nodesets(df, nodelists, nodeset_names=None, weightlists=None, level_name="node",
                                  use_metrics=None, print_looptime=True):
    """Aggregates a dataframe by nodes (into nodesets), returning a data frame with same structure
    but with nodesets instead of nodes.

    Accepts different weights for each node.

    Let it registered: this single function with ugly implementation took me days of suffering to figure out
       a way for it to *simply work* using the pandas logic.

    Parameters
    ----------
    df : pd.DataFrame
        Thebig_df
        Signature: df[(strat_name, sim_prefix, exec, node)] -> [metrics]
    nodelists : list or None
        This represents the sets of nodes to aggregate.
        It is a nested list, each one containing the node indexes of each set. Nodes may be in more than
        one set.
    nodeset_names : sequence
        List of names of the nodesets. Used in out_df to label the sets, so they must be unique.
    weightlists : list
        This represents the weights of each node in each nodeset.
        Must have the same nested structure (and lenghts) of nodelists.
    level_name : hashable
        Name of the level to be aggregated (i.e., the nodes).
        The method is agnostic to the other levels.
    use_metrics : sequence
        Metrics to use. Must be a subset of the names of columns in df.
    print_looptime : bool
        Whether the method should print the main loop execution time.
        It's quite costly and can be optimized...

    Returns
    -------
    Possible signature of the output dataframe:
        out_df(strategy, sim_prefix, exec, nodeset)[use_metrics]
    """
    num_nodesets = len(nodelists)

    if nodeset_names is None:
        # Uses simple numbers
        nodeset_names = list(range(num_nodesets))

    if weightlists is None:
        # Weights not passed - set all to 1
        weightlists = [[1.]*len(l_nodes) for l_nodes in nodelists]
    # Precalculate the normalization of weights in each nodeset
    weight_sums = [sum(weights) for weights in weightlists]

    if use_metrics is None:
        use_metrics = df.columns

    # print(df.index)
    # print(df.index.set_levels(nodeset_names, level="node"))

    # This complicated routine designs a new index obeject, with nodesets instead of nodes
    #   It is agnostic to the other levels, only "node" is replaced.
    new_index = df.index.droplevel(level=level_name)  # Multiindex without level 'node'
    df_from_index = new_index.to_frame(index=False)  # Converts multiindex to a frame with each level as a column
    df_from_index.drop_duplicates(inplace=True, ignore_index=True)  #
    new_index = pd.MultiIndex.from_frame(df_from_index)  # Creates an index yet without level "node"
    num_rows = len(df_from_index)  # Number of rows without the "node" level.
    # nodeset_col = nodeset_names * len(df_from_index)  # Creates a repeated list of nodeset names
    tmp_df = pd.DataFrame({name: np.repeat(np.nan, num_rows) for name in nodeset_names}, index=new_index)
    stacked_df = tmp_df.stack(dropna=False)  # Reshapes the previous df, finally making the desired multiindex
    stacked_df.index.set_names("nodeset", level=-1, inplace=True)

    # After everything, allocates the output df.
    #  Possible Signature: out_df(strategy, sim_prefix, exec, nodeset)[metrics]
    out_df = pd.DataFrame({metric: np.repeat(np.nan, len(stacked_df)) for metric in use_metrics},
                          index=stacked_df.index)

    # TODO: remove the trash from memory? (Eg. tmp_df, stacked_df)


    # ---------------------------
    # Main loop over executions (agnostic to the df levels, except the one given as level_name)

    levels_to_iterate = list(df.index.names)
    levels_to_iterate.remove(level_name)  # Only removes the node level, agnostic to the others

    loop_t0 = time.time()

    # Progress monitoring

    # TODO: The next loop seems painfully slow, but I think that actually the ugly ways of indexing and cross-sectioning
    #  are the main bottleneck (rather than the calculations themselves). Check that and see what can be done.
    loop_size = len(df.groupby(level=levels_to_iterate))  # Yup, twice. Sorry.
    param_t0 = time.time()
    for i_param, (params, exec_df) in enumerate(df.groupby(level=levels_to_iterate)):

        # Local loop over each set of nodes
        for i, l_nodes in enumerate(nodelists):

            # Calculates the weighted average of required metrics for the current nodeset
            nset_average = sum(weight * exec_df.loc[(*params, ni)][use_metrics]
                               for (ni, weight) in zip(l_nodes, weightlists[i]))
            nset_average /= weight_sums[i]

            # Put into final data frame
            out_df.loc[(*params, nodeset_names[i])] = nset_average

        # Iteration time feedback
        if i_param % 100 == 0:
            param_tf = time.time()
            print("{:0.3f}%: {:6.4}s\n".format(100 * i_param / loop_size, param_tf - param_t0), end=" ")
            sys.stdout.flush()
            param_t0 = param_tf

    # # WOULD NOT PARALLELIZE - the slow part is probably not paralellizable
    # def func(i_param, params, exec_df):
    #
    #     # Local loop over each set of nodes
    #     for i, l_nodes in enumerate(nodelists):
    #
    #         # Calculates the weighted average of required metrics for the current nodeset
    #         nset_average = sum(weight * exec_df.loc[(*params, ni)][use_metrics]
    #                            for (ni, weight) in zip(l_nodes, weightlists[i]))
    #         nset_average /= weight_sums[i]
    #
    #         # Put into final data frame
    #         out_df.loc[(*params, nodeset_names[i])] = nset_average
    #
    #
    # pool = ProcessPool(num_processes)

    print()

    # Execution time feedback
    loop_tf = time.time()
    if print_looptime:
        print(" - Time calculating nodeset averages: {} ({:0.5f}s)"
              "".format(seconds_to_hhmmss(loop_tf - loop_t0), loop_tf - loop_t0))

    return out_df


def aggregate_executions_with_statistics(df, level_name="exec"):
    """
    For each simulation, calculates the average over executions, as well as other statistical metrics.

    Returns a data frame with mean and std of each metrics, as a multiindex column structure:

    Returns
    -------
    Expected signature:
        out_df(strategy, sim_prefix, nodeset)[(metric, score)]
        ... where score is "mean", "std".
    """
    levels_to_groupby = list(df.index.names)
    levels_to_groupby.remove(level_name)
    grouped = df.groupby(level=levels_to_groupby)

    metrics = df.columns

    def mean_95ci(a):
        """Calculates the bayesian 95% confidence interval for the array-like sample a.
        THIS FUNCTION MAY BE CHANGED, as I don't even know what is this bayesian (I only studied
        z, t-student, etc).
        """
        mvs_tuples = scipy.stats.bayes_mvs(a, 0.95)
        return mvs_tuples[0][1]  # Returns a (lower, upper) tuple for the 95CI of the mean

    # USING THE ABOVE FUNCTION caused a 50-fold execution time increase. So why bother?!?

    return grouped.agg([np.mean, np.std])


# --------------------------------------------
# I/O OPERATIONS WITH THE COMPLETELY PROCESSES DATAFRAME
# --------------------------------------------

def export_final_df(df, fname):

    # Creates output folder if does not exist. No warning if already exists.
    make_folder(os.path.dirname(fname), silent=True)

    # out_string = ""  # In case you want to write first
    # Writes a header with some data (todo: from where?)

    df.to_csv(fname, sep=";")  #, float_format="{:0.8f}") # float_format has a problem. Dtypes could help


def import_final_df(fname):
    """Reads data written with 'export_final_df'."""
    return pd.read_csv(fname, index_col=[0, 1, 2], sep=";", skipinitialspace=True, header=[0, 1])


# --------------------------------------------
# PLOT FUNCTIONS
# --------------------------------------------

# AUX (copy from jupyter)
def make_axes_list(num_axes, max_cols=3, total_width=9., ax_height=5.):
    """Creates a sequence of num_axes axes in a figure.
    The axes are periodically disposed into rows of max_cols elements.
    Exceeding axes in the last row are removed
    """
    num_rows = (num_axes - 1) // max_cols + 1
    exceed_axes = max_cols * num_rows - num_axes

    #     figsize = stdfigsize(scale, num_rows, max_cols, ratio)  # Meh
    figsize = (total_width, ax_height * num_rows)
    fig, axes = plt.subplots(num_rows, max_cols, figsize=figsize, squeeze=False)

    # Removes exceeding axes from the last row
    for i_remove in range(exceed_axes):
        axes[-1][-i_remove - 1].remove()

    # Cuts and reshapes the axes array to a 1D object with the valid axes
    axes = np.reshape(axes, num_axes + exceed_axes)[:num_axes]  # Alternative to concatenate
    #     axes = np.concatenate(axes)[:num_axes]

    return fig, axes


def plot_feature_curves(main_df, fig, axes, plot_metrics, max_cols,
                        xlabel=None, ylabel=None):
    """

    Parameters
    ----------
    main_df : pd.DataFrame
        Expects to have the signature: main_df(strategy, sim_prefix, nodeset)[(metric, score)]
    fig : plt.Figure
    axes : list
    """

    # Pre-setup of the plot
    prop_cycle = (plt.cycler(color=['red', 'sandybrown', 'b', 'y']) +
                  plt.cycler(linestyle=["-"] * 4) +
                  plt.cycler(marker=["o", "s", "^", "h"])
                  )

    # Parameters of the multi-axis plot
    num_axes = len(axes)
    num_rows = (num_axes - 1) // max_cols + 1
    last_row = max_cols - max_cols * num_rows + num_axes

    # Alternative axis labels
    if xlabel is None:
        xlabel = plot_metrics[0]

    if ylabel is None:
        ylabel = plot_metrics[1]

    # print(main_df[("peak_size", "mean")])

    # First level: nodeset - one plot for each
    for i_plot, (nodeset_name, nodeset_df) in enumerate(main_df.groupby("nodeset")):
        # Drops the grouped level
        nodeset_df = nodeset_df.droplevel(level="nodeset")

        ax = axes[i_plot]
        ax.set_prop_cycle(prop_cycle)

        # Second level: strategy - one curve for each
        for i_curve, (strat_name, strat_df) in enumerate(nodeset_df.groupby("strategy")):
            # Drops the grouped level
            strat_df = strat_df.droplevel(level="strategy")

            # Mean value plot - POINTS
            ax.plot(strat_df[(plot_metrics[0], "mean")], strat_df[(plot_metrics[1], "mean")], label=strat_name)

        # Local axes setup
        if i_plot % max_cols == 0:
            ax.set_ylabel(ylabel)

        ax.text(0.7, 0.8, str(nodeset_name), transform=ax.transAxes)

    # General axes setup
    for i_ax in range(last_row):
        axes[-i_ax-1].set_xlabel(xlabel)

    axes[0].legend()


def get_current_timestamp_iso():
    """Returns current date and time, up to seconds, in ISO 8601 format."""
    return datetime.datetime.now().isoformat().split(".")[0]


# ------------------------------------------------------------------------------------
# BUNCH STRUCT FOR STRATEGIES
# ------------------------------------------------------------------------------------

class StrategyBunch:

    __slots__ = ["full_prefixes", "num_exec"]

    def __init__(self,  full_prefixes, num_exec):
        self.full_prefixes = full_prefixes
        self.num_exec = num_exec


# ------------------------------------------------------------------------------------
# --- MAIN FUNCTION
# ------------------------------------------------------------------------------------


def main():

    #
    # -------------------------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------------------------
    #

    # # Definition of the metrics. May contain several, but the first two are used to plot
    use_metrics = ("abar_max", "r_max", "outb_size", "peak_size", "rel_peak_time", "a_impulse")  # LT pair + extras
    # use_metrics = ("a_impulse", "peak_size", "outb_size", "rel_peak_time", "abs_herd_time", "rel_herd_time",
    #                "time_above_val", "abar_max")  # ST pair + extras (added abar_max)
    num_exec = 120  # Can be set individually for each strategy

    # Parameters
    try_to_import_metrics = False  # Whether metric files should be imported instead of calculated
    stop_after_calc_metrics = False  # Stops after generating nodewise metrics

    do_export_final_df = True
    RGN = "04"
    # final_df_fname = "execwise_curves/varynet/" + "rgn{}_ST_trav1e-3.csv".format(RGN)  # + get_current_timestamp_iso() + ".csv"
    # final_df_fname = "outputs/tests/welcome_back_review1/er_a_test.csv"
    # final_df_fname = "execwise_curves/sir_test_ba-m5/LT_trav1e-5.csv"

    # --- Real networks
    # final_df_fname = "execwise_curves/real_nets_curves/test.csv"
    # final_df_fname = "execwise_curves/real_nets_curves/spain/st_trav5p0e-1.csv"
    # final_df_fname = "execwise_curves/real_nets_curves/brazil/lt_trav5p0e-1.csv"
    final_df_fname = "execwise_curves/another_lt_01.csv"

    # --- Sensitivity
    # final_df_fname = "execwise_curves/test/sensitivity.csv"
    # final_df_fname = "execwise_curves/sensitivity/st_trav1e-2_r02p50.csv"
    # final_df_fname = "execwise_curves/sensitivity/st_extra-rgn_trav0p34.csv"

    # ---------------------------------
    # COLLECTION OF THE STRATEGY DATA
    # strat_dict is a dictionary of StrategyBunch objects.
    strat_dict = {}  # Signature: d[strat_name] = list of full prefixes to sim files

    # LT global
    # sim_folder = "outputs/sir_test_sims/loc_vs_glob/erk10_trav1e-3_execs/global/"
    # sim_folder = os.path.expanduser("outputs/sir_1k_outputs/loc_vs_glob/rgn0p25-04_trav1e-4/global/")
    # sim_folder = os.path.expanduser("outputs/sir_test_sims/loc_vs_glob/ba-m5_trav1e-5/global/")
    sim_folder = "outputs/sensitivity_outs/another_st_rgn_trav1p0e-3/global/"
    # k_list_02 = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.]
    k_list_02 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.]
    sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    full_prefixes = [sim_folder + name for name in sim_prefixes]
    strat_dict["LT Global"] = StrategyBunch(full_prefixes, num_exec)

    # LT local
    # sim_folder = "outputs/sir_test_sims/loc_vs_glob/erk10_trav1e-3_execs/local/"
    # sim_folder = os.path.expanduser("outputs/sir_1k_outputs/loc_vs_glob/rgn0p25-04_trav1e-4/local/")
    # sim_folder = os.path.expanduser("outputs/sir_test_sims/loc_vs_glob/ba-m5_trav1e-5/local/")
    sim_folder = "outputs/sensitivity_outs/another_st_rgn_trav1p0e-3/local/"
    # k_list_02 = [1., 3.0, 12.0, 15.0, 20., 25., 30., 40., 50.]
    sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    full_prefixes = [sim_folder + name for name in sim_prefixes]
    strat_dict["LT Local"] = StrategyBunch(full_prefixes, num_exec)

    # # ST global
    # # sim_folder = "outputs/sir_test_sims/loc_vs_glob/erk10_trav1e-4_execs/global/"
    # # sim_folder = os.path.expanduser("outputs/sir_1k_outputs/loc_vs_glob/rgn0p25-04_trav1e-3/global/")
    # sim_folder = "outputs/sensitivity_outs/another_st_rgn_trav1p0e-3/global/"
    # # sim_folder = os.path.expanduser("outputs/sir_test_sims/loc_vs_glob/ba-m5_stub_trav1e-3/global/")
    # k_list_01 = [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Global"] = StrategyBunch(full_prefixes, num_exec)
    #
    # # ST local
    # # sim_folder = "outputs/sir_test_sims/loc_vs_glob/erk10_trav1e-4_execs/local/"
    # sim_folder = "outputs/sensitivity_outs/another_st_rgn_trav1p0e-3/local/"
    # # sim_folder = os.path.expanduser("outputs/sir_test_sims/loc_vs_glob/ba-m5_stub_trav1e-3/local/")
    # k_list_01 = [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Local"] = StrategyBunch(full_prefixes, num_exec)

    # --- REAL NETWORKS - SPAIN and BRAZIL

    # # LT global
    # # sim_folder = os.path.expanduser("outputs/real_networks/brazil/trav1p0e-0_global/")
    # # sim_folder = os.path.expanduser("outputs/real_networks/spain/trav5p0e-1/global/")
    # sim_folder = os.path.expanduser("outputs/real_networks/brazil/roraima/global/")
    # k_list_02 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.]
    # sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["LT Global"] = StrategyBunch(full_prefixes, num_exec)
    #
    # # LT local
    # # sim_folder = os.path.expanduser("outputs/real_networks/brazil/trav1p0e-0_local/")
    # # sim_folder = os.path.expanduser("outputs/real_networks/spain/trav5p0e-1/local/")
    # sim_folder = os.path.expanduser("outputs/real_networks/brazil/roraima/local/")
    # sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["LT Local"] = StrategyBunch(full_prefixes, num_exec)

    # # ST global
    # # sim_folder = os.path.expanduser("outputs/real_networks/brazil/trav1p0e-0_global/")
    # # sim_folder = os.path.expanduser("outputs/real_networks/spain/trav5p0e-1/global/")
    # sim_folder = os.path.expanduser("outputs/real_networks/brazil/roraima/global/")
    # k_list_01 = [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Global"] = StrategyBunch(full_prefixes, num_exec)

    # # ST local
    # # sim_folder = os.path.expanduser("outputs/real_networks/brazil/trav1p5e-0_local/")
    # # sim_folder = os.path.expanduser("outputs/real_networks/spain/trav5p0e-1/local/")
    # sim_folder = os.path.expanduser("outputs/real_networks/brazil/roraima/local/")
    # k_list_01 = [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Local"] = StrategyBunch(full_prefixes, num_exec)

    # --- SENSITIVITY ANALYSIS
    # # LT global
    # # sim_folder = os.path.expanduser("outputs/tests/r0_sensitest/r01p20/sir_global/")
    # # sim_folder = os.path.expanduser("outputs/sensitivity_outs/trav1e-2_r01p20/global/")
    # sim_folder = os.path.expanduser("outputs/extra_rgn_out/trav0p34/global/")
    # k_list_02 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.]
    # sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["LT Global"] = StrategyBunch(full_prefixes, num_exec)
    #
    # # LT Local
    # # sim_folder = os.path.expanduser("outputs/sensitivity_outs/trav1e-2_r01p20/local/")
    # sim_folder = os.path.expanduser("outputs/extra_rgn_out/trav0p34/local/")
    # # k_list_02 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.]
    # sim_prefixes = ["sir_k{}_l1p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_02]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["LT Local"] = StrategyBunch(full_prefixes, num_exec)

    # # ---
    # # ST global
    # # sim_folder = os.path.expanduser("outputs/sensitivity_outs/trav1e-2_r02p50/global/")
    # sim_folder = os.path.expanduser("outputs/extra_rgn_out/trav0p34/global/")
    # k_list_01 = [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Global"] = StrategyBunch(full_prefixes, num_exec)
    #
    # # ST local
    # # sim_folder = os.path.expanduser("outputs/sensitivity_outs/trav1e-2_r02p50/local/")
    # sim_folder = os.path.expanduser("outputs/extra_rgn_out/trav0p34/local/")
    # sim_prefixes = ["sir_k{}_l0p00/".format("{:05.2f}".format(k).replace(".", "p")) for k in k_list_01]
    # full_prefixes = [sim_folder + name for name in sim_prefixes]
    # strat_dict["ST Local"] = StrategyBunch(full_prefixes, num_exec)

    #
    # -------------------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------------------
    #

    print("Calculating metrics...")
    metrics_t0 = time.time()

    def import_and_calc(i_sim, full_prefix):

        # sim_df_list = []  # Collects results over different simulations (k = 3, 6, ...)

        # Manipulates the prefix to provide info for new files and df indexing
        sim_dir = os.path.dirname(full_prefix)
        sim_prefix = os.path.basename(full_prefix)
        # Clause for when simulations are split into folders
        if sim_prefix == "":
            sim_prefix = os.path.basename(sim_dir)  # + SEP
            sim_dir = os.path.dirname(sim_dir)

        metrics_fname = sim_dir + SEP + "metrics/" + sim_prefix + "_metrics.csv"

        # -------------------------------------
        # PRODUCTION OF METRICS - either by calculation or import
        # Alternative to try: use try except to calculate if import metrics are not found.
        s_t0 = time.time()
        if try_to_import_metrics:
            # # IMPORT FROM FILES
            sim_df, input_dict = import_simulation_metrics(metrics_fname)
        else:
            # CALCULATE AND EXPORT TO FILES
            sim_df = calc_simulation_metrics(full_prefix, current_num_exec, use_metrics)
            export_simulation_metrics(sim_df, metrics_fname)
        s_tf = time.time()

        # sim_df_list.append(sim_df)

        # Screen feedback
        print("    {} ({:0.3f} s)".format(sim_prefix, s_tf - s_t0), end="\n")
        sys.stdout.flush()

        return sim_df

    # -----------------
    # ACTUAL IMPORT EXECUTION
    pool = ProcessPool(ncpus=num_processes)

    strat_df_list = []  # Collects results over different strategies (global, local, ...)

    for strategy_name, strat_bunch in strat_dict.items():

        full_prefixes = strat_bunch.full_prefixes
        current_num_exec = strat_bunch.num_exec

        print(strategy_name, end=":\n")
        b_t0 = time.time()

        # DO STUFF
        # for i_sim, full_prefix in enumerate(full_prefixes):
        sim_df_list = pool.map(lambda x: import_and_calc(x[0], x[1]), list(enumerate(full_prefixes)))
        #----

        b_tf = time.time()

        print()
        print("\t - Time: {} ({:0.3f} s)".format(seconds_to_hhmmss(b_tf - b_t0), b_tf - b_t0))

        strat_df = make_strategy_df_from_data(sim_df_list, sim_prefixes)

        strat_df_list.append(strat_df)

    # strat_df_list = pool.map(lambda x: import_and_calc(x[0], x[1]), list(strat_dict.items()))


    # Metric calculation time feedback
    metrics_tf = time.time()
    metrics_dt = metrics_tf - metrics_t0
    print("- Time calculating/importing metrics: {} ({:0.3f} s)".format(seconds_to_hhmmss(metrics_dt), metrics_dt))

    if stop_after_calc_metrics:
        print("-- You told me to stop after calculating the metrics. See ya. --")
        exit()

    # This dict contains all the calculated metrics in the most grained level - i.e., executions, nodes, etc.
    thebig_df = make_thebig_df_from_data(strat_df_list, strat_dict.keys())

    # ------------------------
    # LOADS THE NETWORK ONLY ONCE, from a sample file
    # Change if single exec
    sample_file = gen_sim_fname(list(strat_dict.values())[0].full_prefixes[0], "I")
    # print(sample_file)

    zipped = possibly_unzip_file(sample_file)
    sample_input_dict = read_config_file(sample_file)
    if zipped:
        remove_file(sample_file)

    g = load_metapop_from_input_dict(sample_input_dict)
    seed_node = get_seed_node(sample_input_dict)
    nodes = list(g.nodes())  # Globally used list of nodes

    # -----------------------
    # NODESET DEFINITION
    # DON'T USE IF STRATEGIES HAVE DIFFERENT SEEDS. Silently bad results!!
    nodelists = get_nodelists_by_shortestpath(g, seed_node) + [nodes]  # By regular distance to seed
    weightlists = None  # Weights of each node to the aggregation. None for arithmetic average.

    names = ["Seed", "1 step"] + ["{} steps".format(i) for i in range(2, len(nodelists)-1)] + ["All nodes"]
    # nodeset_dict = {name: nodeset for name, nodeset in zip(names, nodelists)}

    # -----------------------------
    # AGGREGATION OF METRICS AND STATISTICAL CALCULATIONS
    print("Aggregating and averaging over nodesets...")
    nodeset_aggregated_df = \
        aggregate_metrics_by_nodesets(thebig_df, nodelists, names, weightlists=weightlists,
                                      use_metrics=None)

    print("Aggregating over executions...")
    agg_t0 = time.time()
    exec_aggregated_df = aggregate_executions_with_statistics(nodeset_aggregated_df)
    agg_tf = time.time()
    print(" - Time aggregating over executions{} ({:0.5f}s)"
          "".format(seconds_to_hhmmss(agg_tf - agg_t0), agg_tf - agg_t0))

    # Checkpoints the final processed data into a single file with the final df
    if do_export_final_df:
        print("Exporting final df to file: '{}'".format(final_df_fname))
        export_final_df(exec_aggregated_df, final_df_fname)

    # Now exec_aggregated df has the average (and std) of each metrics, for each nodeset... and other levels.

    # ------------------------------
    # PLOT THE FEATURE PARAMETRIC CURVES
    # Also check the notebook 'plot_execwise_features.ipynb', devoted just to plot.

    # Some useful things
    num_axes = len(nodelists)
    max_cols = 3

    fig, axes = make_axes_list(num_axes, max_cols=max_cols, ax_height=3.5)
    plot_feature_curves(exec_aggregated_df, fig, axes, plot_metrics=use_metrics[:2],
                        max_cols=max_cols)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
