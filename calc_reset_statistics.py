""""""
import sys
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd

from calc_execwise_features import calc_simulation_metrics as load_and_calc_simulation_metrics  # Borrowed function
from nodewise_features_metapop import calc_node_metric, load_and_process_sim_data
from toolbox.file_tools import make_folder
from toolbox.plot_tools import set_color_cycle, colorbrewer_pf_02, stdfigsize


def main():

    # ---------------------------------------
    # INPUT PARAMETERS
    # ---------------------------------------

    # use_metrics = ("a_impulse", "peak_size", "outb_size", "rel_peak_time", "abs_herd_time", "rel_herd_time",
    #                "time_above_val")  # ST pair + extras
    use_metrics = ("num_outb_threshold", )

    num_exec = 50
    # sim_prefix = "outputs/sir_1k_outputs/reset/" \
    #              "rgn0p25-04-trav1e-3/k05p00_10cycles_eps1e-4/sir-reset"
    # sim_prefix = "outputs/sir_1k_outputs/reset/" \
    #              "rgn0p25-04-trav1e-3/condit_k05p00_10cycles_eps1e-4_histher0p20/sir-reset"
    # sim_prefix = "outputs/tests/resets_trav1p0e-2/sir-reset/"
    sim_prefix = "outputs/reset_sensitivity/01/sir-reset_k20p00_tau1p0e-3/"

    # --- Sim prefix can be overridden by an argv input
    if len(sys.argv) > 1:
        sim_prefix = sys.argv[1]

    act_thres = 1.E-3
    histher = 0.5

    yscale = "log"  # linear, log
    plot_style = "designer"  # "designer", "default"

    calc_and_export_df = False  # True = recalculate all and export / False = try to load from df.
    keep_exec_bunches = False  # If True, all sim data is loaded before calculation. Horribly bad for large sets.

    save_plots = True
    show_plots = False
    check_executions = True  # Remove executions that had no more a (given) small number of outbreaks
    inv_thres = 2  # Must have at least this number of outbreaks to be valid (if check_executions is True).
    should_remove_seed = True
    seed_node = 40  # ALWAYS CHECK HERE IF SOMETHING CHANGES
    should_remove_zero_outbs = False  # Removes entries with zero outbreaks

    report_each_exec = False  # Whether to print a number for each finished execution.
    plot_average = True

    #
    #
    #
    # -----------------------------------------
    # EXECUTION
    # -----------------------------------------
    #
    #
    #

    df_path = os.path.join(os.path.dirname(sim_prefix), "reset_metrics_df.csv")
    sim_name = os.path.basename(sim_prefix[:-1])

    if calc_and_export_df:
        # --- // Load and calculation - COMMENT THE WHOLE ELIF BLOCK TO LOAD FROM PREMADE DF FILE and 'export/import'
        print("Calculating metrics from raw data...")
        if keep_exec_bunches:
            xt0 = time.time()
            exec_bunches, i_exec_list = load_execution_bunches(sim_prefix, num_exec)
            xtf = time.time()
            print("Time importing exec data: {:0.7f} s".format(xtf - xt0))

            xt0 = time.time()
            sim_metrics_df = calc_simulation_metrics(exec_bunches, use_metrics)
            xtf = time.time()
            print("Time calculating metrics: {:0.7f} s".format(xtf - xt0))
        else:
            # exec_bunches = None
            xt0 = time.time()
            sim_metrics_df = load_and_calc_simulation_metrics(sim_prefix, num_exec, use_metrics,
                                                              critical_value=act_thres, histher=histher,
                                                              report=report_each_exec)
            xtf = time.time()
            print("Time importing data and calculating metrics: {:0.7f} s".format(xtf - xt0))
        # ---- \\

        # Checkpoint the executionwise calculated metrics
        export_df(sim_metrics_df, df_path)

    else:  # if not calc_and_export_df:
        sim_metrics_df = import_df(df_path)

    # ------ ------------
    # Extra (as of Alberto's request/suggestion)
    if check_executions:
        # Remove executions that had no more than inv_thres outbreaks
        invalid_executions = pick_invalid_executions(sim_metrics_df, minimum_attacked_nodes=inv_thres)
        remove_executions_from_df(sim_metrics_df, invalid_executions)

        print("Invalid executions (removed from dataset):")
        print(invalid_executions)

    if should_remove_seed:
        remove_nodes_from_df(sim_metrics_df, seed_node)

    if should_remove_zero_outbs:
        remove_zero_outbreaks(sim_metrics_df)

    # # --- Display the entire final df
    # pd.set_option("display.max_rows", None)
    # print(sim_metrics_df)

    # --- Statistic aggregation
    # Simple mean and std
    # test_df = aggregate_executions_with_statistics(sim_metrics_df, "exec")

    # AVERAGE OVER EXECUTIONS
    # A data frame with the average metrics over all nodes for each execution
    # avg_over_execs = sim_metrics_df.mean(level="exec")  # Deprecation warning
    avg_over_execs = sim_metrics_df.groupby(level="exec").mean()

    # A list of execution histograms for each node.
    # Signature: hist_list[i_ni] = pd.Series with the counts of each number of outbreaks.
    hist_list = make_integer_histogram_executions(sim_metrics_df, use_metrics[0], level_name="exec")

    # Overall occurrence histogram, for all nodes and executions
    total_hist = pd.concat(hist_list).groupby(level=0).sum()

    # --- Some cool feedback
    print("Overall histogram entries")
    print(total_hist)
    print("Average num outbreaks = {:0.4f}".format(calc_avg_num_outb_from_hist(total_hist)))

    # TODO - checkpoint here?

    # --- PLOTS AND STUFF
    setup_pyplot_style(plot_style)

    # fig and ax are generated inside the function for better control of each figure's style
    fig, ax = plot_overall_histogam(total_hist, yscale=yscale)
    # fig, ax = plot_overall_histogam(hist_list[0])

    # PLOTS THE HISTOGRAM OF AVERAGE NUMBER OF OUTBREAKS OVER ALL EXECUTIONS
    if "num_outb_threshold" in use_metrics:
        ea_fig, ea_ax = plot_exec_averaged_histogram(avg_over_execs, yscale=yscale, plot_average=plot_average)
        if save_plots:
            figname = os.path.join("tmp_figs", "num-outb_exec-avg") + "_" + sim_name
            if should_remove_seed:
                figname += "_noseed"
            if should_remove_zero_outbs:
                figname += "_nozeros"
            ea_fig.savefig(figname + ".png")
            ea_fig.savefig(figname + ".pdf")

    # # Prints average number of outbreaks in each execution
    # for a in avg_over_execs.iterrows():
    #     print(a[1])
    # print()

    fig.tight_layout()

    if save_plots:
        figname = os.path.join("tmp_figs", "num-outb_overall") + "_" + sim_name
        if should_remove_seed:
            figname += "_noseed"
        if should_remove_zero_outbs:
            figname += "_nozeros"
        fig.savefig(figname + ".png")
        fig.savefig(figname + ".pdf")

    if show_plots:
        plt.show()


def load_execution_bunches(sim_prefix, num_exec):
    """ Loads the executions of a single simulation.
    Returns a list of execution bunches (SimBunch objects).

    If calc_metrics
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

    # Main execution loading
    bunches = []
    for i_exec in i_exec_list:
        bunches.append(load_and_process_sim_data(sim_prefix, i_exec=i_exec, g=g, nodes=nodes))

    return bunches, i_exec_list


def calc_simulation_metrics(exec_bunches, use_metrics, i_exec_list=None):
    """ Calculates, for each execution dataset of a simulation, the required epidemic metrics.

    Parameters
    ----------
    exec_bunches : list
        List of pre-loaded execution bunches from a simulation. List of SimBunch objects.
    use_metrics : sequence of str
        Names of the metrics to be calculated, as accepted in 'calc_node_metric'.
    i_exec_list : list
        Optional. The sequence of indexes of the executions. Must match the size of exec_bunches.
        If not informed, it is simply set to [0, 1, 2, ..., num_exec], extracted from len(exec_bunches).
    """
    # If not informed, i_exec_list is set to the first integers.
    if i_exec_list is None:
        i_exec_list = list(range(len(exec_bunches)))

    # Gets node id list from first execution
    nodes = exec_bunches[0].nodes

    # Allocates containers for the calculated metrics
    index = pd.MultiIndex.from_product((i_exec_list, nodes), names=("exec", "node"))
    # noinspection PyTypeChecker
    sim_metrics_df = pd.DataFrame(index=index,  columns=use_metrics, dtype=(float, float))  # Signature: df[i_exec, ni]

    # Loops over execution files.
    for i_exec, exec_bunch in enumerate(exec_bunches):

        # For each metric, calculates in all nodes
        for metric in use_metrics:
            # Metric calculation command
            d = calc_node_metric(exec_bunch, metric, monitor_states=("I", "R"), monitor_value=0.01,
                                 critical_value=1.E-3, histher=0.8)  # For num_outbreaks, critical_value is used.

            # Storing of the metrics into the multiindex dataframe
            # sim_metrics_df.xs(i_exec, level="exec")[metric] = d["array"]
            sim_metrics_df.loc[(i_exec, ), metric][:] = d["array"][:]

    return sim_metrics_df


def aggregate_executions_with_statistics(df, level_name="exec"):
    """
    For each simulation, calculates the average over a given level, as well as other statistical metrics.

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

    return grouped.agg([np.mean, np.std])


def make_integer_histogram_executions(df, metric, level_name="exec"):
    """
    For an execution set, calculates a histogram of the data over a given level and for a given metrics.
    Assumes unique values, so this is NOT A CONTINUOUS VARIABLE HISTOGRAM.

    Returns
    -------
    A list of data pd.Series with the counts of occurrences of each number of outbreaks. Each item is a node.
    Expected signature:
        count_series_list[i_ni] = pd.Series of {num_outbreak: occurrence_count}
    """
    xt0 = time.time()
    levels_to_groupby = list(df.index.names)
    levels_to_groupby.remove(level_name)
    grouped = df.groupby(level=levels_to_groupby)

    # Counts unique occurrences for each node.
    count_series_list = []
    for ni, df in grouped:
        dropped = df[metric].droplevel(level=levels_to_groupby)  # This is an agnostic removal of grouped levels.
        count_series_list.append(dropped.value_counts(sort=False))

    xtf = time.time()
    print("Time making nodewise histograms: {:0.6f} s".format(xtf - xt0))
    return count_series_list


def setup_pyplot_style(type="default"):

    if type == "default":
        # --- First style used, green bars, etc
        plt.style.use("mystyle_02")
        set_color_cycle(colorbrewer_pf_02)

        # Specific style parameters
        plt.rcParams["xtick.top"] = "off"
        plt.rcParams["xtick.bottom"] = "off"
        # plt.rcParams["ytick.right"] = "off"

    elif type == "designer":
        # --- To match Yamir's designer style
        plt.style.use("mystyle_02")
        set_color_cycle(["#f492a5"])

        # Hide the right and top spines
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False

        # Only show ticks on the left spines
        mpl.rcParams["xtick.top"] = False
        mpl.rcParams["xtick.bottom"] = False
        mpl.rcParams["ytick.right"] = False

        # Sets width of remaining spines and ticks
        spines_width = 0.896
        mpl.rcParams["axes.linewidth"] = spines_width
        mpl.rcParams["ytick.major.width"] = spines_width
        mpl.rcParams["ytick.minor.width"] = spines_width

        # Bar


def plot_overall_histogam(total_df, yscale="linear", normalize=True):
    """Histogram of outbreak counts for all executions and nodes."""

    fig, ax = plt.subplots(figsize=stdfigsize(scale=0.8, xtoy_ratio=1.61))

    if normalize:
        norm = total_df.sum()
        ylabel = "Normalized frequency"
    else:
        norm = 1.
        ylabel = "Frequency"

    # Overall average over nodes and executions plot
    ax.bar(total_df.index, total_df / norm)

    ax.set_xticks(total_df.index)
    ax.tick_params(axis="x", which="both", length=0)  # Removes x ticks, leaving the labels on

    ax.set_xlabel("Number of outbreaks")
    ax.set_ylabel(ylabel)

    ax.set_yscale(yscale)

    return fig, ax


def plot_exec_averaged_histogram(avg_over_execs, bins=10, yscale="linear", normalize=True, plot_average=True):
    """
    Parameters
    ----------
    avg_over_execs : pd.dataFrame
    """

    fig, ax = plt.subplots(figsize=stdfigsize(scale=0.8, xtoy_ratio=1.61))

    # ----------- USES NUMPY HISTOGRAM , THEN MATPLOTLIB BAR PLOT
    # Calc histogram using numpy
    hist_array, bin_edges = np.histogram(avg_over_execs["num_outb_threshold"], bins)

    if normalize:
        norm = np.sum(hist_array)
        ylabel = "Normalized frequency"
    else:
        norm = 1.
        ylabel = "Frequency"

    ax.bar(bin_edges[:-1], hist_array.astype(np.float32) / norm, align="edge",
           width=bin_edges[1:]-bin_edges[:-1] - 0.002)  # Tiny gap between them

    # Shows the average on the plot
    if plot_average:
        avg = avg_over_execs["num_outb_threshold"].to_numpy().mean()
        ax.text(0.8, 0.88, "mean = {:0.1f}".format(avg), transform=ax.transAxes, fontdict={"size": 18})
        # plt.text()


    # ------------ DIRECTLY USES pd.DataFrame.hist (no normalization possible, only 'density', which is not the same.
    # # Can't normalize this way:
    # hist = avg_over_execs.hist(column="num_outb_threshold", bins=bins, ax=ax, grid=False)

    ax.set_title(None)
    ax.set_xlabel("Average number of outbreaks")
    ax.set_ylabel(ylabel)

    ax.set_yscale(yscale)

    fig.tight_layout()

    return fig, ax


def export_df(df, fname):
    make_folder(os.path.dirname(fname), silent=True)
    df.to_csv(fname, sep=";")


def import_df(fname):
    """Reads data written with 'export_df'."""
    return pd.read_csv(fname, index_col=[0, 1], sep=";", skipinitialspace=True, header=[0])


def pick_invalid_executions(metrics_df, minimum_attacked_nodes=1):
    """Detects executions whose number of nodes that had at least one outbreak is smaller than a given
    threshold.
    """
    # Count the number of nodes that had at least one outbreak for each exec.
    num_attacked_nodes = np.empty(len(metrics_df), dtype=int)
    i_exec_array = np.empty(len(metrics_df), dtype=int)
    invalid_executions = list()
    print("------------\nCounting the nodes that had at least one outbreak")
    for i, (i_exec, exec_df) in enumerate(metrics_df.groupby(level="exec")):
        count = np.sum(exec_df["num_outb_threshold"] > 0)  # Performs the count of attacked nodes
        num_attacked_nodes[i] = count
        i_exec_array[i] = i_exec  # Just in case i_exec is not sequential

        # Criterion to determine the validity of an execution
        if count < minimum_attacked_nodes:
            invalid_executions.append(i_exec)

    return invalid_executions


def remove_executions_from_df(df, i_exec_list):
    """Removes all entries of a df corresponding to a given set of execution indexes.
    Changes are made in place.

    Assumes a df with a multiindex structure with levels: (exec, node)
    """
    df.drop(labels=i_exec_list, level="exec", inplace=True)


def remove_nodes_from_df(df, nodes_list):
    """
    Assumes a df with a multiindex structure with levels: (exec, node)
    """
    df.drop(labels=nodes_list, level="node", inplace=True)


def remove_zero_outbreaks(df):
    """
    Assumes a column named "num_outb_threshold"
    """
    to_drop = df[df["num_outb_threshold"] == 0].index
    df.drop(to_drop, inplace=True)


def calc_avg_num_outb_from_hist(hist_df, field="num_outb_threshold"):
    """
    Parameters
    ----------
    hist_df : pd.DataSeries
    """
    vals = hist_df.index.values
    weights = hist_df.values

    norm = np.sum(weights)
    if norm == 0.0:
        raise ValueError("Hey, total number of outbreaks on the histogram is zero. This could cause a math error")

    return np.sum(vals * weights) / norm


if __name__ == "__main__":
    main()
