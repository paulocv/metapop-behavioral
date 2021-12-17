"""
Author: Paulo Cesar Ventura da Silva

Usage
-----
python metapop_simulate.py [[path_to_input_file]] [[path_to_output_folder]] [[num_process]]

The number is used to parallelize executions. If not informed, uses all available
CPU threads.

Example input file
------------------
SIMULATION PARAMETERS
> sim_type = act_global  # basic, soc_dist, act_local, act_global
> num_exec = 4
> tmax = 6E3

----
EPIDEMIC MODEL PARAMETERS
> model_class = MetapopSEIR
> mu = 1 / 5.1
> nu = 1 / 5.2  # For SEIR
> r0 = 2.5  # beta = mu * ro

# Social distancing parameters
> sd_reac_exponent = 3
> sd_long_term_coef = 1
> sd_globality_coef = 1
> act_threshold = 0.01  # For 'act_' sim types.
> act_long_term_coef = 1

----
POPULATION PARAMETERS
# > pop_type = ?use this?
> pop_path = networks/path-graph/n10_tr1_N1E6_01
> travel_fac = 0.1

# If node and edges file have not the same prefix, use this !!AND!! Comment pop_path.
> pop_node_path = networks/path-graph/n10_tr1_N1E6_01.csv
> pop_edge_path = networks/path-graph/n10_tr1_N1E6_01.edgl

INITIAL CONDITIONS
> init_mode = infected_dict  # infected_dict
> init_data = {0: 10}

----
MISC
> sim_prefix = file_test
---------------------------------- end of input file

"""

from sim_modules.metapopulation import Metapop
from sim_modules.models import MODELS, MetapopSIR, MetapopSEIR
from toolbox.network_gen_tools import load_network_with_data
from toolbox.file_tools import make_folder, write_config_string, list_to_csv, \
    HEADER_END, read_config_file, read_argv_optional, get_folder_name_from_argv, \
    float_as_frac, get_bool_from_dict, read_optional_from_dict, zip_file, tar_folder
import datetime
import sys
import os
import numpy as np
# import pathos.multiprocessing as mp
import pathos.pools as pp
import matplotlib.pyplot as plt
import time

EXPORT = True
WRITE_HEADER = True
PLOT = False
STD_EXPORT_EXECUTIONS = False
STD_EXPORT_A_ARRAYS = False
STD_ZIP_OUTPUTS = True
STD_TAR_OUTPUTS = True

# -----------------------------------------
# Inputs and Parameters
input_dict = read_config_file(sys.argv[1])
output_folder = get_folder_name_from_argv(2, argi_check=False)
num_processes = read_argv_optional(3, int)



# SIMULATION PARAMETERS
sim_type = input_dict["sim_type"]
num_exec = int(input_dict["num_exec"])
tmax = int(float(input_dict["tmax"]))  # Converts twice to accept 1E5 notation.

# EPIDEMIC MODEL PARAMETERS
model_name = input_dict["model_class"]
if model_name in MODELS:
    model_class = eval(model_name)
else:
    raise ValueError("Hey, model '{}' was not found in MODELS dict: \n{}"
                     "".format(model_name, list(MODELS.keys())))


# MISC
sim_prefix = input_dict["sim_prefix"]
export_executions = get_bool_from_dict(input_dict, "export_executions",
                                       std_value=STD_EXPORT_EXECUTIONS)
export_a_arrays = get_bool_from_dict(input_dict, "export_a_arrays",
                                     std_value=STD_EXPORT_A_ARRAYS)
zip_outputs = get_bool_from_dict(input_dict, "zip_outputs",
                                 std_value=STD_ZIP_OUTPUTS)
tar_outputs = get_bool_from_dict(input_dict, "tar_outputs",
                                 std_value=STD_TAR_OUTPUTS)

# -------------------------------------------------------------------------
# ------------------  Simulation -----------------------
# -------------------------------------------------------------------------


print("\n------------------------------------")
print(datetime.datetime.now())
# PRINT HERE SOME BASIC FEATURES of the simulation


# Loads population from file.
# Currently, can only read a file. This helps the posterior analysis
# on jupyter.
if "pop_path" in input_dict:
    pop_path = input_dict["pop_path"]
    node_path = pop_path + ".csv"
    edge_path = pop_path + ".edgl"

# Node and edge files with different names
else:
    node_path = input_dict["pop_node_path"]
    edge_path = input_dict["pop_edge_path"]


def generate_pop():
    # Currently using only the path mode
    # Other ways are still in 'hardcode_simulate.py'

    g = load_network_with_data(node_path, edge_path,
                               edge_attrs=["weight"])

    g = Metapop(g)
    g.make_travel_arrays(dtype=float)

    return g


# ------------------------------------
# Demonstration of the characteristics of a sampled population
# Also used during data analysis
test_g = generate_pop()
num_nodes = len(test_g)
num_indiv = test_g.total_pop_size()


# ----------------------------
# Model creator function

def generate_model(g):
    # args = tuple()
    mu = float_as_frac(input_dict["mu"])
    nu = float_as_frac(input_dict["nu"])
    r0 = float(input_dict["r0"])
    beta = r0 * mu

    if model_class is MetapopSIR:
        args = (beta, mu)

    elif model_class is MetapopSEIR:
        args = (beta, mu, nu)

    else:
        raise ValueError("Hey, model class not included. \n{}".format(model_class))

    return model_class(g, *args)


class SimResBunch(object):
    """Joins the results of a single execution in a bunch.
    The advantage is that multiple strategies, which give different outcomes,
    can be handled with the same object."""

    # noinspection *??
    def __init__(self, statecount=None, t_outbreak=None, a_array=None, local_a_array=None,
                 events=None, local_events=None, outb_size=None,
                 total_outb_size=None):

        self.statecount = statecount
        self.t_outbreak = t_outbreak
        self.a_array = a_array  # Used for either local or global, agnosticly
        # self.local_a_array = local_a_array
        self.events = events
        self.local_events = local_events
        self.outb_size = outb_size
        self.total_outb_size = total_outb_size


# -----------------------------------------
# Model simulation
def execute_sim(i_ex):

    out = SimResBunch()

    # Initial conditions
    init_mode = input_dict["init_mode"]
    init_data = input_dict["init_data"]

    travel_fac = float(input_dict["travel_fac"])

    # Detection ratio handling. Read globally to make it optional and usable for threshold correction.
    # Besides changing the social distancing term, it divides the thresholds (which is
    # easier then multiplying the numbers of cases in each sim function).
    detec_ratio = read_optional_from_dict(input_dict, "detec_ratio",
                                          standard_val=1.,
                                          typecast=float)

    # ---------------
    g = generate_pop()
    model = generate_model(g)

    # Detection ratio cannot be zero, because it divides the thresholds.
    if detec_ratio < 1 / g.total_pop_size():
        raise ValueError("Hey, detec rate cannot be zero (or negative). To make a "
                         "simulation with no social distancing or detection of "
                         "cases, use '> sim_type = basic'.")

    model.initialize_states(init_mode, init_data)

    np.random.seed()

    # -----------------------
    # SIMULATION COMMAND
    # Basic simulation (no behavioral response)
    if sim_type == "basic":

        out.statecount, out.t_outbreak = \
            model.simulate_basic(tmax, step_kwargs={"travel_fac": travel_fac})

    # Simple social distancing
    elif sim_type == "soc_dist":
        step_kwargs = {
            "reac_exponent": float(input_dict["sd_reac_exponent"]),
            "long_term_coef": float(input_dict["sd_long_term_coef"]),
            "globality_coef": float(input_dict["sd_globality_coef"]),
            "detec_ratio": detec_ratio,
            "travel_fac": travel_fac,
        }
        out.statecount, out.t_outbreak, out.a_array = \
            model.simulate_basic(tmax, step_function="step_socialdist",
                                 step_kwargs=step_kwargs, get_a_array=True)

    # Locally activated social distancing
    elif sim_type == "act_local":
        step_kwargs = {
            "reac_exponent": float(input_dict["sd_reac_exponent"]),
            "long_term_coef": float(input_dict["sd_long_term_coef"]),
            "globality_coef": float(input_dict["sd_globality_coef"]),
            "detec_ratio": detec_ratio,
            "travel_fac": travel_fac,
            "act_thres": float(input_dict["act_threshold"]) / detec_ratio,
            "act_long_term_coef": float(input_dict["act_long_term_coef"]),
        }

        out.statecount, out.t_outbreak, out.a_array = \
            model.simulate_basic(tmax, step_function="step_activation_local",
                                 step_kwargs=step_kwargs,
                                 init_node_attrs={"soc_dist_active": False},
                                 get_a_array=True)

    # Globally activated social distancing
    elif sim_type == "act_global":
        step_kwargs = {
            "reac_exponent": float(input_dict["sd_reac_exponent"]),
            "long_term_coef": float(input_dict["sd_long_term_coef"]),
            "globality_coef": float(input_dict["sd_globality_coef"]),
            "detec_ratio": detec_ratio,
            "travel_fac": travel_fac,
        }
        act_thres = float(input_dict["act_threshold"]) / detec_ratio

        out.statecount, out.t_outbreak, out.a_array = \
            model.simulate_activation_global(tmax, act_thres,
                                             reac_step_function="step_socialdist",
                                             reac_step_kwargs=step_kwargs,
                                             basic_step_kwargs={"travel_fac":
                                                                travel_fac})

    # Globally activated uniform contact reduction
    elif sim_type == "act_uniform_global":
        step_kwargs = {
            "a": float(input_dict["uniform_a"]),
            "travel_fac": travel_fac,
        }
        act_thres = float(input_dict["act_threshold"]) / detec_ratio

        out.statecount, out.t_outbreak, out.a_array = \
            model.simulate_activation_global(tmax, act_thres,
                                             reac_step_function="step_uniform",
                                             reac_step_kwargs=step_kwargs,
                                             basic_step_kwargs={"travel_fac":
                                                                travel_fac})

    # Global reset soc. dist. strategy (NO HISTHERESIS)
    # Only the global monitored number of cases are reset, though it is
    # possible to partly use the (not reset) local prevalence to rho_ef.
    elif sim_type == "reset_global":

        step_kwargs = {
            "reac_exponent": float(input_dict["sd_reac_exponent"]),
            "long_term_coef": float(input_dict["sd_long_term_coef"]),
            "globality_coef": float(input_dict["sd_globality_coef"]),
            "detec_ratio": detec_ratio,
            "travel_fac": travel_fac,
        }
        if model_class is MetapopSEIR:
            # monitor_states = ("I", "E")  # Not desirable
            monitor_states = ("I",)
        else:
            monitor_states = ("I",)

        # Threshold as int or float, each one interpreted differently.
        try:
            reset_threshold = int(input_dict["reset_threshold"])
            # Corrects by detection factor while maintaining int type.
            reset_threshold = round(reset_threshold / detec_ratio)
        except ValueError:
            reset_threshold = float(input_dict["reset_threshold"]) / detec_ratio
        max_cycles = int(input_dict["max_cycles"])

        out.statecount, out.t_outbreak, out.a_array, i_t_resets = \
            model.simulate_reset_global(tmax, init_mode="infec_dict", init_data=init_data,
                                        reset_threshold=reset_threshold,
                                        max_global_cycles=max_cycles,
                                        reac_step_kwargs=step_kwargs,
                                        basic_step_kwargs={"travel_fac": travel_fac},
                                        monitor_states=monitor_states,
                                        )
        out.events = {"reset": i_t_resets}

    # Conditional (two way) threshold, global social distancing strategy
    elif sim_type == "condit_uniform_global":

        step_kwargs = {
            "a": float(input_dict["uniform_a"]),
            "travel_fac": travel_fac,
        }

        monitor_states = ("I",)

        # Threshold as int or float, each one interpreted differently.
        try:
            reset_threshold = int(input_dict["reset_threshold"])
            # Corrects by detection factor while maintaining int type.
            reset_threshold = round(reset_threshold / detec_ratio)
        except ValueError:
            reset_threshold = float(input_dict["reset_threshold"]) / detec_ratio
        max_cycles = int(input_dict["max_cycles"])
        histher = read_optional_from_dict(input_dict, "histher", standard_val=None, typecast=float)

        out.statecount, out.t_outbreak, out.a_array, i_t_resets, i_t_deacts = \
            model.simulate_condit_global(tmax, init_mode="infec_dict", init_data=init_data,
                                         reset_threshold=reset_threshold,
                                         max_global_cycles=max_cycles,
                                         reac_step_function="step_uniform",
                                         reac_step_kwargs=step_kwargs,
                                         basic_step_kwargs={"travel_fac": travel_fac},
                                         monitor_states=monitor_states,
                                         histher=histher, reset_rho0=False,
                                         )
        out.events = {"reset": i_t_resets, "deact": i_t_deacts}

    # Conditional (two way) threshold, global social distancing strategy
    elif sim_type == "condit_global":

        step_kwargs = {
            "reac_exponent": float(input_dict["sd_reac_exponent"]),
            "long_term_coef": float(input_dict["sd_long_term_coef"]),
            "globality_coef": float(input_dict["sd_globality_coef"]),
            "detec_ratio": detec_ratio,
            "travel_fac": travel_fac,
        }
        if model_class is MetapopSEIR:
            # monitor_states = ("I", "E")
            monitor_states = ("I",)
        else:
            monitor_states = ("I", )

        # Threshold as int or float, each one interpreted differently.
        try:
            reset_threshold = int(input_dict["reset_threshold"])
            # Corrects by detection factor while maintaining int type.
            reset_threshold = round(reset_threshold / detec_ratio)
        except ValueError:
            reset_threshold = float(input_dict["reset_threshold"]) / detec_ratio
        max_cycles = int(input_dict["max_cycles"])
        histher = read_optional_from_dict(input_dict, "histher", standard_val=None, typecast=float)

        out.statecount, out.t_outbreak, out.a_array, i_t_resets, i_t_deacts = \
            model.simulate_condit_global(tmax, init_mode="infec_dict", init_data=init_data,
                                         reset_threshold=reset_threshold,
                                         max_global_cycles=max_cycles,
                                         reac_step_kwargs=step_kwargs,
                                         basic_step_kwargs={"travel_fac": travel_fac},
                                         monitor_states=monitor_states,
                                         histher=histher,
                                         )
        out.events = {"reset": i_t_resets, "deact": i_t_deacts}

    else:
        raise ValueError("Hey, sim_type '{}' not understood.".format(sim_type))

    # --------------------
    # Early data processing, related to node properties.

    # Outbreak size normalized by node population.
    out.outb_size = {ni: g.num(ni, "R") / g.pop_size(ni) for ni in g}

    # Total outbreak size in the whole population
    out.total_outb_size = g.total_num("R") / g.total_pop_size()

    # ----------
    # Feedback and return
    print("Exec. {} of {}".format(i_ex+1, num_exec), end="\t")
    sys.stdout.flush()

    return out

# ------------------------------
# PARALLEL EXECUTION COMMAND


print("Simulations begun")
t0 = time.time()

# Parallel executions
# pool = mp.ProcessPool(nodes=num_processes)
pool = pp.ProcessPool(nodes=num_processes)
# pool = pp.ParallelPool(nodes=num_processes)
# pool = pp.SerialPool(nodes=num_processes)

# os.system('read -r -p "Ready to start sims, press enter\n" key')
sim_outputs = pool.map(execute_sim, range(num_exec))
print()

print("Time during simulations: {:0.3f} s".format(time.time() - t0))

# pool.clear()  # [2020/07/15 - This command decided to cause a segfault. If comment, more segfaults occur later.]


# ------------------------------------------
# Data processing

# DETERMINES NODE LIST FROM TEST G. Could be different.
nodes = test_g.nodes()

statelist = model_class.statelist
avg_statecount = {state: np.zeros((tmax+1, num_nodes), dtype=float)
                  for state in statelist}
avg_last_count = {state: np.zeros(num_nodes, dtype=float) for state in statelist}
avg_t_outbreak = 0.
max_t_outbreak = 0

avg_final_frac = {ni: 0. for ni in nodes}
avg_total_fin_frac = 0.

if export_a_arrays:
    # Agnostic to a_array's shape. #EDIT: NO MORE AGNOSTIC, IF a_array and local_a_array are different!!!
    avg_a_array = np.zeros_like(sim_outputs[0].a_array, dtype=float)
else:
    avg_a_array = None

for i_exec, sim_out in enumerate(sim_outputs):

    # Interprets outputs of simulations
    statecount = sim_out.statecount
    t_outbreak = sim_out.t_outbreak
    a_array = sim_out.a_array  # Can be None if not provided.
    final_fraction = sim_out.outb_size
    total_fin_frac = sim_out.total_outb_size
    events = sim_out.events
    local_events = sim_out.local_events

    # -----------------------------
    # Incorporation to the average simulation

    # Average time series
    for state in statelist:
        avg_statecount[state] += statecount[state]

    # Average a_array
    if export_a_arrays:
        if a_array is None:
            raise ValueError("Hey, you asked to export a_arrays but the simulation "
                             "did not provide them.")
        avg_a_array += a_array

    # Average outbreak size
    for state in statelist:
        avg_last_count[state] += statecount[state][-1]

    # Average final affected fraction
    for i, ni in enumerate(nodes):
        avg_final_frac[ni] += final_fraction[ni]

    # Average total final affected fraction
    avg_total_fin_frac += total_fin_frac

    # Average outbreak durations
    avg_t_outbreak += t_outbreak
    if t_outbreak > max_t_outbreak:
        max_t_outbreak = t_outbreak

    # ----------------------
    # Individual execution export
    if export_executions:
        # -------
        # Header function
        def construct_exec_header(other_info=None):
            header = ""

            # Basic info
            header += "Output from '{0}'.\n".format(__file__)
            header += "{}\n\n".format(datetime.datetime.now())

            header += "> i_exec = {:d}\n".format(i_exec)
            if other_info is not None:
                header += write_config_string(other_info)
                header += "\n"

            header += write_config_string(input_dict)
            header += "\n"

            # --------------------
            # Global results from analysis
            header += "> t_outbreak = {:0.0f} \n".format(t_outbreak)
            header += "> total_final_frac = {:f}\n".format(total_fin_frac)

            # Time stamps of (optional) events
            if events is not None:
                def write_if_exists(event_key, output_key):
                    try:
                        event_list = events[event_key]
                    except KeyError:
                        return ""
                    else:
                        return "> {} = {}\n".format(output_key, "[" + list_to_csv(event_list) + "]")

                # List of events (not all strategies return)
                header += write_if_exists("reset", "i_t_resets")
                header += write_if_exists("deact", "i_t_deacts")

            # -------------------
            # Header of nodes
            header += "\n"
            header += "> nodes = " + list_to_csv(nodes, sep="; ") + "\n"

            # ---------
            header += HEADER_END

            return header

        # -------------
        # Actual export

        for state in statelist:
            fname = output_folder + sim_prefix + "_{}_exec{:04d}.out".format(state, i_exec)
            exp_statecount = statecount[state][:t_outbreak + 2]
            # SLICES STATECOUNT UNTIL T_OUBREAK only.

            with open(fname, "w") as fp:

                if WRITE_HEADER:
                    fp.write(construct_exec_header({"state": state}))

                for i, line in enumerate(exp_statecount):
                    for count in line:
                        fp.write("{:09d}\t".format(count))
                    fp.write("\n")

            if zip_outputs:
                zip_file(fname, remove_orig=True)

        # Reaction time series a_array export
        if export_a_arrays:
            fname = output_folder + sim_prefix + "_{}_exec{:04d}.out".format("a", i_exec)

            # If the a series is global (i.e., a single value to all nodes), reshapes to a 2D array
            if len(a_array.shape) == 1:
                exp_statecount = a_array[:t_outbreak + 2].reshape((-1, 1))
            else:
                exp_statecount = a_array[:t_outbreak + 2]

            with open(fname, "w") as fp:
                if WRITE_HEADER:
                    fp.write(construct_exec_header({"state": "a"}))
                for i, line in enumerate(exp_statecount):
                    for count in line:
                        fp.write("{:10.8f}\t".format(count))
                    fp.write("\n")

            if zip_outputs:
                zip_file(fname, remove_orig=True)


# Final data normalization
for state in statelist:
    avg_statecount[state] /= num_exec
    avg_last_count[state] /= num_exec

if export_a_arrays:
    avg_a_array /= num_exec

for i, ni in enumerate(nodes):
    avg_final_frac[ni] /= num_exec

avg_total_fin_frac /= num_exec

avg_t_outbreak /= num_exec

print("Average t_outbreak: {}".format(avg_t_outbreak))
print("Max t_outbreak: {}".format(max_t_outbreak))
# print("Average affected fraction per node:")
# for i, r in enumerate(avg_last_count["R"]):
# for i, r in avg_final_frac.items():
#     print("{}: {:0.4f}".format(i, r))

print("Total affected pop. fraction: {:0.4f}".format(avg_total_fin_frac))


# -----------------------
# DATA EXPORTING
# -----------------------
# Simulation file
if EXPORT:

    # Creates output folder
    # out_dir = os.path.dirname(output_folder)
    if not os.path.isdir(output_folder):
        make_folder(output_folder)

    # HEADER CONSTRUCTION
    def construct_header(other_info=None):
        header = ""

        # Basic info
        header += "Output from '{0}'.\n".format(__file__)
        header += "{}\n\n".format(datetime.datetime.now())

        if other_info is not None:
            header += write_config_string(other_info)
            header += "\n"

        header += write_config_string(input_dict)
        header += "\n"

        # --------------------
        # Global results from analysis
        header += "> avg_t_outbreak = {:0.5f} \n".format(avg_t_outbreak)
        header += "> max_t_outbreak = {:d} \n".format(max_t_outbreak)
        header += "> avg_final_frac = {:f}\n".format(avg_total_fin_frac)

        # -------------------
        # Header of nodes
        header += "\n"
        header += "> nodes = " + list_to_csv(nodes, sep="; ") + "\n"

        # ---------
        header += HEADER_END

        return header


    # ---------------
    # Simulation time series export

    for state in statelist:
        fname = output_folder + sim_prefix + "_{}.out".format(state)
        exp_statecount = avg_statecount[state][:max_t_outbreak+2]

        with open(fname, "w") as fp:

            if WRITE_HEADER:
                fp.write(construct_header({"state": state}))

            for i, line in enumerate(exp_statecount):
                for count in line:
                    fp.write("{:12.5f}\t".format(count))
                fp.write("\n")

        if zip_outputs:
            zip_file(fname, remove_orig=True)

    # ----------------
    # Average reaction timeseries a_array
    if export_a_arrays:
        fname = output_folder + sim_prefix + "_{}.out".format("a")

        # If the a series is global (i.e., a single value to all nodes), reshapes to a 2D array
        if len(avg_a_array.shape) == 1:
            exp_statecount = avg_a_array[:max_t_outbreak + 2].reshape((-1, 1))
        else:
            exp_statecount = avg_a_array[:max_t_outbreak + 2]

        with open(fname, "w") as fp:

            if WRITE_HEADER:
                fp.write(construct_header({"state": "a"}))

            for i, line in enumerate(exp_statecount):
                for count in line:
                    fp.write("{:10.8f}\t".format(count))
                fp.write("\n")

        if zip_outputs:
            zip_file(fname, remove_orig=True)

# Finally make a tar and delete
if tar_outputs:
    sim_folder = os.path.dirname(output_folder + sim_prefix)
    tar_folder(sim_folder)


# ---------------
# Test: plot sim
if PLOT:
    # # Use this line to get a particular simulation
    # plot_statecount, t_outbreak, final_fraction, total_fin_frac = sim_outputs[0]

    # Use this chunk to get the average simulation instead
    plot_statecount = avg_statecount
    t_outbreak = avg_t_outbreak
    final_fraction = avg_final_frac
    total_fin_frac = avg_total_fin_frac

    # ------
    sum_states = np.zeros((max_t_outbreak + 1, num_nodes), dtype=int)
    num_cases = np.zeros((max_t_outbreak + 1, num_nodes), dtype=int)
    frac_cases = np.zeros((max_t_outbreak + 1, num_nodes), dtype=float)

    for t in range(max_t_outbreak + 1):
        for i, ni in enumerate(nodes):
            sum_states[t][i] = sum(plot_statecount[state][t][i] for state in statelist)
            cases = sum(plot_statecount[state][t][i] for state in ["I", "R"])
            num_cases[t][i] = cases
            frac_cases[t][i] = cases / test_g.pop_size(i)

    fig, ax = plt.subplots()
    # ax.plot(range(max_t_outbreak+1), sum_states)  # population sizes
    # ax.plot(range(max_t_outbreak+1), num_cases)  # Number of cases.
    ax.plot(range(max_t_outbreak+1), frac_cases)  # Regional fraction of infecteds
    # ax.plot(range(max_t_outbreak+1), plot_statecount["I"][:max_t_outbreak+1])
    if export_a_arrays:
        ax.plot(range(max_t_outbreak+1), avg_a_array[:max_t_outbreak+1])
    plt.show()

print(datetime.datetime.now())
