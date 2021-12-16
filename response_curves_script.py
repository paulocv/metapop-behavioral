"""
A simple python script to run simulations from a single input dict, but
with some different parameters (reac exponent and long termness).

Roughly, a hardcoded paramvar.

You can change use_strategies and the stract_dict to define the parameters
that are varied and overwritten over the original input dict.

Used a temporary file identified by sim_prefix, thus you can run this script
multiple simultaneous times, provided that each has a different sim_prefix.

Usage
---------------------
python response_curves_script.py [input_file] [output_folder] [num_processes]

Optional flag -p specifies the python executable command.
Currently, if flags are used, then NUM_PROCESSES IS MANDATORY!!

num_processes is optional.
"""

import sys
import os
from toolbox.file_tools import read_config_file, read_argv_optional, write_config_string, SEP, make_folder, \
    get_bool_from_dict, read_flag_argument
import datetime
import random as rnd

TMP_DIR = "tmp/"
RUN_SCRIPT = "metapop_simulate.py"
STD_PYTHON_EXEC = "python"
USE_BASIC = True
STD_SPLIT_INTO_FOLDERS = True

# -------------
# Script to run the different activation strategies.

# Argv interpretation
input_file = sys.argv[1]
output_folder = sys.argv[2]
num_processes = read_argv_optional(3, str, default="")
# If flag options are passed, num_processes is mandatory!
python_exec = read_flag_argument(sys.argv, "-p", default=STD_PYTHON_EXEC)

# Strategies
use_strategies = []
strat_dict = {}
# k_list = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.]
# lt_list = [0.0]

# One k_list for each lt_list (change the loop command as well)
lt_dict = {
    0.0: [3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50., 60., 90.],  # ST
    1.0: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 12., 15., 20., 25., 30., 40., 50.],  # LT

    # 0.0: [9.0, 12., 15., 20., 25.],  # Short test version
    # 1.0: [5.0, 6.0, 9.0, 12., 15.],  # Short test version
}


# Constructs the dictinary of strategies
for lt, k_list in lt_dict.items():
# for i, lt in enumerate(lt_list):
    for k in k_list:
        strat_name = "k{:05.2f}_l{:0.2f}".format(k, lt).replace(".", "p")

        # Override input dictionary
        strat_dict[strat_name] = {"sd_reac_exponent": k, "sd_long_term_coef": lt}
        use_strategies.append(strat_name)

# Appends the 'basic' case (baseline model with no intervention)
if USE_BASIC:
    # Appends the basic case (no intervention)
    use_strategies.append("basic")
    strat_dict["basic"] = {"sd_reac_exponent": 0, "sd_long_term_coef": 0.0}


# Override input dictionary for each strategy
# strat_dict = {
#     "act_gg": {"sim_type": "act_global", "sd_globality_coef": "1"},
#     "act_lg": {"sim_type": "act_local",  "sd_globality_coef": "1"},
#     "act_gl": {"sim_type": "act_global", "sd_globality_coef": "0"},
#     "act_ll": {"sim_type": "act_local",  "sd_globality_coef": "0"},
# }


# ---------------------------
input_dict = read_config_file(input_file)
base_sim_prefix = input_dict["sim_prefix"]
split_into_folders = get_bool_from_dict(input_dict, "split_into_folders", std_value=STD_SPLIT_INTO_FOLDERS)
tmp_file = TMP_DIR + "tmp_" + base_sim_prefix + str(rnd.randint(0, 10000)) + ".in"
output_folder = output_folder + SEP if output_folder[-1] != SEP else output_folder

# For each strat
#   update input dict and sim_prefix
#   export dict to current sim tmp file
#   calls the simulation with tmp file

num_sim = len(use_strategies)
for i_sim, strat in enumerate(use_strategies):
    # Feedback
    print("\nSimulation {} of {}: {}".format(i_sim+1, num_sim, strat))

    # Sets the current parameters and strategy prefix to input dict
    input_dict.update(strat_dict[strat])
    full_sim_prefix = base_sim_prefix + "_" + strat
    if split_into_folders:
        full_sim_prefix += SEP
        make_folder(output_folder + full_sim_prefix)
    input_dict["sim_prefix"] = full_sim_prefix

    # Export to tmp file
    with open(tmp_file, "w") as fp:
        fp.write(write_config_string(input_dict))

    # Runs the simulation script
    out = os.system(python_exec + " {} {} {} {}".format(RUN_SCRIPT, tmp_file, output_folder,
                                                 num_processes))

    # Ctrl + C (stop execution)
    if int(out) == 2:
        break

os.system("rm " + tmp_file)
print(datetime.datetime.now())
