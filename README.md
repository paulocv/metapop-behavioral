# Epidemic simulation on metapopulation with behavioral responses
Source code of the simulations used in the work: "Modeling the effects of social distancing on the large-scale spreading of diseases".

Paper (Open Access): https://doi.org/10.1016/j.epidem.2022.100544
Preprint: https://arxiv.org/abs/2105.09697


## Python environment setup with Conda

We used Conda to setup Python environments for running the simulation and data analysis. A snapshot of the environment packages (with exact version numbers and builds) can be found at the file metapop_nx1_snapshot_2021-12-03.yml. It was created on a Ubuntu 20.04 OS.

**Important:** the simulations (on sim_modules folder) are implemented using version 1.9.1 of the Networkx library, as it presents considerably higher performance than Networkx 2.x. The 1.x and 2.x libraries are not mutually compatible, therefore you should also use version 1.x or update the code to be compatible with 2.x. Alternatively, we keep a Networkx 2.x implementation on folder "nx2_version_bkp", but it is potentially outdated.

For Linux OS, you can reconstruct the environment with the following command (assuming you already have Conda installed):

```
conda env create -f metapop_nx1_snapshot_2021-12-03.yml
```

or follow [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for instructions to create a conda env from a .yml file. For other operational systems, it is possible that you will face "package not found" due to the build numbers. You can still manually create the environment and install the packages, paying attention to their version numbers.

The analysis module (nodewise_features_metapop.py), on the other hand, use NetworkX 2.x. The recommended way to work is to create a clone of the metapop_nx1 environment, activate it and then type `pip install networkx==2.6.2`.


## Usage

### Simulations – implementation

The simulations are mostly defined by two modules inside the "sim_modules" directory:

* **metapopulation.py** – Defines the Metapop class, derived from the Networkx's Graph class. The class stores all the variables of a metapopulation, and defines methods to modify it before, during and after the simulations.

* **models.py** – This is where the epidemic models are defined. A base class MetapopModel defines complete simulation routines (e.g. `simulate_basic`, `simulate_reset_global`). The rules of the epidemic model are implemented as derived classes from MetapopModel, by implementing methods such as`epidemic_step_basic`. The module contains two concrete epidemic models: MetapopSIR and MetapopSEIR.


### Simulations – running

To run a simulation round with a single set of parameters, use `metapop_simulate.py` (see the code for usage instructions). There is an example input file on the directory "inputs".

Example of simulation call (make sure to have the Python enviroment setup first, otherwise import and other errors will pop up):

```
python metapop_simulate.py inputs/sample_input.in outputs/my_first_test/
```

Be aware that, if successful, this should use all the CPU threads available in your computer (up to 16). Alternativelly, you can specify the number of threads to be used as a third argument. For example, run:

```
python metapop_simulate.py inputs/sample_input.in outputs/my_first_test/ 4
```

to use only 4 CPU threads.

For figures 7 to 10 of our manuscript, we need to run simulations with multiple _k_ and _l_ values. For this purpose, you can use `response_curves_script.py`. Again, instructions are found in the source file. You can run it with the same example input file, for example: 

```
python response_curves_script.py inputs/sample_input.in outputs/my_second_test/ 4
```

This will take considerably more time, and will produce a large numbre of output files. 

### Data analysis

We have additional scripts to post-process our outputs from simulations. These are calc_execwise_features_parallel.py and calc_reset_statistics.py. Both use tools defined in "nodewise_features_metapop.py", which assumes that Networkx is of a 2.x version. 

The scripts are called without extra arguments. Please edit the inputs directly in the `main()` function of each script.


