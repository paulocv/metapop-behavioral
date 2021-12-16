from sim_modules.metapopulation import Metapop
from toolbox.file_tools import str_to_dict
import numpy as np

# Update this dict when a new model is created
MODELS = {
    "MetapopSIR": {},
    "MetapopSEIR": {},
}

REAC_KEY = "reac_term"  # Keyword for the reaction term a(I, r) in each node.
GLOBAL_RHO0_KEY = "global_rho0"


class MetapopModel(object):

    statelist = []
    transitions = []

    def __init__(self, g):
        """

        Parameters
        ----------
        g : Metapop
        """
        self.g = g
        self.g.statelist = self.statelist

    def initialize_states(self, mode, data):
        """Can be overriden for other algorithms."""
        initialize_infective_states(self.g, mode, data, healthy_state="S",
                                    infective_state="I")

    # ------------------------------
    # Calculation of effective probabilities
    def calc_pinf_basic(self, ni, beta, infective_state="I"):
        return calc_p_infection(beta, self.g.num(ni, infective_state),
                                self.g.pop_size(ni))

    def gen_calc_pinf_basic(self, beta, infective_state="I"):
        """Generator version of calc_pinf"""

        def pinf(ni):
            self.g.nodes[ni][REAC_KEY] = 1.
            return calc_p_infection(beta, self.g.num(ni, infective_state),
                                    self.g.pop_size(ni))

        return pinf

    def gen_calc_pinf_statelist(self, beta, infective_states=("I",)):
        """Generator version of calc_pinf, using various infective states."""
        pop_size = self.g.pop_size  # Function

        def pinf(ni):
            self.g.nodes[ni][REAC_KEY] = 1.
            return calc_p_infection(beta, self.g.num_in_statelist(ni, infective_states),
                                    pop_size(ni))

        return pinf

    def gen_calc_pinf_uniform(self, beta, a, infective_state="I"):
        """Generator for the infec. probab. with given a(I, R) value.
        """
        def pinf(ni):
            self.g.nodes[ni][REAC_KEY] = a
            return calc_p_infection(a * beta, self.g.num(ni, infective_state),
                                    self.g.pop_size(ni))

        return pinf

    def gen_calc_pinf_socialdist(self, beta, infective_state="I",
                                 short_term_state="I", long_term_state="R",
                                 reac_exponent=1.,
                                 long_term_coef=1.0, globality_coef=1.0,
                                 global_st_density=None, global_lt_density=None,
                                 local_rho0_key=None):
        """
        A generalization of the social distancig model by Eskyn et. al. 2019
        for metapopulations. In this version, both long term and short term
        strategies can be considered by adjusting long_term_coef. Also regional
        and global strategies are achieved by adjusting globality_coef.
        An offset to the global and local long term states can be added
        with parameters global_rho0 and local_rho0_key (the latter access a
        node attribute). If local_rho0_key is not informed, local offsets are
        all set to zero.

        [Function generator].

        Perceived prevalence:
        rho_ef = a*(y + b*(z - z0)) + (1-a)*(y_i + b*(z_i - z0_i))

        Effective beta (indiv transmission probab):
        beta_ef = beta * (1 - rho_ef)**k

        Where:
        a = globality_coef
        b = long_term_coef
        k = reaction_exponent
        y, y_i = global and local densities in short_term_state
        z, z_i = global and local densities in long_term_state
        z0, z0_i = global and local long term offsets

        For slight performance opt., you can inform the global densities of
        the short and long term states as global_st_density and global_lt_density.
        """
        # Reset value for the prevalence
        try:
            global_rho0 = self.g.graph[GLOBAL_RHO0_KEY]
        except KeyError:
            global_rho0 = 0.0

        # If not informed, global densities are calculated.
        if global_st_density is None:
            global_st_density = self.g.total_num(short_term_state) / self.g.total_pop_size()
        if global_lt_density is None:
            global_lt_density = self.g.total_num(long_term_state) / self.g.total_pop_size()

        # Local densities
        def calc_rho_st(ni):
            return self.g.num(ni, short_term_state) / self.g.pop_size(ni)

        def calc_rho_lt(ni):
            return self.g.num(ni, long_term_state) / self.g.pop_size(ni)

        # Local offset
        if local_rho0_key is None:
            # noinspection PyUnusedLocal
            def local_rho0(ni):
                return 0.
        else:
            def local_rho0(ni):
                return self.g.nodes[ni][local_rho0_key]

        def pinf(ni):
            # Perceived prevalence
            rho_ef = globality_coef * (global_st_density + long_term_coef * global_lt_density - global_rho0)
            rho_ef += (1.-globality_coef) * (calc_rho_st(ni) + long_term_coef * calc_rho_lt(ni) - local_rho0(ni))

            # Reaction term and effective beta
            a = (1. - rho_ef)**reac_exponent
            self.g.nodes[ni][REAC_KEY] = a
            beta_ef = beta * a

            return calc_p_infection(beta_ef, self.g.num(ni, infective_state),
                                    self.g.pop_size(ni))

        return pinf

    # --------------------------------------------------------
    # Step/Iteration functions

    def epidemic_step_basic(self):
        pass

    def epidemic_step_uniform(self, a):
        pass

    def epidemic_step_socialdist(self, reac_exponent=1, long_term_coef=1,
                                 globality_coef=1, local_rho0_key=None):
        pass

    def epidemic_step_activation_local(self, act_thres,
                                       reac_exponent=1, long_term_coef=1,
                                       globality_coef=1,
                                       act_long_term_coef=1):
        pass

    def motion_step_basic(self, travel_fac=1.):
        """"""
        for ni in self.g.nodes():
            travel_p = list(travel_fac * self.g.travel_array(ni) / self.g.pop_size(ni))
            travel_p.append(0.)  # Last point, ignored by multinomial

            # Determine the numbers of travelers in each state, to each neighbor.
            # Excludes last element (which is 'non travelers').
            for state in self.statelist:
                nums = np.random.multinomial(self.g.num(ni, state), travel_p)[:-1]
                self.g.set_tomove(ni, state, nums)

    def step_basic(self, travel_fac=1.):
        """Can be overriden if the model requires."""

        # Calculates and applies epidemic transitions
        self.epidemic_step_basic()

        for s1, s2 in self.transitions:
            self.g.apply_state_changes(s1, s2)

        # Calculates and applies motion rules
        self.motion_step_basic(travel_fac=travel_fac)
        for state in self.statelist:
            self.g.apply_moves(state)

        # Consistency check
        self.g.check_states_and_nums_consistency()

    # General step function with social distancing
    def step_uniform(self, a, travel_fac=1):
        """"""
        # Calculates and applies epidemic transitions
        self.epidemic_step_uniform(a)

        for s1, s2 in self.transitions:
            self.g.apply_state_changes(s1, s2)

        # Calculates and applies motion rules
        self.motion_step_basic(travel_fac=travel_fac)
        for state in self.statelist:
            self.g.apply_moves(state)

        # Consistency check
        self.g.check_states_and_nums_consistency()

    # General step function with social distancing
    def step_socialdist(self, reac_exponent=1, long_term_coef=1,
                        globality_coef=1, travel_fac=1, local_rho0_key=None):
        """"""
        # Calculates and applies epidemic transitions
        self.epidemic_step_socialdist(reac_exponent, long_term_coef,
                                      globality_coef, local_rho0_key=local_rho0_key)

        for s1, s2 in self.transitions:
            self.g.apply_state_changes(s1, s2)

        # Calculates and applies motion rules
        self.motion_step_basic(travel_fac=travel_fac)
        for state in self.statelist:
            self.g.apply_moves(state)

        # Consistency check
        self.g.check_states_and_nums_consistency()

    # Social distancing with local activation threshold.
    def step_activation_local(self, act_thres,
                              reac_exponent=1, long_term_coef=1,
                              globality_coef=1, act_long_term_coef=1,
                              travel_fac=1.):
        # Calculates and applies epidemic transitions
        self.epidemic_step_activation_local(act_thres,
                                            reac_exponent, long_term_coef,
                                            globality_coef, act_long_term_coef)
        for s1, s2 in self.transitions:
            self.g.apply_state_changes(s1, s2)

        # Calculates and applies motion rules
        self.motion_step_basic(travel_fac=travel_fac)
        for state in self.statelist:
            self.g.apply_moves(state)

        # Consistency check
        self.g.check_states_and_nums_consistency()

    # ------------------------
    # Stop conditions

    def stop_condition_basic(self, current_statecount):
        raise NotImplementedError

    # ------------------------------
    # Simulation functions

    def simulate_basic(self, tmax, init_mode=None, init_data=None,
                       step_function=None, step_kwargs=None,
                       init_node_attrs=None, get_a_array=False):
        """
        For simplicity and due to current goals, no transient time is
        considered

        Step function
        -------------
        In the basic simulation, a single type of model step function is used.
        It is specified as step_function argument, and can either be a class method
        or a string (with the exact name of the class method). Arguments are
        specified in step_kwargs, as a dictionary of keyword arguments.

        init_node_attrs is a dict of attributes and values to be set for each
        node at the beginning, which then may be used by some step functions.
        """
        # ---------------
        # Initialization

        # Population initialization
        self.g.make_travel_arrays()
        if init_mode is not None and init_data is not None:
            self.initialize_states(init_mode, init_data)

        # Initializes node auxiliary attributes
        if init_node_attrs is not None:
            for ni in self.g.nodes():
                self.g.nodes[ni].update(init_node_attrs)

        # Initialization of data collectors
        i_max = int(tmax / 1.)  # A future remainder for non-integer time steps dt != 1
        statecount = dict()
        for state in self.statelist:
            statecount[state] = np.zeros((i_max + 1, len(self.g)), dtype=int)

        current_statecount = {state: [self.g.num(ni, state) for ni in self.g.nodes()]
                              for state in self.statelist}

        a_array = np.ones((i_max + 1, len(self.g)), dtype=float) if get_a_array else None

        # Definition of the simulation step function
        if step_kwargs is None:
            step_kwargs = {}

        if type(step_function) is str:
            step_function = self.__getattribute__(step_function)
        elif step_function is None:
            step_function = self.step_basic
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(step_function))

        # Registers initial state to the counters
        for state in self.statelist:
            statecount[state][0] = [self.g.num(ni, state) for ni in self.g.nodes()]

        # --------------
        # Simulation execution
        t = 0.  # Shuts down PyCharm warning.
        i_0 = 1  # First vacant entry on time series.
        for i_t, t in enumerate(np.arange(tmax)):  # This allows for non-integer t

            # noinspection PyArgumentList
            step_function(**step_kwargs)

            # Registers new statecounts
            for state in self.statelist:
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]
                # Looks redundant, but is useful to test stop condition and, futurely,
                # implement a transient time sim.
                if get_a_array:
                    for i, ni in enumerate(self.g):
                        a_array[i_0 + i_t][i] = self.g.nodes[ni][REAC_KEY]
                    # a_array[i_0 + i_t] = (self.g.nodes[ni][REAC_KEY] for ni in self.g)

            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies all current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    if get_a_array:
                        a_array[j_t] = a_array[i_0 + i_t]
                break

        if get_a_array:
            return statecount, t, a_array
        else:
            return statecount, t

    def simulate_activation_global(self, tmax, act_thres,
                                   init_mode=None, init_data=None,
                                   basic_step_function=None, basic_step_kwargs=None,
                                   reac_step_function=None, reac_step_kwargs=None,
                                   act_long_term_coef=1,
                                   short_term_state="I", long_term_state="R"):
        """
        Simulates the activation of the social distancing mechanism after the
        perceived global prevalence reaches a certain threshold. The activation
        is simultaneous to all regions.

        Reaction Step function
        -------------
        The function to be used for model step with social distancing
        is specified as reac_step_function argument, and can either be a class method
        or a string (with the exact name of the class method). Arguments are
        specified in reac_step_kwargs, as a dictionary of keyword arguments.

        Acivation long term coefficient
        -------------
        The coefficient for long termness of the perceived prevalence.
        """
        # ---------------
        # Initialization

        # Population initialization
        self.g.make_travel_arrays()
        if init_mode is not None and init_data is not None:
            self.initialize_states(init_mode, init_data)

        # Initialization of data collectors
        i_max = int(tmax / 1.)  # A future remainder for non-integer time steps dt != 1
        statecount = dict()
        for state in self.statelist:
            statecount[state] = np.zeros((i_max + 1, len(self.g)), dtype=int)

        current_statecount = {state: [self.g.num(ni, state) for ni in self.g.nodes()]
                              for state in self.statelist}
        a_array = np.ones(i_max + 1, dtype=float)
        sample_node = list(self.g)[0]
        sample_node_data = self.g.nodes[sample_node]

        # Definition of the simulation step function
        # Basic step
        if basic_step_kwargs is None:
            basic_step_kwargs = {}

        if basic_step_function is None:
            basic_step_function = self.step_basic
        elif type(basic_step_function) is str:
            basic_step_function = self.__getattribute__(basic_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(basic_step_function))

        # Reaction step function
        if reac_step_kwargs is None:
            reac_step_kwargs = {}

        if reac_step_function is None:
            reac_step_function = self.step_socialdist
        elif type(reac_step_function) is str:
            reac_step_function = self.__getattribute__(reac_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(reac_step_function))

        # Registers initial state to the counters
        for state in self.statelist:
            statecount[state][0] = [self.g.num(ni, state) for ni in self.g.nodes()]

        # ---------------------------------
        # Simulation execution: first stage (no activation)
        t = 0.  # Shuts down PyCharm warning.
        i_t = 0  # Shuts down PyCharm warning.
        i_0 = 1  # First vacant entry on time series.
        for i_t, t in enumerate(np.arange(tmax)):  # This allows for non-integer t

            # Check for activation threshold here
            rho_ef = self.g.total_num(short_term_state)
            rho_ef += act_long_term_coef * self.g.total_num(long_term_state)
            rho_ef /= self.g.total_pop_size()
            if rho_ef > act_thres:
                break  # Goes to next stage

            # Basic step (no reaction)
            basic_step_function(**basic_step_kwargs)

            # Registers new statecounts
            for state in self.statelist:
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]
                # Looks redundant, but is useful to test stop condition and, futurely,
                # implement a transient time sim.
                # a_array[i_0 + i_t] = sample_node_data[REAC_KEY]  # Assumes a = 1 during first stage

            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies all current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    # a_array[j_t] = a_array[i_0 + i_t]  # Comment this to return ones as remaining.
                # break  # Instead returns, avoiding the next stage.
                return statecount, t, a_array

        # -----------------------------------
        # Simulation execution: second stage (social distance activated)
        t0 = t
        i_0 = i_t  # First vacant entry on time series.
        for i_t, t in enumerate(np.arange(t0, tmax)):

            # Step with social distance triggered
            # noinspection PyArgumentList
            reac_step_function(**reac_step_kwargs)

            # Registers new statecounts
            for state in self.statelist:
                # Looks redundant, but is useful to test stop condition and, futurely,
                # implement a transient time sim.
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]

                a_array[i_0 + i_t] = sample_node_data[REAC_KEY]

            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies all current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    a_array[j_t] = a_array[i_0 + i_t]  # Comment this to return ones as remaining.
                break

        return statecount, t, a_array

    def simulate_reset_global(self, tmax, init_mode=None, init_data=None,
                              basic_step_function=None, basic_step_kwargs=None,
                              reac_step_function=None, reac_step_kwargs=None,
                              long_term_state="R", monitor_states=("I", ),
                              reset_threshold=1.E-4,
                              max_global_cycles=None, run_last_cycle=True,
                              ):
        """
        Simulates the global social distancing mechanism with reset to the memory
        of the long term state (usually, removed state) to the effective
        prevalence. This is the fully global strategy, thus the resets
        also occur globally.

        Reaction Step function
        -------------
        The function to be used for model step with social distancing
        is specified as reac_step_function argument, and can either be a class method
        or a string (with the exact name of the class method). Arguments are
        specified in reac_step_kwargs, as a dictionary of keyword arguments.
        It must contain global_rho0 as an argument, which is the global offset.


        reset_threshold : triggers a memory reset whenever the monitored states
          go under this value (from above). Can either be int or float. If int,
          it is interpreted as a number of cases. If float, as a fraction of the
          total population.

        max_global_cycles : defines the number of resets allowed (plus 1).
            If not informed, no limit is defined, and resets will be done
            until another condition stops the simulation.
            If informed and run_last_cycle = True, simulation is continued
            with no social distancing.
            If informed and run_last_cycle = False, simulation is continued
            with social distancing, but no more resets are performed.
        """
        # ---------------
        # Initialization

        # Population initialization
        self.g.make_travel_arrays()
        if init_mode is not None and init_data is not None:
            self.initialize_states(init_mode, init_data)

        # Initialization of data collectors
        i_max = int(tmax / 1.)  # A future remainder for non-integer time steps dt != 1
        statecount = dict()
        for state in self.statelist:
            statecount[state] = np.zeros((i_max + 1, len(self.g)), dtype=int)

        current_statecount = {state: [self.g.num(ni, state) for ni in self.g.nodes()]
                              for state in self.statelist}
        a_array = np.ones(i_max + 1, dtype=float)
        sample_node = list(self.g)[0]
        sample_node_data = self.g.nodes[sample_node]

        # "Reset-strategy" variables
        self.g.graph[GLOBAL_RHO0_KEY] = 0.
        num_cycles = 0  # Counter of cycles
        glob_num_inf = self.g.total_num_in_statelist(monitor_states)
        # glob_num_next = glob_num_inf
        i_t_resets = []  # List of time indexes at which reset occurs

        # Converts a float threshold to int.
        if isinstance(reset_threshold, (np.floating, float)):
            reset_threshold = int(self.g.total_pop_size() * reset_threshold)

        if max_global_cycles is None:
            # No maximum
            max_global_cycles = np.inf

        # Definition of the simulation step function
        # Basic step
        if basic_step_kwargs is None:
            basic_step_kwargs = {}

        if basic_step_function is None:
            basic_step_function = self.step_basic
        elif type(basic_step_function) is str:
            basic_step_function = self.__getattribute__(basic_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(basic_step_function))

        # Reaction step function
        if reac_step_kwargs is None:
            reac_step_kwargs = {}

        if reac_step_function is None:
            reac_step_function = self.step_socialdist
        elif type(reac_step_function) is str:
            reac_step_function = self.__getattribute__(reac_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(reac_step_function))

        # Registers initial state to the counters
        for state in self.statelist:
            statecount[state][0] = [self.g.num(ni, state) for ni in self.g.nodes()]
        # a_array[0] = 1.  # For simplicity, first is regarded as one.

        # ---------------------------------
        # Simulation execution: loop of outbreak cycles
        t = 0.  # Shuts down PyCharm warning.
        i_t = 0  # Shuts down PyCharm warning.
        i_0 = 1  # First vacant entry on time series.

        for i_t, t in enumerate(np.arange(tmax)):  # This allows for non-integer t

            # Social distancing step function
            reac_step_function(**reac_step_kwargs)

            # Registers new statecounts
            for state in self.statelist:
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]
            # Registers the global a(I, R) using the sample node
            a_array[i_0 + i_t] = sample_node_data[REAC_KEY]

            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max+1):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    # a_array[j_t] = a_array[i_0 + i_t]  # Comment this to return ones as remaining.
                # break  # Instead returns, avoiding the next stage.
                return statecount, t,  a_array, i_t_resets

            # Check for reset threshold in the monitored prevalence
            glob_num_next = self.g.total_num_in_statelist(monitor_states)
            if glob_num_next < reset_threshold < glob_num_inf:
                # Update count of cycles, the prevalence offset, annotate i_t
                num_cycles += 1
                self.g.graph[GLOBAL_RHO0_KEY] = self.g.total_num(long_term_state) / self.g.total_pop_size()
                i_t_resets.append(i_t)

                # Checks max number of cycles
                if num_cycles == max_global_cycles:
                    if run_last_cycle:
                        # Goes to the next stage
                        break
                    else:
                        # Just allows the simulation to continue as it is
                        pass

            # Updates the current global number of infecteds (monit. states)
            glob_num_inf = glob_num_next

        # -----------------------------------
        # Simulation execution: last stage (no social distancing)
        t0 = t
        i_0 = i_t  # First vacant entry on time series.
        for i_t, t in enumerate(np.arange(t0, tmax)):

            # Step with social distance triggered
            # noinspection PyArgumentList
            basic_step_function(**basic_step_kwargs)

            # Registers new statecounts
            for state in self.statelist:
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]
            # No need to store a; it is already initialized as 1.
            # a_array[i_0 + i_t] = sample_node_data[REAC_KEY]

            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies all current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max+1):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    # a_array[j_t] = a_array[i_0 + i_t]  # Comment this to return ones as remaining.
                break

        return statecount, t, a_array, i_t_resets

    def simulate_condit_global(self, tmax, init_mode=None, init_data=None,
                               basic_step_function=None, basic_step_kwargs=None,
                               reac_step_function=None, reac_step_kwargs=None,
                               short_term_state="I", long_term_state="R", monitor_states=("I", ),
                               reset_threshold=1.E-4, histher=None,
                               max_global_cycles=None, reset_rho0=True):
        """
        Simulates the global social distancing mechanism with activation/deactivation
        thresholds (i.e., mechanism is conditioned to the value of the prevalence) and
        memory reset at each activation.
        The activation threshold is lowered from the deactivation one by a factor given
        as histher.
        This is the fully global strategy, thus the resets also occur globally.

        Reaction Step function
        -------------
        The function to be used for model step with social distancing
        is specified as reac_step_function argument, and can either be a class method
        or a string (with the exact name of the class method). Arguments are
        specified in reac_step_kwargs, as a dictionary of keyword arguments.
        It must contain global_rho0 as an argument, which is the global offset.

        reset_threshold : triggers a memory reset whenever the monitored states
          go under this value (from above). Can either be int or float. If int,
          it is interpreted as a number of cases. If float, as a fraction of the
          total population.

        max_global_cycles : defines the number of resets allowed (plus 1).
            If not informed, no limit is defined, and resets will be done
            until another condition stops the simulation.
            If informed and run_last_cycle = True, simulation is continued
            with no social distancing.
            If informed and run_last_cycle = False, simulation is continued
            with social distancing, but no more resets are performed.
        """
        # ---------------
        # Initialization
        if histher is None:
            # Better than using 0.1 directly as default in this case
            histher = 0.1

        # Population initialization
        self.g.make_travel_arrays()
        if init_mode is not None and init_data is not None:
            self.initialize_states(init_mode, init_data)

        # Initialization of data collectors
        i_max = int(tmax / 1.)  # A future remainder for non-integer time steps dt != 1
        statecount = dict()
        for state in self.statelist:
            statecount[state] = np.zeros((i_max + 1, len(self.g)), dtype=int)

        current_statecount = {state: [self.g.num(ni, state) for ni in self.g.nodes()]
                              for state in self.statelist}
        a_array = np.ones(i_max + 1, dtype=float)
        sample_node = list(self.g)[0]
        sample_node_data = self.g.nodes[sample_node]

        # "Reset-strategy" variables
        global_rho0 = 0.  # Initial prevalence offset
        self.g.graph[GLOBAL_RHO0_KEY] = global_rho0
        num_cycles = 0  # Counter of cycles
        glob_num_inf = self.g.total_num_in_statelist(monitor_states)
        # glob_num_next = glob_num_inf
        i_t_resets = []  # List of time indexes at which reset occurs
        i_t_deacts = []  # List of time indexes at which deactivations occurs

        # Activation(reset) and deactivation thresholds
        if isinstance(reset_threshold, (np.floating, float)):
            reset_threshold = round(self.g.total_pop_size() * reset_threshold)
        deact_threshold = round((1. - histher) * reset_threshold)

        if max_global_cycles is None:
            # No maximum
            max_global_cycles = np.inf

        # Definition of the simulation step function
        # Basic step
        if basic_step_kwargs is None:
            basic_step_kwargs = {}

        if basic_step_function is None:
            basic_step_function = self.step_basic
        elif type(basic_step_function) is str:
            basic_step_function = self.__getattribute__(basic_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(basic_step_function))

        # Reaction step function
        if reac_step_kwargs is None:
            reac_step_kwargs = {}

        if reac_step_function is None:
            reac_step_function = self.step_socialdist
        elif type(reac_step_function) is str:
            reac_step_function = self.__getattribute__(reac_step_function)
        else:
            raise ValueError("Hey, step_function '{}' not understood."
                             "".format(reac_step_function))

        # Registers initial state to the counters
        for state in self.statelist:
            statecount[state][0] = [self.g.num(ni, state) for ni in self.g.nodes()]
        # a_array[0] = 1.  # For simplicity, first is regarded as one.

        # Decides the initial step algorithm, based on initial prevalence and activation thresh.
        if glob_num_inf < reset_threshold:
            step_function = basic_step_function
            step_kwargs = basic_step_kwargs
            # Possibly set rho_0 as well
        else:
            step_function = reac_step_function
            step_kwargs = reac_step_kwargs

        # ---------------------------------
        # Simulation execution: loop of outbreak cycles
        t = 0.  # Shuts down PyCharm warning.
        # i_t = 0  # Shuts down PyCharm warning.
        i_0 = 1  # First vacant entry on time series.

        for i_t, t in enumerate(np.arange(tmax)):  # This allows for non-integer t

            # Social distancing step function
            step_function(**step_kwargs)

            # -----------------------------
            # Registers new statecounts
            for state in self.statelist:
                current_statecount[state] = [self.g.num(ni, state) for ni in self.g.nodes()]
                statecount[state][i_0 + i_t] = current_statecount[state]
            # Registers the global a(I, R) using the sample node
            a_array[i_0 + i_t] = sample_node_data[REAC_KEY]

            # ------------------------------
            # Absorbing stop condition achievement
            if self.stop_condition_basic(current_statecount):
                # Absorbing state achieved. Copies current to all remaining time stamps.
                for j_t in range(i_0 + i_t + 1, i_max+1):
                    for state in self.statelist:
                        statecount[state][j_t] = current_statecount[state]
                    # a_array[j_t] = a_array[i_0 + i_t]  # Comment this to return ones as remaining.
                break  # - No next stage
                # return statecount, t, a_array, i_t_resets

            # ---------------------------------
            # Event handling
            glob_num_next = self.g.total_num_in_statelist(monitor_states)

            # Reset/activation threshold (down/up cross)
            if glob_num_next > reset_threshold >= glob_num_inf and num_cycles < max_global_cycles and \
                    step_function is basic_step_function:
                # Update count of cycles, the prevalence offset, annotate i_t
                num_cycles += 1
                # Future: move the rho0 calculation to a function.
                if reset_rho0:
                    global_rho0 = self.g.total_num(short_term_state)
                    global_rho0 += reac_step_kwargs["long_term_coef"] * self.g.total_num(long_term_state)
                    global_rho0 /= self.g.total_pop_size()
                    self.g.graph[GLOBAL_RHO0_KEY] = global_rho0
                step_kwargs = reac_step_kwargs
                step_function = reac_step_function
                i_t_resets.append(i_t)

                # Checks max number of cycles
                if num_cycles == max_global_cycles:
                    # Resets the step function to basic
                    step_kwargs = basic_step_kwargs
                    step_function = basic_step_function

            # Deactivation threshold (up/down cross)
            if glob_num_next < deact_threshold <= glob_num_inf:
                step_kwargs = basic_step_kwargs
                step_function = basic_step_function
                i_t_deacts.append(i_t)

            # Updates the current global number of infecteds (monit. states)
            glob_num_inf = glob_num_next

        return statecount, t, a_array, i_t_resets, i_t_deacts


def calc_p_infection(eff_beta, num_infective, pop_size):
    return 1. - (1. - eff_beta/pop_size)**num_infective


class MetapopSIR(MetapopModel):

    statelist = list("SIR")
    transitions = [("S", "I"), ("I", "R")]

    def __init__(self, g, beta, mu):
        super().__init__(g)
        self.beta = beta
        self.mu = mu

    # ---------------------------------------
    # Probability of infection: calc functions

    # As a legacy, these are the hardcoded functions. May delete when feeling cute.
    def calc_pinf(self, ni):
        return calc_p_infection(self.beta, self.g.num(ni, "I"),
                                self.g.pop_size(ni))

    def calc_pinf_react(self, ni, k, frac):
        """Reduces the individual probability by (1 - frac)**k.
        Used to the social distancing model by Eskyn et. al. 2019.
        """
        beta_eff = self.beta * (1. - frac) ** k
        return calc_p_infection(beta_eff, self.g.num(ni, "I"),
                                self.g.pop_size(ni))

    def calc_pinf_short_term_reg(self, ni, k):
        """Short-term reaction, regional strategy.
        Proportional to the number of infecteds."""
        frac = self.g.num(ni, "I") / self.g.pop_size(ni)
        return self.calc_pinf_react(ni, k, frac)

    def calc_pinf_long_term_reg(self, ni, k):
        """Long-term reaction, regional strategy."""
        frac = self.g.num_in_statelist(ni, "IR") / self.g.pop_size(ni)
        return self.calc_pinf_react(ni, k, frac)

    # --------------------------------
    # Stop conditions
    def stop_condition_basic(self, current_statecount):
        # Checks if all nodes have no infected
        return sum(current_statecount["I"]) == 0

    # -------------------------------
    # Iteration functions

    def epidemic_step_basic(self):
        # Generates the infection probability function
        calc_pinf = self.gen_calc_pinf_basic(self.beta)

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # # Immediate change
            # self.g.change_state(ni, "S", "I", num_inf)
            # self.g.change_state(ni, "I", "R", num_heal)

            # Flag to change later
            self.g.set_tochange(ni, "S", "I", num_inf)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_uniform(self, a):
        # Generates the infection probability function
        calc_pinf = self.gen_calc_pinf_uniform(self.beta, a)

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "I", num_inf)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_socialdist(self, reac_exponent=1, long_term_coef=1,
                                 globality_coef=1, local_rho0_key=None):
        """
        Epidemic network iteration applying the social distancing mechanism,
        as explained in self.gen_calc_pinf_socialdist.
        """

        # Generates the infection probability function
        calc_pinf = self.gen_calc_pinf_socialdist(self.beta,
                                                  reac_exponent=reac_exponent,
                                                  long_term_coef=long_term_coef,
                                                  globality_coef=globality_coef,
                                                  local_rho0_key=local_rho0_key,
                                                  )

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "I", num_inf)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_activation_local(self, act_thres,
                                       reac_exponent=1,
                                       long_term_coef=1, globality_coef=1,
                                       act_long_term_coef=1
                                       ):
        """Epidemic network iteration with social distancing activated after the
        local (regional) prevalence reaches a certain threshold.
        Activation is permanent.
        """

        # Generates the infection probability functions, with and without social distancing
        calc_pinf_basic = self.gen_calc_pinf_basic(self.beta)
        calc_pinf_socdist = self.gen_calc_pinf_socialdist(
            self.beta,
            reac_exponent=reac_exponent,
            long_term_coef=long_term_coef,
            globality_coef=globality_coef,
        )

        for ni in self.g.nodes():

            # Updates the activation status
            if not self.g.nodes[ni]["soc_dist_active"]:  # Only inactive nodes are tested
                rho_ef = self.g.num(ni, "I") + act_long_term_coef * self.g.num(ni, "R")
                rho_ef /= self.g.pop_size(ni)
                if rho_ef > act_thres:
                    # Activates the reaction
                    self.g.nodes[ni]["soc_dist_active"] = True

            # Probability of infection by activation status
            if self.g.nodes[ni]["soc_dist_active"]:
                p_inf = calc_pinf_socdist(ni)
            else:
                p_inf = calc_pinf_basic(ni)

            # Proceeds with the calculations of number of events
            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "I", num_inf)
            self.g.set_tochange(ni, "I", "R", num_heal)


class MetapopSEIR(MetapopModel):

    statelist = list("SEIR")
    transitions = [("S", "E"), ("E", "I"), ("I", "R")]

    def __init__(self, g, beta, nu, mu):
        """
        Parameters
        ----------
        g : Metapop
            Metapopulation.
        beta : float
            Infection probability (times number of interactions).
        nu : float
            Rate of transition from latent (E) to infectious (I).
        mu : float
            Healing/removal probability.

        """
        super().__init__(g)
        self.beta = beta
        self.nu = nu
        self.mu = mu

    # --------------------------------
    # Stop conditions
    def stop_condition_basic(self, current_statecount):
        # Checks if all nodes have no infected or latent.
        return sum(current_statecount["I"]) + sum(current_statecount["E"]) == 0

    # -------------------------------
    # Iteration functions

    def epidemic_step_basic(self):
        calc_pinf = self.gen_calc_pinf_basic(self.beta)

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_symptom = np.random.binomial(self.g.num(ni, "E"), self.nu)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # # Immediate change
            # self.g.change_state(ni, "S", "E", num_inf)
            # self.g.change_state(ni, "E", "I", num_symptom)
            # self.g.change_state(ni, "I", "R", num_heal)

            # Flag to change later
            self.g.set_tochange(ni, "S", "E", num_inf)
            self.g.set_tochange(ni, "E", "I", num_symptom)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_uniform(self, a):
        calc_pinf = self.gen_calc_pinf_uniform(self.beta, a)

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_symptom = np.random.binomial(self.g.num(ni, "E"), self.nu)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "E", num_inf)
            self.g.set_tochange(ni, "E", "I", num_symptom)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_socialdist(self, reac_exponent=1, long_term_coef=1,
                                 globality_coef=1, local_rho0_key=None):
        calc_pinf = self.gen_calc_pinf_socialdist(self.beta,
                                                  reac_exponent=reac_exponent,
                                                  long_term_coef=long_term_coef,
                                                  globality_coef=globality_coef,
                                                  local_rho0_key=local_rho0_key)

        for ni in self.g.nodes():

            # Probability of infection
            p_inf = calc_pinf(ni)

            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_symptom = np.random.binomial(self.g.num(ni, "E"), self.nu)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "E", num_inf)
            self.g.set_tochange(ni, "E", "I", num_symptom)
            self.g.set_tochange(ni, "I", "R", num_heal)

    def epidemic_step_activation_local(self, act_thres,
                                       reac_exponent=1,
                                       long_term_coef=1, globality_coef=1,
                                       act_long_term_coef=1
                                       ):
        """Epidemic network iteration with social distancing activated after the
        local (regional) prevalence reaches a certain threshold.
        Activation is permanent.
        """

        # Generates the infection probability functions, with and without social distancing
        calc_pinf_basic = self.gen_calc_pinf_basic(self.beta)
        calc_pinf_socdist = self.gen_calc_pinf_socialdist(
            self.beta,
            reac_exponent=reac_exponent,
            long_term_coef=long_term_coef,
            globality_coef=globality_coef,
        )

        for ni in self.g.nodes():

            # Updates the activation status
            if not self.g.nodes[ni]["soc_dist_active"]:  # Only inactive nodes are tested
                rho_ef = self.g.num(ni, "I") + act_long_term_coef * self.g.num(ni, "R")
                rho_ef /= self.g.pop_size(ni)
                if rho_ef > act_thres:
                    # Activates the reaction
                    self.g.nodes[ni]["soc_dist_active"] = True

            # Probability of infection by activation status
            if self.g.nodes[ni]["soc_dist_active"]:
                p_inf = calc_pinf_socdist(ni)
            else:
                p_inf = calc_pinf_basic(ni)

            # Proceeds with the calculations of number of events
            num_inf = np.random.binomial(self.g.num(ni, "S"), p_inf)
            num_symptom = np.random.binomial(self.g.num(ni, "E"), self.nu)
            num_heal = np.random.binomial(self.g.num(ni, "I"), self.mu)

            # Flag to change later
            self.g.set_tochange(ni, "S", "E", num_inf)
            self.g.set_tochange(ni, "E", "I", num_symptom)
            self.g.set_tochange(ni, "I", "R", num_heal)


def initialize_infective_states(self, mode, data, infective_state="I",
                                healthy_state="S"):
    """ Initializes the numbers of each node/subpopulation based on an
    infective and a healthy state.

    Modes
    -----
    infected_count_dict, infected_dict, infec_dic
        A dictionary, keyed by subpopulations, with the number of initially
        infected individuals. Does not need to contain all nodes, missing nodes
        will be set with no infecteds.

    Parameters
    ----------
    self : Metapop
    mode : str
        Mode of initialization. See "Modes".
    data : any
        Data used to initialize, according to mode. See "Modes"
    infective_state : hashable
    healthy_state : hashable

    """

    if mode in ["infected_count_dict", "infected_dict", "infec_dict"]:
        if type(data) is str:
            data = str_to_dict(data)

        # First makes whole population as susceptible
        self.set_whole_population_to(healthy_state)

        # For each entry in dict, sets the number of individuals to infected.
        for ni, num in data.items():
            self.change_state(ni, healthy_state, infective_state, num)

    else:
        raise ValueError("Hey, mode '{}' for initializing the "
                         "network states was not recognized. "
                         "Please check 'initialize_infective_states'"
                         "documentation.".format(mode))
