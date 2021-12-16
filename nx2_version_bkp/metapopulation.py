import networkx as nx
import numpy as np


# [2020-07-15] nx1 downgrade
#
# * set_attributes() functions   -   not found
# * Iteration over .nodes() => nodes_iter()    - TODO
# * Acessing .nodes() attrs   =>  .node[]      - TODO
# * Iteration over .neighbors  => .neighbors_iter()  - TODO


class Metapop(nx.Graph):

    statelist = []

    def __init__(self, incoming_graph_data=None, pop_sizes=None, statelist=None,
                 **attr):
        """Creates an instance of Metapop object.

        Parameters
        ----------
        incoming_graph_data : input graph. (optional, default: None)
            Data to initialize graph. If None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a NumPy matrix
            or 2d ndarray, a SciPy sparse matrix, or a PyGraphviz graph.

        pop_sizes : population size of each node. Can be:
            * int : all nodes initialized with the same population.
            * list: each entry is the population of each node in self.nodes()
            * dict: {node: population}. In this case, not all nodes are required.
            * None: nothing is initialized

        declare_states : any iterable
            States to have their counts declared (initialized as zero),
            avoiding future KeyErrors.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        """
        super().__init__(incoming_graph_data, **attr)

        # Population sizes data
        if pop_sizes is None:
            pass
        elif type(pop_sizes) is int:
            for ni in self.nodes():
                self.set_pop_size(ni, pop_sizes)
        elif type(pop_sizes) in [list, np.ndarray, tuple]:
            if len(self) != len(pop_sizes):
                raise ValueError("Hey, the size of pop_sizes in this case "
                                 "must match the network size.")
            self.set_pop_size_nodelist(self.nodes(), pop_sizes)
        elif type(pop_sizes) is dict:
            self.set_pop_size_nodedict(pop_sizes)

        if statelist is not None:
            self.statelist = statelist

    # -------------------------------
    # Basic attributes: population sizes

    def pop_size(self, ni):
        """Returns the population of node ni."""
        return self.nodes[ni]["size"]

    def set_pop_size(self, ni, pop_size):
        self.nodes[ni]["size"] = pop_size

    def set_pop_size_nodelist(self, nodelist, sizelist):
        for ni, pop_size in zip(nodelist, sizelist):
            self.set_pop_size(ni, pop_size)

    def set_pop_size_nodedict(self, sizedict):
        """Sets population of nodes using a dict, keyed by nodes and
        valued by population sizes.
        {node: pop_size}
        """
        for ni, pop_size in sizedict.items():
            self.set_pop_size(ni, pop_size)

    def total_pop_size(self):
        return sum(self.pop_size(ni) for ni in self.nodes())

    # -----------------------------------
    # Counts of individuals in each state
    # Protected functions may create state count inconsistencies.
    def _set_num(self, ni, state, num):
        """Sets the number of individuals in node ni at state to num."""
        self.nodes[ni]["num_"+state] = num

    def num(self, ni, state):
        return self.nodes[ni]["num_"+state]

    def num_in_statelist(self, ni, statelist):
        return sum(self.num(ni, state) for state in statelist)

    def total_num(self, state):
        return sum(self.num(ni, state) for ni in self.nodes())

    def total_num_in_statelist(self, statelist):
        return sum(self.total_num(state) for state in statelist)

    def add_num(self, ni, state, num_add):
        """Adds a given number of individuals to state in node ni.
        This also changes the number of individuals on that node."""
        self.nodes[ni]["num_" + state] += num_add
        self.nodes[ni]["size"] += num_add

    def _add_num_nofix(self, ni, state, num_add):
        """Adds a given number of individuals to state in node ni.
        Does not adjust the population size of the node.
        """
        self.nodes[ni]["num_" + state] += num_add

    def set_all_individuals_to(self, ni, state):
        """Sets the state of all individuals in a single node."""
        for other_state in self.statelist:
            self._set_num(ni, other_state, 0)
        self._set_num(ni, state, self.pop_size(ni))

    def set_whole_population_to(self, state):
        """Sets the state of all individuals in all nodes."""
        for ni in self.nodes():
            self.set_all_individuals_to(ni, state)

    def change_state(self, ni, from_state, to_state, num):
        """Changes num individuals from_state to_state, in node ni."""
        self._add_num_nofix(ni, from_state, -num)
        self._add_num_nofix(ni, to_state, num)

    def move_individuals(self, from_node, to_node, state, num):
        """Moves num individuals from_node to_node, in state.
        This also changes the population size in each node.
        """
        self.add_num(from_node, state, -num)
        self.add_num(to_node, state, num)

    # --------------------
    # Flags handling: variables that store changes to be applied syncronously.

    def set_tochange(self, ni, from_state, to_state, num):
        """Sets the num of indiv. that will change from_state to_state, in node ni."""
        self.nodes[ni]["tochange_{}_{}".format(from_state, to_state)] = num

    def tochange(self, ni, from_state, to_state):
        return self.nodes[ni]["tochange_{}_{}".format(from_state, to_state)]

    def apply_state_changes(self, from_state, to_state):
        """Consolidates state changes marked as 'tochange' in each node."""
        for ni in self.nodes():
            self.change_state(ni, from_state, to_state,
                              self.tochange(ni, from_state, to_state))

    def set_tomove(self, from_ni, state, nums_list):
        """Sets the nums of indiv. that will move from_node to neighbors, in state.
        nums_list is a list in the same order as self.neighbors(from_ni).
        """
        self.nodes[from_ni]["tomove_{}".format(state)] = nums_list

    def tomove(self, from_ni, state):
        return self.nodes[from_ni]["tomove_{}".format(state)]

    def apply_moves(self, state):
        """Consolidates inidividual travels between all neighboring nodes,
        in state.
        """
        for ni in self.nodes():
            for nj, num in zip(self.neighbors(ni), self.tomove(ni, state)):
                self.move_individuals(ni, nj, state, num)

    def check_states_and_nums_consistency(self):
        """"""
        for ni in self.nodes():
            # Looks for negative values
            for state in self.statelist:
                if self.num(ni, state) < 0:
                    raise ValueError("Hey, node {} has negative count in"
                                     " state {}.".format(ni, state))

            # Looks for inconsistent sums of states and total population
            s = sum(self.num(ni, state) for state in self.statelist)
            if s != self.pop_size(ni):
                raise ValueError("Hey, node {} has inconsistent sum of "
                                 "state counts.\n"
                                 "State sum: {}\n"
                                 "Pop size:  {}".format(ni, s, self.pop_size(ni)))

    # ---------------------
    # Link weights
    def weight(self, ni, nj):
        return self.edges[ni, nj]["weight"]

    def set_weight(self, ni, nj, weight):
        self.edges[ni, nj]["weight"] = weight
        # self.edges[nj, ni]["weight"] = weight

    def make_travel_arrays(self, dtype=float):
        """From the weights stored at the edges, creates the travel
        arrays from each node. It is stored as a numpy array in the
        node attribute "travel_array".

        Possible directed nature of links is preserved.
        """
        for ni in self.nodes():
            trav = np.fromiter((self.weight(ni, nj) for nj in self.neighbors(ni)),
                               dtype=dtype)
            self.nodes[ni]["travel_array"] = trav

    def travel_array(self, ni):
        """"""
        return self.nodes[ni]["travel_array"]

    def calc_average_travel_probab(self, travel_fac=1.):
        """Returns the average individual travel probability in a unit time step,
        with weights multiplied by travel_fac.
        """
        self.make_travel_arrays(dtype=float)

        num_travel = 0.
        for ni in self.nodes():
            num_travel += self.travel_array(ni).sum()

        return travel_fac * num_travel / self.total_pop_size()


# --------------------------------------
# Population distributions
# --------------------------------------
def powerlaw_seq_with_totalsize(num_samples, total_size, gamma,
                                k_min=1, k_max=1000):
    """Generates a sequence of num_samples random numbers with a power-law
    distribution P(N) = A*N**(-gamma), such that the sum of the sequence
    is given by totalsize."""
    # Creates a grid of points
    k_sample = np.arange(k_min, k_max + 1, dtype=float)

    # Power-law probabilities for the sequence.
    p = k_sample ** (-float(gamma))
    p = p / p.sum()

    result = np.random.choice(k_sample, num_samples, p=p)
    result *= total_size / result.sum()

    return np.array(result, int)


def stationary_population_from_degree(g, total_size):
    """
    Defines the node populations according to their degrees, following the
    stationary distribution obtained by constant travel probability (Aleta 2017).
    Returns a dictionary keyed by node and valued by populations sizes.

    N_i = <N> * k / <k> = total_size * k / (2*num_edges)
    (where <N> = total_size / num_nodes).

    Parameters
    ----------
    g : nx.Graph
    total_size : int

    Returns
    -------
    pop_dict : dict
        {node: pop_size}
    """
    num_edges = len(g.edges())

    pop_dict = dict()
    for ni, k in nx.degree(g):
        pop_dict[ni] = int(total_size * k / 2 / num_edges)

    return pop_dict


# --------------------------------------
# Definition of weights (travel matrices)
# --------------------------------------
def def_weight_by_pop_size(g, coeff, make_arrays=True, weight_key="weight"):
    """ Defines weights of a metapopulation proportional to the
    products of the populations sizes.

    T_ij = coeff * N_i * N_j / N

    For a fully connected graph, coeff is interpreted as, approximately,
    the fraction of a node's population that travels in one day

    Parameters
    ----------
    g : Metapop
    coeff : float
        Travel coefficient. Avoid numbers close to 1, use smaller than that.
    make_arrays : bool
        Calculate node travel arrays at the end.
    weight_key : any hashable
        Edge attribute to set as travel amounts.
    """
    total_size = g.total_pop_size()
    for ni, nj in g.edges():
        g.edges[ni, nj][weight_key] = coeff * g.pop_size(ni) * g.pop_size(nj) / total_size

    if make_arrays:
        g.make_travel_arrays()


# def def_weight_by_degree()
