import networkx as nx
import numpy as np
import warnings
import gc

E_CONST = np.e ** (-np.euler_gamma)


# [2020-07-21] I'll try to make this module quickly convertible between networkx 1 and 2.
# Just look for the tag [NXVERSION] and do the changes.

def simple_unweighted_distance(g, source, return_as_dicts=True):
    """Returns the unweighted shortest path length between nodes and source."""
    dist_dict = nx.shortest_path_length(g, source)

    if return_as_dicts:
        return dist_dict
    else:
        return np.fromiter((dist_dict[ni] for ni in g), dtype=int)


# ---------------------------------
# SIR RANDOM WALK HITTING TIME
# ----------------
# Distances from Ianelli et. al. 2017.
# Effective distances for epidemics spreading on complex networks.
# It considers not only the shortest weighted path between nodes,
# but (as a computationally feasible version) instead the combination
# of random walks through the network.

def calc_transition_matrix_and_total_flow(g, weight="weight"):
    """

    Parameters
    ----------
    g : nx.Graph
    weight : str
    """
    g_size = len(g)

    # Initializes the transition matrix
    p = np.zeros((g_size, g_size),  dtype=float)
    flow = 0.

    # Main loop
    for i, ni in enumerate(g):
        flow_i = 0.

        # Neighbor indexes (not ids) and weights
        i_neighbors = [(j, g[ni][nj][weight]) for j, nj in enumerate(g)
                       if nj in g[ni]]
        for j, w in i_neighbors:
            # w = data[weight]
            p[i][j] = w
            flow_i += w

        # Final normalization of row i
        if flow_i != 0.:  # Float comparison (weight could be very small)
            p[i] /= flow_i
        else:
            # The whole i row is zero
            raise ValueError("Hey, warning: node {} has a net outflux equal to zero."
                             "".format(ni))

        # Total flow
        flow += flow_i

    return p, flow


def calc_rwsir_distances_matrix(p, e_delta):
    """
    Calculates the SIR hitting time with random walk paths, from the paper:
    Effective distances for epidemics spreading on complex networks - Ianelli, 2017.

    Using the transition matrix p, a row-stochastic matrix of the flows from i
    to j. e_delta is a parameter that, for SIR, is read as:

    e_delta = e ** delta = (beta - mu)/alpha * e ** (-gamma_e)

    Where alpha is the overall travel rate (or probability of travel per day),
    gamma_e is the Euler-Mascheroni constant and beta/mu are the infection/heal
    rates.

    Parameters
    ----------
    p : np.ndarray
        Transition matrix. Square numpy array.
    e_delta : float
        e ** delta, where delta = ln((beta - mu)/alpha) - gamma_e.
        It is the coefficient that multiplies the identity matrix.
    """
    # Condition
    if e_delta <= 1:
        warnings.warn("Hey, the effective distance calc function received "
                      "e_delta <= 1, meaning that delta < 0. This could mean "
                      "that the overall travel flows are too high for the "
                      "calculation of the SIR RW hitting time. Try reducing "
                      "the flows to get a result that is proportionally right.")

    p_size = p.shape[0]  # Assumes square matrix
    d_rw = np.zeros((p_size, p_size), dtype=float)

    # Initialize reduced objects
    a_reduc = np.empty((p_size-1, p_size-1), dtype=float)
    b_reduc = np.empty(p_size-1, dtype=float)
    x_reduc = np.empty(p_size-1, dtype=float)

    # Copy of p, negative and with e_delta added to diagonals
    a_full = -p.copy()
    for i in range(p_size):
        a_full[i][i] += e_delta

    for j in range(p_size):
        # Construct j-removed matrix and column vector
        a_reduc[0:j, 0:j] = a_full[0:j, 0:j]
        a_reduc[0:j, j:] = a_full[0:j, j+1:]
        a_reduc[j:, 0:j] = a_full[j+1:, 0:j]
        a_reduc[j:, j:] = a_full[j+1:, j+1:]

        b_reduc[:j] = -a_full[:j, j]  # Minus to compensate the signal of a_full
        b_reduc[j:] = -a_full[j+1:, j]

        # Solve the linear system equivalent to inverse matrix.
        # Exact method from numpy
        x_reduc = np.linalg.solve(a_reduc, b_reduc)

        # Take -logarithm of the resulting vector
        # Store vector into final matrix, avoiding the diagonal
        d_rw[0:j, j] = -np.log(x_reduc[0:j])
        d_rw[j+1:, j] = -np.log(x_reduc[j:])

    return d_rw


def sir_hitting_time(g, beta_minus_mu, num_indiv=None, pop_size_key="size",
                     weight="weight", travel_fac=1., return_as_dicts=True):
    """
    Calculates the SIR random walk hitting time for a weighted graph, in which
    weights are interpreted as flow rates between nodes. Formula from Ianelli, 2017:
    "Effective distances for epidemics spreading on complex networks".
    This method considers not only the shortest weighted path between i and j, but
    a set of random walks (represented as a Markov chain) as a computationally feasible
    approximation of "all possible" paths.

    The hiiting time d_ij / (beta - mu) for an SIR disease seeded at node i to reach node j is given
    by:

    d_ij = -ln{ [e^delta * I - P(jj)]^(-1) p(j) }_i

    Where P(jj) is the transition matrix (stochastic) with row and column j removed,
    p(j) is the column j of P with entry j removed, I is the identity of size N-1 and
    delta = ln((beta - mu)/alpha) - gamma_e, the latter being the Euler-Mascheroni constant,
    and alpha is the average travel probability.

    Parameters
    ----------
    g : nx.Graph, nx.DiGraph
        The network.
    beta_minus_mu : float
        beta - mu, the difference between transmission and healing probabilities.
    num_indiv : int
        Total population. If not informed, it is extracted from node attributes given by...
    pop_size_key : hashable
        Node attribute name of the population size, used if num_indiv is None.
    weight : hashable
        Edge attribute name of the weights. Default is "weight".
    travel_fac : float
        Global factor by which the travel weights are multiplied before the calculations.
        Notice that the method may fail if the travel flows are too high, so you can use
        travel_fac < 1 to reduce these flows so the method works.
    return_as_dicts : bool
        If True (default), result is returned as a dict of dicts. The first key for the source
        of infection, the second for the targets.
        If False, returns as a square numpy 2D array, ordered by g.nodes(), with first index (rows)
        as source nodes and second index (columns) as targets.
    """

    if num_indiv is None:
        # Inverse of the number of individuals
        num_indiv = sum(data[pop_size_key] for ni, data in g.nodes(data=True))

    # Transition matrix (row-normalized traffic flows)
    p, phi = calc_transition_matrix_and_total_flow(g, weight=weight)

    # Exponential of delta parameter
    alpha = travel_fac * phi / num_indiv  # Average individual travel probab.
    e_delta = beta_minus_mu / alpha * E_CONST

    if return_as_dicts:
        # Constructs a dict of dictionaries, all keyed by node ids.
        a = calc_rwsir_distances_matrix(p, e_delta) / beta_minus_mu
        a_dict = {ni: {nj: a[i][j] for j, nj in enumerate(g)}
                  for i, ni in enumerate(g)}
        return a_dict
    else:
        return calc_rwsir_distances_matrix(p, e_delta) / beta_minus_mu


# ----------------------------------------
# EFFECTIVE DISTANCE minimum path
# ---------------------
# From Gatreau, Barrat, Barthelmy - 2008
# Effective distances for epidemics spreading on complex networks
# Probably one of the first works. Considers the SI model, but beta-mu can be
# a simple change to SIR.
# It defines a new weight and considers the shortest path using this weight.


def calc_eff_weights(g, beta_minus_mu, weight="weight", pop_size_key="size",
                     eff_weight="eff_weight", travel_fac=1.):
    """Calculates the effective weight for the SI model hitting time.
    Defined in Gatreau et al 2008 -  Effective distances for epidemics spreading on complex networks.

    This is a directed metric, so if the graph is undirected, it returns
    another (directed) instance of it. Otherwise, the same g instance is
    returned, with the new weights calculated.

    Parameters
    ----------
    g : nx.Graph, nx.DiGraph
    beta_minus_mu : beta - mu, difference between infection and healing rates.
    weight : str
    pop_size_key : str
    eff_weight : str
        Edge attribute name to store the SI weights
    travel_fac : float
    """

    if isinstance(g, nx.Graph):
        h = nx.DiGraph(g)
        # Copies the population sizes
        for ni in g:
            # [NXVERSION]
            h.nodes[ni][pop_size_key] = g.nodes[ni][pop_size_key]  # nx2
            # h.node[ni][pop_size_key] = g.node[ni][pop_size_key]  # nx1
    elif isinstance(g, nx.DiGraph):
        h = g
    else:
        # Not sure if Multigraphs are actually incompatible, might review this.
        raise TypeError("Hey, this function only supports nx.Graph and "
                        "nx.DiGraph to calculate weights.")

    # Stores the new weight in place
    for u, v, data in h.edges(data=True):
        w = data[weight] * travel_fac
        # Source population
        # [NXVERSION]
        n_u = h.nodes[u][pop_size_key]  # nx2
        # n_u = h.node[u][pop_size_key]  # nx2
        weff = 1. / beta_minus_mu * (np.log(n_u * beta_minus_mu / w) - E_CONST)  # Already normalized by beta
        h[u][v][eff_weight] = weff
        if weff < 0:
            warnings.warn("Hey, negative weight produced between {} and {} (and"
                          "possibly bewteen others). Consider reducing travel fac."
                          "".format(u, v))

    return h


def gatreau_effective_distance(g, source, beta_minus_mu, pop_size_key="size", weight="weight",
                               travel_fac=1., return_as_dicts=True, eff_weight="eff_weight"):
    """Calculates the single source effective distances (arrival times)
    using the effective weights proposed by Gatreau et. al. 2018.
    "Effective distances for epidemics spreading on complex networks."

    It takes the shortest weighted path using the effective weight.

    Notice that the effective weights are directed. Therefore, if the input graph
    is undirected, the algorithm creates a temporary directed copy, which is
    used to store the weights and compute the distances.

    Parameters
    ----------
    g : nx.Graph, nx.DiGraph
        Networkx graph instance, directed or undirected.
    source : any hashable
        Id of the source node.
    beta_minus_mu : float
        Transmission minus healing rate. This accounts for SIR instead of SI.
    pop_size_key : str
        Node attribute name of the population size.
    weight : str
        Edge attribute name of the basic weights.
    eff_weight : str
        Edge attribute name to store the effective weights (in the directed
        network instance).
    travel_fac : float
        Factor to multiply the basic weights.
    return_as_dicts : bool
        If True, returns the result as a dictionary keyed by node and valued by
        the arrival times from source. If False, returns an array with the same
        ordering as g.nodes().
    """

    # Calculates effective weights.
    h = calc_eff_weights(g, beta_minus_mu, pop_size_key=pop_size_key, weight=weight,
                         travel_fac=travel_fac, eff_weight=eff_weight)

    # Finds shortest paths using effective weight.
    d = nx.single_source_dijkstra_path_length(h, source, weight=eff_weight)

    # Collects garbage due to possible creation of temporary graph copy.
    # Might save memory in ipython.
    gc.collect()

    if return_as_dicts:
        return d  # Normalization by beta already in the weights.
    else:
        return np.fromiter((d[ni] for ni in g), dtype=float)


def main():
    pass


if __name__ == "__main__":
    main()
