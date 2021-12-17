from toolbox.network_gen_tools import *
from toolbox.file_tools import *
from sim_modules.metapopulation import *

EXPORT = True

num_nodes = 50
num_indiv = int(1E7)

file_name = "networks/rgn/n{:d}_r0p25_fromdegree_N1E7_32".format(num_nodes)
# file_name = "networks/path-graph/n{}_tr1_N1E6_01".format(num_nodes)
# file_name = "networks/sf-cm/n{}_gamma2p0_N1E7_05".format(num_nodes)
# file_name = "networks/er/n{}_k10_N1E7_05".format(num_nodes)
# file_name = "networks/ba/n{}_m5_N1E7_05".format(num_nodes)

avg_pop_size = num_indiv / num_nodes
print("Creating network...")

# # -----------------------
# # PATH GRAPH
# g = nx.path_graph(num_nodes)

# ------------------------
# Random Geometric Network
# Remember to activate "pos" as node data
g = nx.random_geometric_graph(num_nodes, 0.25)

# ------------------------
# Watts-Strogatz

# # ------------------------
# # Scale-free
# g = generate_layer("SF-CM", num_nodes, gamma=2.0, k_min=2)

# - - - - - - - - - - -

# # -------------------
# # Regular populations
# pop_sizes = num_nodes * [avg_pop_size]

# # -------------------
# # Uniformly random populations
# N_min = 1000
# N_max = 1E5
# raise NotImplementedError

# ---------------------
# Logarithmically uniform: reciprocal distribution


# # -----------------
# # ERDOS-RENYI
# p = 10 / num_nodes
# g = nx.erdos_renyi_graph(num_nodes, p=p)

# # --------------------
# # BARABASI-ALBERT
# m = 5
# g = nx.barabasi_albert_graph(num_nodes, m, seed=42)

# -------------------
# Population proportional to degree
print("Generating populations...")
pop_sizes = stationary_population_from_degree(g, num_indiv)

# - - - - - - - - - - - -
# -------------------
# Mobility proportional to population
print("Defining weights...")
mobility_coef = 1.
g = Metapop(g, pop_sizes)
def_weight_by_pop_size(g, mobility_coef, make_arrays=False)


# - - - - - - - - - - - - -
# -------------------

# # Population report
# print("Nodes:\n-----")
# for ni, data in g.nodes(data=True):
#     print(ni, data)
#
# print()
# print("Edges:\n-----")
# for ni, nj, data in g.edges(data=True):
#     print("({}, {})".format(ni, nj), data)

if not nx.is_connected(g):
    print("UNCONNEX NERWORK")

# - - - - - - - - - - - - -
# --------------------
# Population export
if EXPORT:
    print("Exporting to '{}'".format(file_name))
    make_folder(os.path.dirname(file_name))

    save_network_with_data(g, file_name+".csv", file_name+".edgl",
                           edge_attrs=["weight"], node_attrs=["size", "pos"]
                           )
