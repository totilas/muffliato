import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm
import matplotlib.colors as colors
from random import seed
from tqdm import tqdm, trange

from muffliato import simulation, acceleratedsimulation
from graphutils import gossip_matrix, compute_max_n_degree, T_mix

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

seed(1)
np.random.seed(1)
eps = 5
alpha = 1.5
sigma = np.sqrt(alpha/2*eps)

the_ego = 1912
name_file = str(the_ego)+".pdf"

name_edgelist = "facebook/"+str(the_ego)+".edges"
my_graph = nx.read_edgelist(name_edgelist)
my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
G0 = my_graph.subgraph(Gcc[0]).copy()
to_remove = []
for node in G0.nodes():
	if G0.degree[node] <= 0:
		to_remove.append(node)
for node in to_remove:
	G0.remove_node(node)

n = G0.number_of_nodes()
u = np.random.randint(n)
# we convert nodes to integer and keep the previous key of the node as an attribute fb_id
G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")
G0 = gossip_matrix(G0)
T = T_mix(G0, sigma)
eps_inst, error, approx, precision = simulation(G0, T, n, sigma=sigma, debug=False, approx=True, u=u)

fig, ax = plt.subplots()
G0.remove_edges_from(nx.selfloop_edges(G0))
nx.draw(G0, node_color=np.clip(eps_inst.sum(axis=0), 0, 1), node_size=30, alpha=0.5, edge_color='xkcd:silver', width=.2, cmap=plt.cm.Spectral_r)
# plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap = plt.cm.Spectral_r))
plt.savefig(name_file,  bbox_inches='tight', pad_inches=0)
plt.show()
