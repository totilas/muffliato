import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as colors
from random import seed

from muffliato import  acceleratedsimulation
from graphutils import gossip_matrix, T_mix

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

seed(0)
np.random.seed(1)
eps = 5
alpha = 1.5
sigma = np.sqrt(alpha/2*eps)

def process_ego(ego):
    name_edgelist = "facebook/"+str(ego)+".edges"
    my_graph = nx.read_edgelist(name_edgelist)
    my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
    Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
    G0 = my_graph.subgraph(Gcc[0]).copy()
    n = G0.number_of_nodes()
    G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")
    G0 = gossip_matrix(G0)
    for node in G0.nodes():
        if G0.degree[node] <= 3:
            u = node
            break
    T = T_mix(G0, sigma)
    eps_inst, _, _ = acceleratedsimulation(G0, T, n, sigma=sigma, debug=False, approx=False, u=u)
    G0.remove_edges_from(nx.selfloop_edges(G0))
    return G0, np.clip(eps_inst.sum(axis=0), 0, 1)


egos = [0, 107, 348, 414, 686, 698, 1684, 3437, 3980]

plt.figure(figsize=(10,10))
for i, ego in enumerate(egos):
    G0, colors = process_ego(ego)
    plt.subplot(3, 3, i+1)
    nx.draw(G0, node_color=colors, node_size=10, alpha=0.5, edge_color='xkcd:silver', width=.5, cmap=plt.cm.Spectral_r)
plt.savefig("allego.pdf", bbox_inches='tight', pad_inches=0)
plt.show()