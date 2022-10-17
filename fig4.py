import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm

from random import seed

from muffliato import acceleratedsimulation
from graphutils import gossip_matrix, T_mix

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 24})



seed(1)
np.random.seed(1)
eps_local = 1
alpha = 2
sigma = np.sqrt(alpha / 2 * eps_local)


n = 2000
u = 0

cmap = plt.cm.Spectral_r

pos = {i: (np.random.random(), np.random.random()) for i in range(n)}
geometric = nx.random_geometric_graph(n, 0.08, pos=pos)


my_graph = geometric


print("Preprocessing of the graph")
my_graph = gossip_matrix(my_graph)


T = T_mix(my_graph, sigma)


print("Simulation of the gossip")

eps_inst1, error1, approx1, precision1 = acceleratedsimulation(my_graph, T, n, sigma=sigma, debug=False, approx=True, u=u)



print("plotting the privacy loss as a function of euclidean distance")
fig_sim, ax_sim = plt.subplots()
distance =np.array([ np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) for v in my_graph.nodes()])
print(distance.shape, "distance shape", eps_inst1[-1].shape)
ax_sim.scatter(distance, eps_inst1.sum(axis=0), alpha=.3, color="xkcd:blue")
ax_sim.set_yscale("log")
ax_sim.set_xlabel("Euclidean distance")
ax_sim.set_ylabel("Privacy Loss")
plt.savefig("geoeuclideandistance.pdf")

plt.show()