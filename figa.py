import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm

from random import seed
from tqdm import tqdm, trange

from muffliato import simulation, acceleratedsimulation
from graphutils import gossip_matrix, compute_max_n_degree, T_mix, degree_max, vplambda
from ERsampled import eps_random

# For passing automatic checks
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 24})

# various constants
seed(0)
np.random.seed(0)
eps_local = 1
alpha = 2
sigma = np.sqrt(alpha / 2 * eps_local)


def summarize_vectors(list_vect):
    """Compute a summary of several trials by taking overall min, max, and the average
    Parameters
    ----------
    list_vec: list of numpy array
        List of the different trials statistics
    Returns
    -------
    summary: array, shape [max_length, 3]
        array with (mean, min, max) as function of the distance
    """
    print(list_vect, "listvect")
    max_length = 6
    print("maxi", max_length)
    complete = np.empty((len(list_vect), max_length, 3))
    summary= np.zeros((max_length, 3))
    complete[:] = np.NaN
    for i, l in enumerate(list_vect):
        complete[i][:len(l)] = l
    for i in range(max_length):
        summary[i][0] = np.nanmean(complete[:, i, 0])
        summary[i][1] = np.nanmin(complete[:, i, 1])
        summary[i][2] = np.nanmax(complete[:, i, 2])
    return summary


def to_interval(vector):
    """Utility function to convert (mean, min, max) into (mean, length of lower error, length of upper errors ) 
    """
    vector[:, 1] = vector[:, 0] - vector[:, 1]
    vector[:, 2] = vector[:, 2] - vector[:, 0]
    return vector

# Compute the privacy loss as function of the shortest path
def vector_loss(graph, u=0):
    """
    Compute an array of abscisse the distance to node u and with first coordinate mean privacy loss, the minimum privacy loss and maximum one
    Parameters
    ----------
    graph: networkx graph
    u: int
        the node from which distance are computed
    Returns
    -------
    stats: numpy array, shape (max dist, 3)
    """

    n = graph.number_of_nodes()

    print("Preprocessing of the graph")
    graph = gossip_matrix(graph)
    T = T_mix(graph, sigma)
    print("we need ",T, " iterations")
    print("Simulation of Muffliato")
    eps_inst, error, precision = acceleratedsimulation(graph, T, n, sigma=sigma, debug=False, approx=False, u=u)  
    print("Computing the privacy loss")
    distance = nx.shortest_path_length(graph, source=u)
    max_dist = max(distance.values())
    privacy_losses = [ [] for i in range(max_dist+1)]
    stats = np.zeros((max_dist, 3))
    eps_node = np.clip(eps_inst.sum(axis=0), 0, eps_local)
    for i in range(n):
        privacy_losses[distance[i]].append(eps_node[i])
    for i in range(max_dist):
        stats[i] = np.mean(privacy_losses[i]),  np.min(privacy_losses[i]), np.max(privacy_losses[i]) 
    return stats


fig, ax = plt.subplots(figsize=(8,10))

right_side = ax.spines["right"]
right_side.set_visible(False)
up_side = ax.spines["top"]
up_side.set_visible(False)


ax.set_yscale('log')
ax.set_xlim([0, 25.5])
ax.set_ylim([1e-5, 1.2])
ax.axhline(y=1, label="LDP loss", color="xkcd:black", lw=2)

print("For the hypercube")
# For exponential graph
hypercube = nx.hypercube_graph(11)
hypercube = nx.convert_node_labels_to_integers(hypercube)
loss_exp = vector_loss(hypercube)
loss_exp = to_interval(loss_exp)
ax.scatter([i for i in range(len(loss_exp))], loss_exp[:,0], label="Exponential", marker='^', color="xkcd:coral", s=100)

print("For the Erdos Renyi graph")
# For ER
n = 2048
trials = 5
connex = False
while not connex:	
	binomial = nx.gnp_random_graph(n, 1*np.log(n)/n)
	connex = nx.is_connected(binomial)
all_loss_ER = []
for trial in range(trials):
    loss_er = vector_loss(binomial)
    all_loss_ER.append(loss_er)
loss_er = summarize_vectors(all_loss_ER)
loss_er = to_interval(loss_er)
ax.errorbar([i for i in range(len(loss_er))], loss_er[:,0], yerr= loss_er[:, 1:].T, label="Erdos Renyi", color="xkcd:darkish purple", capthick=1, capsize = 4, lw=2 )


print("For geometric graph")
# For geometric
n = 2048
pos = {i: (np.random.random(), np.random.random()) for i in range(n)}
geometric = nx.random_geometric_graph(n, 0.07, pos=pos)
loss_geo = vector_loss(geometric)
loss_geo = to_interval(loss_geo)

plt.errorbar([i for i in range(len(loss_geo))], loss_geo[:,0], label="Random Geometric", yerr=loss_geo[:, 1:].T, color='xkcd:amber', capthick=1, capsize = 4, lw=2)

print("For the grid")
# For grid
grid = nx.grid_2d_graph(45, 45)
grid = nx.convert_node_labels_to_integers(grid)
u = 1035
loss_grid = vector_loss(grid)
loss_grid = to_interval(loss_grid)

ax.errorbar([i for i in range(len(loss_grid))], loss_grid[:,0], label="Grid", yerr= loss_grid[:, 1:].T, color='xkcd:cherry', capthick=1, capsize = 4, lw=2 )

ax.set_xlabel("Shortest Path Length")
ax.set_ylabel("Privacy Loss")
ax.legend(loc='lower left')
fig.savefig("fig1a.pdf",  bbox_inches='tight', pad_inches=0)
plt.show()


