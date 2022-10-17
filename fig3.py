import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm


from muffliato import acceleratedsimulation
from graphutils import gossip_matrix, T_mix

# For passing automatic checks
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 24})
cmap = plt.cm.coolwarm_r

eps_local = 1
alpha = 2
sigma = np.sqrt(alpha / 2 * eps_local)

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
mypower=13

plt.figure(figsize=(9,9))
for d in range(4, mypower):
    hypercube = nx.hypercube_graph(d)
    hypercube = nx.convert_node_labels_to_integers(hypercube)
    loss_exp = vector_loss(hypercube)
    plt.scatter([i for i in range(len(loss_exp))], loss_exp[:,0], label=str(d), marker='^', color=cmap(d/(mypower+1)), s=100)
plt.legend()
plt.xlabel('Shortest Path Length')
plt.ylabel('Privacy Loss')
plt.savefig('nevo.pdf', bbox_inches='tight', pad_inches=0)
plt.show()