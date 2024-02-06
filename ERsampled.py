import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from graphutils import vplambda, T_mix, gossip_matrix
import random

def eps_random(graph, u, T, alpha, sigma):
    """Compute the privacy loss for a Erdos Renyi Graph drawn randomly
    Amplification for everyone except direct neighbors to u
    graph: networkx graph of communication
    u: int the vertex target
    T: int number of steps of the gossip
    alpha: float of Renyi DP
    sigma: float amount of the noise
    """
    n = graph.number_of_nodes()
    degree = np.array([d for (i,d) in nx.degree(graph)])
    eps_ndp = alpha *T / (sigma**2) * degree / (n - degree)
    eps_loc = alpha / (2*sigma**2)
    for v in nx.all_neighbors(graph, u):
        eps_ndp[v] = eps_loc
    return np.minimum(eps_ndp, eps_loc)

def eps_global(alpha, sigma):
    """Classical Gaussian formula for RDP"""
    return alpha / (2 *sigma**2)

if __name__ == "__main__":
    n = 500

    sigma = 1
    alpha = 1.5
    u = 0
    np.random.seed(1)
    random.seed(1)
    # Compute graph
    graph = nx.gnp_random_graph(n, 3*np.log(n)/n)
    graph = gossip_matrix(graph)
    T = T_mix(graph, sigma)
    eps_ndp = eps_random(graph, u, T, alpha, sigma)
    print(eps_ndp.min(), eps_ndp.mean(), eps_ndp.max())


    
