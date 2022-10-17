import numpy as np 

from graphutils import gossip_matrix, contrib
from numpy.random import default_rng
import networkx as nx

import typer



def partial_renyi(n,p,q, rng, debug=False):
    G = nx.gnp_random_graph(n, p)
    if q!=0:
        my_mask = rng.binomial(1, q, size=n)
        for u in nx.nodes(G):
            if my_mask[u]:
                nodelist = list(nx.neighbors(G,u))
                for v in nodelist:
                    if u!=v:
                        G.remove_edge(u,v)
    G = gossip_matrix(G, debug=debug)
    return G



def versatilesimulation(n, p, q, rng, sigma=1, u=0, debug=False, alpha=2):
    """Run muffliato without Tchebychev acceleration
    Parameters
    ----------
    n: int
        number of nodes
    sigma: float
    u: int
        fixed node for the computation of privacy
    debug: boolean, default: False
        print the gossip matrix
    alpha: float
        parameter of RDP
    Returns
    -------
    eps_inst: array, shape(T, nb of nodes)
        the privacy loss per step for each node
    error: array, shape (T,)
        the convergence of x towards the mean
    proba_2: array, shape (T, number of nodes)
        privacy loss when approx is used
    precision: float
        magnitude of the difference between the true and the noisy version. going below this precision is not meaningful
    """
    contribution = np.zeros(n) # array to stock the contribution of a neighbor
    eps_inst = [] # privacy loss due to iteration t in node i
    x_exact = rng.random(size=n)
    x_inst = [np.clip(x_exact + rng.normal(size=n) * sigma, 0, 1)]
    #x_exact = np.array([n]+[0]*(n-1))
    #x_inst[0] = np.clip(x_exact + np.random.randn(n) * sigma, 0, 1)

    x_mean_exact = np.mean(x_exact)
    x_mean = np.mean(x_inst[0])
    precision = (x_mean - x_mean_exact)**2
    threshold = 3 * precision
    error = [np.linalg.norm(x_inst[-1]-x_mean)**2 /n]  

    Wt = np.eye(n)
    t = 0
    while True :
        t+=1
        # generate active nodes
        graph = partial_renyi(n,p,q, rng)
        W = nx.to_numpy_array(graph)
        for v in nx.nodes(graph):
            contribution[v] = alpha * contrib(Wt, u, v) / (2 * sigma**2) 
        eps_curr = np.zeros(n)
        for v in nx.nodes(graph):
            for w in nx.neighbors(graph, v):
                if w != v:
                    eps_curr[v]+=contribution[w]
        eps_inst.append(eps_curr)
        x_inst.append(W @ x_inst[-1])
        error.append(np.linalg.norm(x_inst[-1]-x_mean)**2 /n)
        if error[-1] < threshold:
            break
        Wt = W @ Wt
    return np.array(eps_inst), np.array(error), precision



app = typer.Typer()

def main(n_nodes: int = 1000,
    seed:int = 1,
    p: float = .002,
    prefix: str=""):

    rng = default_rng(seed=seed)

    print("No dropout")
    eps, error, precision = versatilesimulation(n_nodes, p, 0, rng)
    print("Few dropout")
    eps1, error1, precision1 = versatilesimulation(n_nodes, p, .1, rng)
    print("Mid dropout")
    eps2, error2, precision2 = versatilesimulation(n_nodes, p, .5, rng)
    print("High dropout")
    eps3, error3, precision3 = versatilesimulation(n_nodes, p, .9, rng)

    np.save("result/"+prefix+"errornodropout", error/precision)
    np.save("result/"+prefix+"errorfewdropout", error1/precision1)
    np.save("result/"+prefix+"errormiddropout", error2/precision2)
    np.save("result/"+prefix+"errorhighdropout", error3/precision3)

    np.save("result/"+prefix+"privacynodropout", np.array([ eps[:i].sum()/n_nodes for i in range(error.shape[0])]))
    np.save("result/"+prefix+"privacyfewdropout",np.array([ eps1[:i].sum()/n_nodes for i in range(error1.shape[0])]))
    np.save("result/"+prefix+"privacymiddropout",np.array([ eps2[:i].sum()/n_nodes for i in range(error2.shape[0])]))
    np.save("result/"+prefix+"privacyhighdropout", np.array([ eps3[:i].sum()/n_nodes for i in range(error3.shape[0])]))



if __name__ == "__main__":
    typer.run(main)
