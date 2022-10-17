import data
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.linear_model._logistic import _intercept_dot
from scipy.special import expit
from sklearn.utils.validation import check_X_y
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from graphutils import T_mix, gossip_matrix
from ERsampled import eps_random, eps_global
from muffliato import gossip_vector
from tqdm import trange
import networkx as nx
from copy import deepcopy


def my_logistic_obj_and_grad(theta, X, y, lamb):
    """Computes the value and gradient of the objective function of logistic regression defined as:
    min (1/n) \sum_i log_loss(theta;X[i,:],y[i]) + (lamb / 2) \|w\|^2,
    where theta = w (if no intercept), or theta = [w b] (if intercept)

    Parameters
    ----------
    theta_init : array, shape (d,) or (d+1,)
        The initial value for the model parameters. When an intercept is used, it corresponds to the last entry
    X : array, shape (n, d)
        The data
    y : array, shape (n,)
        Binary labels (-1, 1)
    lamb : float
        The L2 regularization parameter


    Returns
    -------
    obj : float
        The value of the objective function
    grad : array, shape (d,) or (d+1,)
        The gradient of the objective function
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(theta)

    w, c, yz = _intercept_dot(theta, X, y)

    # Logistic loss is the negative of the log of the logistic function
    obj = -np.mean(log_logistic(yz)) + .5 * lamb * np.dot(w, w)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) / n_samples + lamb * w

    # Case where we fit the intercept
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum() / n_samples
    return obj, grad


def private_step_gd(X, y, gamma, n_nodes, obj_and_grad, theta_init,
    sigma=0,
    random_state=None,
    score=None,
    L=1
):
    """Local Gradient descent step. Performs a single private step of gradient descent algorithm. 

    Parameters
    ----------
    X : array, shape (n, d)
        The data
    y : array, shape (n,)
        Binary labels (-1, 1).
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_nodes : int
        number of nodes
    obj_and_grad : callable
        A function which takes as a vector of shape (p,), a dataset of shape (n_batch, d)
        and a label vector of shape (n_batch,), and returns the objective value and gradient.
    theta_init : array, shape (n_nodes, p)
        The initial value for the model parameters
    sigma : float
        Standard deviation of the Gaussian noise added to each gradient
    random_state : int
        Random seed to make the algorithm deterministic
    L : float
        Max norm for the gradient (clipped to L)


    Returns
    -------
    theta : array, shape=(n_nodes, p)
        The final value of the model parameters for each node
    """
    if score is None:
        score = lambda c: 42
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    p = theta_init.shape[1]
    
    theta = deepcopy(theta_init)
    samples_per_node = int(n/n_nodes)

    for idx_node in range(n_nodes):
        # Select all the samples belonging to this node (same size for all)
        idx = np.arange(idx_node*samples_per_node, (idx_node+1)*samples_per_node)
        # Compute the gradient on the private data of the node
        obj, grad = obj_and_grad(theta[idx_node], X[idx, :], y[idx])
        # Noise to be added to the gradient (max sensibility = .5 for normalized features on logistic regression)
        shield = rng.normal(scale=sigma*.5, size=p)

        u = grad + shield
        # clipping the gradient
        if LA.norm(u) > L:
            u = L*u/LA.norm(u)
        # update model
        theta[idx_node] -= gamma * (u)
        
    return theta


class MuffliatoLogisticRegression(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """Our sklearn estimator for private logistic regression defined as:
    min (1/n) \sum_i log_loss(theta;X[i,:],y[i]) + (lamb / 2) \|w\|^2,
    where theta = [w b]
    
    Parameters
    ----------
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    T : int
        The number of iterations
    n_nodes : int
        The number of nodes
    sigma : float
        Standard deviation of the Gaussian noise added to each gradient
    lamb : float
        The L2 regularization parameter    
    freq_obj_eval : int
        Specifies the frequency (in number of iterations) at which we compute the objective
    n_obj_eval : int
        The number of points on which we evaluate the objective
    stopping_criteria : string
        If we carry on with noise or just stop
    max_updates_per_node : float
        Max number of updates per node authorized due to privay constraint
    random_state : int
        Random seed to make the algorithm deterministic
    score : callable 
        Score used to evaluate the model (in practice sklearn score on test set)
    L : float
        Max norm for the gradient (clipped to L)
        
    Attributes
    ----------
    coef_ : (p,)
        The weights of the logistic regression model.
    intercept_ : (1,)
        The intercept term of the logistic regression model.
    obj_list_: list of length (n_iter / freq_obj_eval)
        A list containing the value of the objective function computed every freq_loss_eval iterations
    """
    
    def __init__(self, gamma, T, n_nodes,sigma, lamb=0, freq_obj_eval=10, n_obj_eval=1000, random_state=None, score=lambda c: lambda d: 0, L=1):
        self.gamma = gamma
        self.T = T
        self.n_nodes = n_nodes
        self.sigma = sigma
        self.lamb = lamb
        self.freq_obj_eval = freq_obj_eval
        self.n_obj_eval = n_obj_eval
        self.random_state = random_state
        self.score = score
        self.privacy_loss = np.zeros(n_nodes)
        self.L = L
    
    def fit(self, X, y):
        
        # check data and convert classes to {-1,1} if needed
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=[np.float64, np.float32], order="C")
        self.classes_ = np.unique(y)    
        y[y==self.classes_[0]] = -1
        y[y==self.classes_[1]] = 1
        n, p = X.shape

        
        obj_list = []
        scores = [] 
        # we draw a fixed subset of points to monitor the objective
        idx_eval = np.random.randint(0, n, self.n_obj_eval)
        
        theta = np.zeros((self.n_nodes, p+1)) # initialize parameters to zero
        # define the function for value and gradient needed by SGD
        # Compute a step at each node, depending on current local theta
        obj_grad = lambda theta, X, y: my_logistic_obj_and_grad(theta, X, y, lamb=self.lamb)
        for t in range(self.T):
            # evaluation code
            score_aux, obj_aux = 0,0
            if t % self.freq_obj_eval == 0:
            # evaluate objective
                for idx_node in range(self.n_nodes):
                    obj, _ = obj_grad(theta[idx_node], X[idx_eval, :], y[idx_eval])
                    obj_aux += obj
                    score_aux += self.score(np.unique(y))(theta[idx_node])
                score_aux, obj_aux = score_aux/self.n_nodes, obj_aux/self.n_nodes
                obj_list.append(obj_aux)
                scores.append(score_aux)

            # Compute a step of GD
            theta = private_step_gd(X, y, 
                self.gamma, 
                self.n_nodes, 
                obj_grad, 
                theta,
                sigma=self.sigma
            )
            # compute a new ER graph ensuring connectivity
            connex = False
            while not connex:	
                graph = nx.gnp_random_graph(self.n_nodes, 1*np.log(self.n_nodes)/self.n_nodes)
                connex = nx.is_connected(graph)
            graph = gossip_matrix(graph)
            # Compute the number of steps needed for a good gossip
            T_gossip = T_mix(graph, self.sigma)
            # compute the result of the gossip
            theta = gossip_vector(theta, graph, T_gossip)
            # Compute the privacy loss associated to this specific gossip
            self.privacy_loss += eps_random(graph, 0, T_gossip, 1, self.sigma)
        
        # save the learned model into the appropriate quantities used by sklearn
        self.intercept_ = np.expand_dims(theta[-1], axis=0)
        self.coef_ = np.expand_dims(theta[:-1], axis=0)
        
        # also save list of objective values during optimization for plotting
        self.obj_list_ = obj_list
        self.scores_ = scores
        
        return self



class CentralLogisticRegression(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """Our sklearn estimator for private logistic regression in a federated setting defined as:
    min (1/n) \sum_i log_loss(theta;X[i,:],y[i]) + (lamb / 2) \|w\|^2,
    where theta = [w b]
    
    Parameters
    ----------
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    T : int
        The number of iterations
    n_nodes : int
        The number of nodes
    sigma : float
        Standard deviation of the Gaussian noise added to each gradient
    lamb : float
        The L2 regularization parameter    
    freq_obj_eval : int
        Specifies the frequency (in number of iterations) at which we compute the objective
    n_obj_eval : int
        The number of points on which we evaluate the objective
    stopping_criteria : string
        If we carry on with noise or just stop
    max_updates_per_node : float
        Max number of updates per node authorized due to privay constraint
    random_state : int
        Random seed to make the algorithm deterministic
    score : callable 
        Score used to evaluate the model (in practice sklearn score on test set)
    L : float
        Max norm for the gradient (clipped to L)
        
    Attributes
    ----------
    coef_ : (p,)
        The weights of the logistic regression model.
    intercept_ : (1,)
        The intercept term of the logistic regression model.
    obj_list_: list of length (n_iter / freq_obj_eval)
        A list containing the value of the objective function computed every freq_loss_eval iterations
    """
    
    def __init__(self, gamma, T, n_nodes,sigma, lamb=0, freq_obj_eval=10, n_obj_eval=1000, random_state=None, score=lambda c: lambda d: 0, L=1):
        self.gamma = gamma
        self.T = T
        self.n_nodes = n_nodes
        self.sigma = sigma
        self.lamb = lamb
        self.freq_obj_eval = freq_obj_eval
        self.n_obj_eval = n_obj_eval
        self.random_state = random_state
        self.score = score
        self.privacy_loss = 0
        self.L = L
    
    def fit(self, X, y):
        
        # check data and convert classes to {-1,1} if needed
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=[np.float64, np.float32], order="C")
        self.classes_ = np.unique(y)    
        y[y==self.classes_[0]] = -1
        y[y==self.classes_[1]] = 1
        n, p = X.shape

        
        obj_list = []
        scores = [] 
        # we draw a fixed subset of points to monitor the objective
        idx_eval = np.random.randint(0, n, self.n_obj_eval)
        
        theta = np.zeros((self.n_nodes, p+1)) # initialize parameters to zero
        # define the function for value and gradient needed by SGD
        # Compute a step at each node, depending on current local theta
        obj_grad = lambda theta, X, y: my_logistic_obj_and_grad(theta, X, y, lamb=self.lamb)
        for t in range(self.T):
            # evaluation code
            if t % self.freq_obj_eval == 0:
            # evaluate objective: 0 suffices as all theta are equal
                obj_, _ = obj_grad(theta[0], X[idx_eval, :], y[idx_eval])
                score_ = self.score(np.unique(y))(theta[0])
                obj_list.append(obj_)
                scores.append(score_)

            # Compute a step of GD
            theta = private_step_gd(X, y, 
                self.gamma, 
                self.n_nodes, 
                obj_grad, 
                theta,
                sigma=self.sigma
            )
            # Compute the average
            theta = np.mean(theta, axis=0, keepdims=True)*np.ones_like(theta)

            # Compute the privacy loss associated to this specific gossip
            self.privacy_loss += eps_global(1, self.sigma)
        
        # save the learned model into the appropriate quantities used by sklearn
        self.intercept_ = np.expand_dims(theta[-1], axis=0)
        self.coef_ = np.expand_dims(theta[:-1], axis=0)
        
        # also save list of objective values during optimization for plotting
        self.obj_list_ = obj_list
        self.scores_ = scores
        
        return self
