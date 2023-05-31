import numpy as np

from sklearn.utils import check_random_state, _is_arraylike_not_scalar
from sklearn.cluster import kmeans_plusplus
from regime.stats import *

def do_E_step(X, means_, covars_, weights_):
    """
    perform E step:
    - compute proba of every sample being in each component
    - evaluate lower bound
    """
    neg_log_density = neg_log_density_normal(X, means_, covars_, "full")
    density = np.exp(-neg_log_density)
    proba_ = density * weights_
    proba_ /= proba_.sum(axis=1, keepdims = True)
    val_ = (proba_ * (-neg_log_density + np.log(weights_) - np.log(proba_))).sum()
    return proba_, val_


def do_M_step(X, proba_):
    """
    M step for a GMM, under full covars_.
    weighted MLE in each component.
    """
    # weights
    Ns_ = proba_.sum(axis=0)
    weights_ = Ns_/Ns_.sum()
    # means
    means_ = (proba_.T @ X) / Ns_[:, np.newaxis]
    # covars_
    X_demeaned = X[np.newaxis, :, :] - means_[:, np.newaxis, :]
    X_demeaned_weighted = X_demeaned * proba_.T[:, :, np.newaxis]
    covars_ = np.matmul(np.transpose(X_demeaned, axes=(0, 2, 1)), X_demeaned_weighted) / Ns_[:, np.newaxis, np.newaxis]
    return means_, covars_, weights_


class GMM():
    def __init__(self,
                 n_components = 2,
                 cov_type = 'full',
                 init = "k-means++",
                 n_init = 10,
                 max_iter = 300, 
                 tol = 1e-6,
                 random_state = None
                ):
        """
        GMM fitting via EM algo.
        Run different initializations and return the best one. Either input the center initial values, or use k-mean++ to initialize.
        
        Parameters:
        ----------------------------------
        n_components: int, default = 2
            number of components.
        cov_type: str, default 'full'
            
        init: str, or an array of shape (n_components, n_features)
            initial values for the centers.
        n_init: int, default = 10
            number of different initializations.
        max_iter: int, default = 300
            maximal number of iteration in each EM algo.
        tol: float,
            tolerance for the improvement of objective value
        random_state:
        
        
        """
        self.n_components = n_components
        self.cov_type = cov_type
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
    
    # shared by different models, to be unified
    def _check_init_n_init(self):
        """
        initialize the centers, to start the algo by following a E step.
        if the shape is consistent, 
        """
        if _is_arraylike_not_scalar(self.init) and self.init.shape == (self.n_components, self.n_features):
            # use this initial centers, don't need to rerun several times
            init = self.init
            n_init = 1
        else:
            # to be intialized by k-means++
            init = None
            n_init = self.n_init
        return init, n_init
        
    def fit(self, X):
        """
        fit the jump model by EM-type algo.
        """
        n_samples, n_features = X.shape
        n_components = self.n_components
        self.n_features = n_features
        
        # get attributes
        cov_type = self.cov_type
        max_iter = self.max_iter
        tol = self.tol
        random_state = self.random_state
        
        # check init and n_init
        init, n_init = self._check_init_n_init()
        
        # the best results over all initializations, compare to it in the last part of each iteration
        best_val_all_inits = -np.inf
        
        # iter over all the initializations
        for n_init_ in tqdm(range(n_init)):
            # initialize means
            if init is not None:
                means_ = init
            else:
                means_ = kmeans_plusplus(X, n_components, random_state=random_state)[0]
            # initialize covars, weights
            covars_ = np.repeat(np.cov(X.T, ddof=1)[np.newaxis, :, :], n_components, axis=0)
            weights_ = np.ones(n_components)/n_components
            
            # value in the previous iteration.
            val_pre = -np.inf
            # do one E step
            proba_, val_ = do_E_step(X, means_, covars_, weights_)    
            
            num_iter = 0
            # iterate between M and E steps
            while (num_iter < max_iter and val_ - val_pre  > tol):
                # update
                num_iter += 1
                # print(val_)
                val_pre = val_
                # M step
                means_, covars_, weights_ = do_M_step(X, proba_)
                # E step
                proba_, val_ = do_E_step(X, means_, covars_, weights_)
                # print(f"diff: {val_pre - val_}")
                
            # print(f"{n_init_}-th init: {num_iter} of iters, val: {val_}")
            if num_iter >= max_iter:
                print(f"No convergence: init {n_init_}")
            # print(val_)
            
            # compare with previous initializations
            if val_ > best_val_all_inits:
                best_val_all_inits = val_
                best_proba_all_inits = proba_
                best_means = means_
                best_covars = covars_
                best_weights = weights_
        
        self.means_ = best_means
        self.covars_ = best_covars
        self.weights_ = best_weights
        self.proba_ = best_proba_all_inits
        self.val_ = best_val_all_inits
        return self