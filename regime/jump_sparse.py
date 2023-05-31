import numpy as np
import numpy.linalg as npla

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from regime.cluster_utils import *
from regime.jump import *

####################################
## solve lasso
####################################

def soft_thres_l2_normalized(x, thres):
    """
    soft thresholding for a nonneg vec x. normalize to to have unit length.
    """
    y = np.maximum(0, x-thres)
    return y / npla.norm(y)

def binary_search_decrease(f, left, right, value,  *args, tol=1e-5, max_iter=100, **kwargs):
    """
    binary search for a decreasing function.
    """
    gap = right-left
    num_iter = 0
    while (gap > tol and num_iter < max_iter):
        # print(f"{left}, {right}")
        num_iter += 1
        middle = (right + left) / 2
        func_call = f(middle, *args, **kwargs)
        if func_call < value:
            right = middle
        elif func_call > value:
            left = middle
        else:
            return middle
        gap /= 2
    if num_iter < max_iter:
        return middle
    raise Exception("Not convergence. must be math error.")
    
def solve_lasso(a, s):
    """
    solve the lasso problem involved in updating the feature weight vector.
    """
    a_plus = np.maximum(a, 0)
    f = lambda x: soft_thres_l2_normalized(a_plus, x).sum()
    thres = binary_search_decrease(f, 0, max(a_plus), s)
    return soft_thres_l2_normalized(a, thres)


def get_BCSS(X, labels_, n_c):
    """
    compute the BCSS for a clustering result.
    """
    centers_ = do_M_step(X, labels_=labels_, n_c=n_c)
    mu = X.mean(0)
    # number of points in each cluster
    num_in_cluster = np.zeros(n_c, dtype=int)
    elements, counts = np.unique(labels_, return_counts=True)
    num_in_cluster[elements] = counts
    # 
    return (((centers_ - mu)**2) * num_in_cluster[:, np.newaxis]).sum(0)

class sparse_KMeans(BaseEstimator):
    """
    Implementation of sparse k-means proposed by daniela witten.
    """
    def __init__(self, n_clusters = 2, thres = 1., max_iter = 10, tol = 1e-4, random_state = None):
        """
        Each k-means fit is initiated 10 times by k-means++.
        
        """
        self.n_clusters = n_clusters
        self.thres = thres
        self.tol, self.max_iter = tol, max_iter
        self.random_state = check_random_state(random_state)
        self.kmeans_model = KMeans(n_clusters, n_init=10, random_state=self.random_state)
        
    def fit(self, X):
        n_s, n_f = X.shape
        thres = np.clip(self.thres, 1, np.sqrt(n_f))
        n_c = self.n_clusters
        tol, max_iter = self.tol, self.max_iter
        kmeans_model = self.kmeans_model
                
        # 
        num_iter = 0
        w = np.ones(n_f) / np.sqrt(n_f)
        labels_ = None
        while (num_iter < max_iter):     # and (w_old is None or npla.norm(w-w_old, 1)/npla.norm(w_old, 1) > tol) and not is_same_clustering(labels_, labels_old)):
            # print(f"w: {w}")
            num_iter += 1
            w_old = w
            labels_old = labels_
            
            # update labels
            weights = np.sqrt(np.sqrt(n_f)*w)
            labels_ = kmeans_model.fit(X * weights).labels_
            if is_same_clustering(labels_, labels_old):
                break
            
            # update w
            BCSS = get_BCSS(X, labels_, n_c)
            w = solve_lasso(BCSS, thres)
            if npla.norm(w-w_old, 1)/npla.norm(w_old, 1) <= tol:
                break
        
        self.labels_ = labels_
        self.centers_ = do_M_step(X, labels_=labels_, n_c=n_c)
        self.w = w/w.sum()
        return self