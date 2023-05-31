import numpy as np
from itertools import product

from hmmlearn._hmmc import viterbi
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import BaseHMM
from scipy.special import logsumexp
from numpy.linalg import norm as norm_np


from sklearn.cluster import kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, clone

import logging
logging.basicConfig(level=logging.WARNING+1)

# from gurobipy import *
# from tqdm import tqdm


from regime.cluster_utils import *
from regime.stats import *



def jump_penalty_to_arr(penalty, n_c):
    """
    if penalty is float, raise it to the correct shape.
    otherwise unchanged.
    """
    if np.isscalar(penalty):
        return penalty * (np.ones((n_c, n_c)) - np.eye(n_c))
    return penalty

def discretize_prob_simplex(n_c, grid_size):
    """
    sample all the grid points on a simplex.
    Combinatorial complexity!
    """
    N = int(1/grid_size)
    tuples = filter(lambda x: sum(x)==N, product(range(N+1), repeat = n_c))
    lst = np.array(list(tuples)[::-1])/N
    return lst

def _sort_centers_by_first_feature(init):
    """
    sort all the centers in each init by the first feature.
    """
    n_i = len(init)
    idx = init[:, :, 0].argsort(axis=1)[:, ::-1]
    return init[np.arange(n_i)[:, np.newaxis] , idx]

def init_centers(X, n_c, n_init=10, init = "k-means++", random_state=None):
    """
    initialize the centers, by k-means++, for n_init times.
    """
    random_state = check_random_state(random_state)
    if init == "k-means++":
        centers = [kmeans_plusplus(X, n_c, random_state=random_state)[0] for _ in range(n_init)]
    elif init == "k-means":
        kmeans_instance = KMeans(n_c, n_init=10, random_state=random_state)
        centers = [kmeans_instance.fit(X).cluster_centers_ for _ in range(n_init)]
    else:
        raise NotImplementedError()
    return _sort_centers_by_first_feature(np.array(centers))

def init_k_means_plusplus(X, n_c, n_init=10, random_state=None):
    """
    initialize the centers, by k-means++, for n_init times.
    """
    return init_centers(X, n_c, n_init=n_init, init = "k-means++", random_state=random_state)
    # random_state = check_random_state(random_state)
    # init = [kmeans_plusplus(X, n_c, random_state=random_state)[0] for _ in range(n_init)]
    # return _sort_centers_by_first_feature(np.array(init))

    
def dp_py(loss, jump_penalty=0., init_penalty = None):
    """
    solve the dp problem involved in E step calculation (hard assignment). 
    Needed for each iteration. should be fast.
    
    Parameters:
    ---------------------------
    loss: arr, shape (n_s, n_c).
        the loss matrix of (l(t, k)).
    jump_penalty: float, arr of shape (n_c, n_c). default 0.
    
    init_penalty: arr, shape (n_c,). default None
        if not None, must be an array of shape (n_c,)
    """
    n_s, n_c = loss.shape
    # raise shape
    jump_penalty = jump_penalty_to_arr(jump_penalty, n_c)
        
    # res
    values, assign = np.empty((n_s, n_c)), np.empty(n_s, dtype=np.int32)
    
    # initial
    if init_penalty is None:
        values[0] = loss[0]
    else:
        values[0] = loss[0] + init_penalty
    # dp 
    for t in range(1, n_s):
        values[t] = loss[t] + (values[t-1][:, np.newaxis] + jump_penalty).min(axis=0)
        
    # find optimal path backwards
    assign[-1] = values[-1].argmin()
    value_opt = values[-1, assign[-1]]
    # traceback
    for t in range(n_s - 1, 0, -1):
        assign[t-1] = (values[t-1]+jump_penalty[:, assign[t]]).argmin()
    return assign, value_opt


def dp_viterbi(loss, jump_penalty=0., init_penalty = None, TPM = None, startprob_ = None):
    """
    implementation of dp via viterbi function in hmmlearn. It is fast when number of state is smaller than 100.
    """
    if TPM is None:
        TPM = np.exp(-jump_penalty_to_arr(jump_penalty, loss.shape[1]))
    if startprob_ is None:
        if init_penalty is None:
            startprob_ = np.ones(loss.shape[1])
        else:
            startprob_ = np.exp(-init_penalty)
    neg_value_opt, assign = viterbi(startprob_, TPM, -loss)
    return assign, -neg_value_opt



# def init_gurobi_model(n_s, n_c):
#     """
#     initiate a gurobi model, if qp is needed in the E step.
#     """
#     gm = Model()
#     gm.Params.LogToConsole = 0
#     constrs = []
    
#     assign = gm.addMVar((n_s, n_c), ub=1.)
#     constrs.append(
#         gm.addConstr(assign.sum(axis=1)==1)
#     )
    
#     assign_diff_ub = gm.addMVar((n_s-1, n_c), ub=1.)
#     constrs.append(
#         gm.addConstr(assign[:-1] - assign[1:] <= assign_diff_ub)
#     )
#     constrs.append(
#         gm.addConstr(assign[1:] - assign[:-1] <= assign_diff_ub)
#     )    
    
#     assign_diff_ub_row_sum = gm.addMVar(n_s-1, ub=2.)
#     constrs.append(
#         gm.addConstr(assign_diff_ub_row_sum == assign_diff_ub.sum(axis=1))
#     )
    
#     return gm, assign, assign_diff_ub_row_sum

# def solve_qp_E_step(loss, lambd=0., gm=None, assign=None, assign_diff_ub_row_sum=None):
#     """
#     solve the qp involved in E step.
#     returns proba, val
#     """
#     if gm is None:
#         gm, assign, assign_diff_ub_row_sum = init_gurobi_model(*loss.shape)
#     gm.setObjective((assign * loss).sum() + (.25*lambd)*(assign_diff_ub_row_sum**2).sum())
#     gm.optimize()    
    
#     return assign.X, gm.ObjVal



def do_M_step(X, proba_=None, labels_=None, n_c=None, centers_ = None):
    """
    perform a single M-step under l-2 loss and both soft / hard clustering.
    - use proba_ first, then labels.
    - dtype of labels_ is recommended to be int64.
    - previous centers values can be inputted to avoid the confusion when there is any class containing no points.
    """
    if centers_ is None:
        centers_new = np.empty((n_c, X.shape[1]))
    else:
        centers_new = centers_.copy()
    
    if proba_ is not None:  # cont model
        Ns_ = proba_.sum(axis=0)
        weighted_sum = proba_.T @ X

        idx = Ns_ > 0
        centers_new[idx] = weighted_sum[idx] / Ns_[idx][:, np.newaxis]
        return centers_new
        
    if labels_ is not None:  # discrete model
        for i in np.unique(labels_): # update centers
            centers_new[i] = X[labels_==i].mean(axis=0)
        return centers_new
    raise Exception("Either labels and proba is needed!")
    
class jump_model(BaseEstimator):
    """
    Jump model, either discrete or continuous.
    """
    def __init__(self,
                 n_components = 2, 
                 state_type = "cont",
                 covars_ = None,
                 cov_type = 'tied_diag',
                 jump_penalty = 0.,
                 solver = "dp",
                 alpha = 2,
                 grid_size = .05,
                 mode_loss = True,
                 init = None,
                 n_init = 10,
                 max_iter = 300, 
                 tol = 1e-6,
                 random_state = None
                ):
        """
        Jump model fitting. Both discrete and continuous models are in this class.
        Run several initializations and return the best one. Either input the center initial values, or use k-mean++ to initialize.
        
        Parameters:
        ----------------------------------
        n_components: int, default = 2
            number of components.
        state_type: either "cont" or "discrete".
        
        jump_penalty: float, or array of shape (n_components, n_components). default = 0.
            can be a penalty matrix for discrete model. For cont model, it is the lambd.
        covars_: 
            the cov mx of each component, when computing negative log likelihood as loss function.
            if None: learn from data (sample version), based on the cov_type, and not updated in the whole algo.
            if array like: prior shape information. shape must be consistent with cov_type
            ``to support later'': covars_ can be updated after each iteration. 
        cov_type: str, default 'tied_diag'
            the type of cov mx to learn from the data. "tied_diag" is just standard scaler.
        alpha: for state = "cont". scalar, default 2. 
            power of the l-1 norm jum penalty
        solver: for state = "cont". either "dp" or "qp".
            the solver for E step in the cont model.
            if "qp", alpha must equal 2.
        grid_size: for state = "cont" and solver="dp".
            grid size to discretize the probability simplex. Once initialized, grid_size can't be modified.
        mode_loss: for state = "cont" and solver="dp".
            whether to penalize the mode loss.
        init: an array of shape (n_i, n_components, n_features), or None.
            initial values for the centers.
            if none, initialize by k-means++.
        n_init: int, default = 10
            number of different initializations.
        max_iter: int, default = 300
            maximal number of iteration in each EM algo.
        tol: float,
            tolerance for the improvement of objective value
        random_state:
        
        
        """
        self.n_components, self.state_type = n_components, state_type
        self.covars_, self.cov_type = covars_, cov_type
        self.jump_penalty = jump_penalty
        self.solver = solver
        self.alpha, self.grid_size, self.mode_loss  = alpha, grid_size, mode_loss
        self.init, self.n_init = init, n_init
        self.max_iter, self.tol = max_iter, tol
        self.random_state = check_random_state(random_state)

        self.discrete = (self.state_type == "discrete")
        self.cont_dp = (self.state_type == "cont" and self.solver == "dp") 
        # self.cont_qp = (self.state_type == "cont" and self.solver == "qp")
        
        self.use_viterbi = True
        if self.cont_dp:
            # grid size cann't be changed later.
            self.prob_vecs = discretize_prob_simplex(self.n_components, self.grid_size)
            self.pairwise_dist = cdist(self.prob_vecs, self.prob_vecs, 'cityblock')/2
            if len(self.prob_vecs) > 100:
                self.use_viterbi = False   # higher speed
                
    def _check_jump_penalty(self):
        """
        check the jump penalty.
        - if discrete model, with float jump penalty, raise it to matrix.
        - if cont model w/ dp, multiply it with alpha power of pairwiase dist.
        """
        if self.discrete:
            return jump_penalty_to_arr(self.jump_penalty, self.n_components)
        if self.cont_dp:
            jump_penalty = self.jump_penalty * (self.pairwise_dist ** self.alpha)
            if self.mode_loss:
                mode_loss = logsumexp(-jump_penalty, axis=1, keepdims=True)
                mode_loss -= mode_loss[0]
                jump_penalty += mode_loss
            return jump_penalty
        # qp
        return self.jump_penalty
    
    def _check_init(self, X):
        """
        initialize the centers, either filled by inputs, or by k-means++.
        start the EM iteration, by following a E step.
        """
        n_i = 0  # already inputted initial centers
        if self.init is not None and check_arr_shape(self.init[0], (self.n_components, X.shape[1]), string="initial centers"):
            n_i = len(self.init)
        #        
        n_init = max(n_i, self.n_init)
        #
        init = np.empty((n_init, self.n_components, X.shape[1]))
        init[:n_i] = self.init
        for i_ in range(n_i, n_init):
            init[i_] = kmeans_plusplus(X, self.n_components, random_state=self.random_state)[0]
        # sort the centers by the first element (assumed to be some returns), in descending order, so that crash periods in regimes w/ higher no.
        return _sort_centers_by_first_feature(init), n_init
    

    # def _check_gurobi(self, n_s):
    #     if self.cont_qp:
    #         self.gm, self.assign, self.assign_diff_ub_row_sum = init_gurobi_model(n_s, self.n_components)
    #     return

    def _check_covars_(self, X):
        """
        - if the input covars_ is None, learn  from data.
        - if the input covars_ doesn't have the correct shape, raise Exception.
        """
        if self.covars_ is None: # learn form data
            return compute_covars_from_data(X, self.cov_type)
        if check_arr_shape(self.covars_, 
                           get_correct_covars_shape(self.n_components, X.shape[1], self.cov_type),
                          action="exception"):
            return self.covars_
        else:
            raise Exception("not supported yet!")
        
    def do_E_step(self, X=None, centers_=None, covars_=None, cov_type=None, X_scaled=None, scales=None, 
                  jump_penalty=None, TPM=None, startprob_=None):
        # compute loss matrix
        loss_mx = neg_log_density_normal(X, centers_, covars_, cov_type, X_scaled, scales)
        if self.cont_dp:
            loss_sample_to_proba_vec = loss_mx @ self.prob_vecs.T
            if self.use_viterbi:
                labels_, val_ = dp_viterbi(loss_sample_to_proba_vec, TPM=TPM, startprob_=startprob_)
            else:
                labels_, val_ = dp_py(loss_sample_to_proba_vec, jump_penalty)  
            proba_ = self.prob_vecs[labels_]
            return proba_, labels_, val_      
            
        if self.discrete:
            labels_, val_ = dp_viterbi(loss_mx, TPM=TPM, startprob_=startprob_)    
            return None, labels_, val_
        #_hmmc.viterbi(np.ones(self.n_components), TPM, -loss_mx)
            # labels_, val_ = dp(loss_mx, jump_penalty) 
            
        # if self.cont_qp:
        #     proba_, val_ = solve_qp_E_step(loss_mx, jump_penalty, self.gm, self.assign, self.assign_diff_ub_row_sum)
        #     return proba_, None, val_
        
    def fit(self, X):
        """
        fit the jump model by EM-type algo.
        """
        # shape
        n_s, n_f = X.shape
        # self.n_features = n_f
        
        # get attributes
        n_c = self.n_components
        cov_type = self.cov_type
        max_iter, tol = self.max_iter, self.tol
        random_state = self.random_state
        
        ### initialize:
        # covars_, loss shape
        covars_ = self._check_covars_(X)
        # penalty
        jump_penalty = self._check_jump_penalty()
        TPM, startprob_ = None, None
        if self.use_viterbi:
            TPM = np.exp(-jump_penalty)
            startprob_ = np.ones(len(jump_penalty))
            
        # init
        init, n_init = self._check_init(X)
        # # gurobi, if needs qp 
        # self._check_gurobi(n_s)
        
        # get scales at the outset. updating covars during the iterations is not yet supported.
        scales = compute_scales_from_covars(covars_, cov_type)
        X_scaled = scale_X(X, scales, cov_type)
        
        # the best results over all initializations, compare to it in the last part of each iteration
        best_val_all_inits = np.inf
        best_labels_all_inits = None
        
        # iter over all the initializations
        # for n_init_ in tqdm(range(n_init)):
        for n_init_ in range(n_init):
            # initialize centers
            centers_ = init[n_init_]
            # labels and value in the previous iteration.
            labels_pre, val_pre = None, np.inf
            # do one E step
            proba_, labels_, val_ = self.do_E_step(centers_=centers_, cov_type=cov_type, X_scaled=X_scaled, scales=scales,
                                                   jump_penalty=jump_penalty, TPM=TPM, startprob_=startprob_)          
            num_iter = 0
            # iterate between M and E steps
            while (num_iter < max_iter and not is_same_clustering(labels_, labels_pre) and val_pre - val_ > tol):
                # update
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                # M step
                centers_ = do_M_step(X, proba_, labels_, n_c, centers_)
                # E step
                proba_, labels_, val_ = self.do_E_step(centers_=centers_, cov_type=cov_type, X_scaled=X_scaled, scales=scales,
                                                       jump_penalty=jump_penalty, TPM=TPM, startprob_=startprob_)
            # print(f"{n_init_}-th init: {num_iter} of iters, val: {val_}")
            # print(centers_)
            
            # compare with previous initializations
            # best_val_all_inits == np.inf or (not is_same_clustering(labels_, best_labels_all_inits) and val_ < best_val_all_inits):
            if not is_same_clustering(best_labels_all_inits, labels_) and val_ < best_val_all_inits:
                best_val_all_inits = val_
                best_labels_all_inits = labels_
                best_centers_all_inits = centers_
                best_proba_all_inits = proba_    
            
        self.val_ = best_val_all_inits
        self.labels_ = best_labels_all_inits
        self.centers_ = best_centers_all_inits
        self.proba_ = best_proba_all_inits
                
        if self.discrete:
            del self.proba_
            
        if self.cont_dp:
            del self.labels_
            
        # sort returns
        idx = self.centers_[:, 0].argsort()[::-1]
        self.centers_ = self.centers_[idx]
        if hasattr(self, "proba_"):
            self.proba_ = self.proba_[:, idx]
        if hasattr(self, "labels_"):
            new_labels = np.full_like(self.labels_, -1, dtype=np.int32)
            for i in range(n_c):
                new_labels[self.labels_==idx[i]] = i
            self.labels_ = new_labels
            
        return self
    
    
    
    
#####################################
## feature selection (sparse models)
####################################

def soft_thres_l2_normalized(x, thres):
    """
    soft thresholding for a nonneg vec x. normalize to to have unit length.
    """
    y = np.maximum(0, x-thres)
    return y / norm_np(y)


def binary_search_decrease(f, left, right, value,  *args, tol=1e-5, max_iter=100, **kwargs):
    """
    binary search for a decreasing function.
    """
    gap = right-left
    num_iter = 0
    while (gap > tol and num_iter < max_iter):
        num_iter += 1
        middle = (right + left) / 2
        func_call = f(middle, *args, **kwargs)
        if func_call < value:
            right = middle
        elif func_call > value:
            left = middle
        else:
            return middle
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
    return soft_thres_l2_normalized(a_plus, thres)


##############################
## baseline model
##############################

class GaussianHMM_model(BaseEstimator):
    """
    GaussianHMM estimation. support several initializations.
    """
    def __init__(self,
                 n_components = 2,
                 n_init = 10,
                 init = "k-means++",
                 random_state = None,
                 **kwargs
                ):
        self.n_components = n_components
        self.n_init = n_init
        self.init = init
        self.random_state = check_random_state(random_state)
        self.hmm_instance = GaussianHMM(n_components,
                                        init_params="sct",
                                        random_state=self.random_state, 
                                        **kwargs
                                       )
        
    def fit(self, X):
        n_c = self.n_components; n_init = self.n_init; init = self.init; hmm_instance = self.hmm_instance
        # initialization by k-means++
        init = init_centers(X, n_c, n_init=n_init, init = init, random_state=self.random_state)
        best_score = -np.inf
        # iter over all inits
        for i_i in range(n_init):
            # fit
            hmm_instance.means_ = init[i_i]
            try:
                hmm_instance.fit(X)
            except:
                continue
            # score
            try:
                score = hmm_instance.score(X)
            except:
                continue
            # print(f"{i_i}: {score}. means: {hmm_instance.means_}")
            if score > best_score:
                best_idx = i_i
                best_score = score
                best_res = {"means_": hmm_instance.means_, 
                            "_covars_": hmm_instance._covars_, 
                            "transmat_": hmm_instance.transmat_,
                           "startprob_": hmm_instance.startprob_}
        self.best_res = best_res
        # print(best_idx)
        hmm_instance.means_ = best_res["means_"]; hmm_instance._covars_ = best_res["_covars_"]
        hmm_instance.transmat_ = best_res["transmat_"]; hmm_instance.startprob_ = best_res["startprob_"]
        # save res
        self.means_ = best_res["means_"].squeeze(); self.covars_ = best_res["_covars_"].squeeze(); self.transmat_ = best_res["transmat_"]
        self.labels_ = hmm_instance.predict(X).astype(np.int32)
        self.proba_ = hmm_instance.predict_proba(X)
        return self
    
    
    
# class GaussianHMM_model(BaseEstimator):
#     """
#     GaussianHMM estimation. support several initializations.
#     """
#     def __init__(self,
#                  n_components = 2,
#                  n_init = 10,
#                  init = "k-means++",
#                  random_state = None,
#                  **kwargs
#                 ):
#         self.n_components = n_components
#         self.n_init = n_init
#         self.init = init
#         self.random_state = check_random_state(random_state)
#         self.hmm_instance = GaussianHMM(n_components, 
#                                         covariance_type='full',
#                                         init_params="sct",
#                                         random_state=self.random_state, 
#                                         **kwargs
#                                        )
        
#     def fit(self, X):
#         n_c = self.n_components; n_init = self.n_init; init = self.init; hmm_instance = self.hmm_instance
#         # initialization by k-means++
#         init = init_centers(X, n_c, n_init=n_init, init = init, random_state=self.random_state)
#         best_score = -np.inf
#         # iter over all inits
#         for i_i in range(n_init):
#             # fit
#             hmm_instance.means_ = init[i_i]
#             try:
#                 hmm_instance.fit(X)
#             except:
#                 continue
#             # score
#             try:
#                 score = hmm_instance.score(X)
#             except:
#                 continue
#             # print(f"{i_i}: {score}. means: {hmm_instance.means_}")
#             if score > best_score:
#                 best_idx = i_i
#                 best_score = score
#                 best_res = {"means_": hmm_instance.means_, 
#                             "covars_": hmm_instance.covars_, 
#                             "transmat_": hmm_instance.transmat_,
#                            "startprob_": hmm_instance.startprob_}
#         self.best_res = best_res
#         # print(best_idx)
#         hmm_instance.means_ = best_res["means_"]; hmm_instance.covars_ = best_res["covars_"]
#         hmm_instance.transmat_ = best_res["transmat_"]; hmm_instance.startprob_ = best_res["startprob_"]
#         # save res
#         self.means_ = best_res["means_"].squeeze(); self.covars_ = best_res["covars_"].squeeze(); self.transmat_ = best_res["transmat_"]
#         self.labels_ = hmm_instance.predict(X).astype(np.int32)
#         self.proba_ = hmm_instance.predict_proba(X)
#         return self
