"""""""""""""""""""""
Module for computing log pdf for normal, under every component dist.

Benchmark against the module <hmmlearn.stats>

every of the four methods is either faster, or the same with hmmlearn codes, plus we allow precomputed cholesky factorization, and the scaled data mx.

"""""""""""""""""""""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import cdist
# from scipy.ndimage import shift
from scipy.linalg import cholesky, solve_triangular, eig, expm, logm

# checked
def compute_scales_from_covars(covars_, cov_type = "tied_diag"):
    if cov_type in ['tied_diag', "diag"]:
        return np.sqrt(covars_)
    elif cov_type == 'tied_full':
        return cholesky(covars_, lower=True)
    elif cov_type == 'full':
        return np.array([cholesky(covar, lower=True) for covar in covars_]) # (n_c, n_f, n_f)
    else:
        raise Exception("Unsupported cov type!!")
        
def scale_X(X, scales, cov_type = "tied_diag"):
    if cov_type == 'tied_diag':
        return X / scales
    elif cov_type == 'tied_full':
        return solve_triangular(scales, X.T, lower=True).T
    elif cov_type == "diag":
        return X[np.newaxis, :, :] / scales[:, np.newaxis, :]
    elif cov_type == 'full':
        return np.array([solve_triangular(scale, X.T, lower=True).T for scale in scales]) # (n_c, n_s, n_f)
    else:
        raise Exception("Unsupported cov type!!")        

def scale_centers(centers_, scales, cov_type = "tied_diag"):
    if cov_type in ['tied_diag', "diag"]:
        return centers_ / scales
    elif cov_type == 'tied_full':
        return solve_triangular(scales, centers_.T, lower=True).T
    elif cov_type == 'full':
        return np.array([solve_triangular(scale, center, lower=True) for scale, center in zip(scales, centers_)])   # (n_c, n_f)
    else:
        raise Exception("Unsupported cov type!!")
        
def neg_log_density_normal(X, centers_, covars_, cov_type="tied_diag", X_scaled=None, scales=None):
    """
    compute the negative log density of multivariate normal of each data point under each component. the constant of d * np.log(2*np.pi) is neglected.
    The result is a mx of
    .5*( (x_t-mu_k) @ Sigma_k^{-1} @ (x_t-mu_k) + log |Sigma_k| )_{t, k}.
    
    You can precompute cholesky decompostion L (scales), and the scaled version of X, i.e. L^{-1} applied to each row of X (X_scaled). Then we don't need the input of X and covars_, and only need to scale centers in computation.
    
    Otherwise every time cholesky decompositions are performed, potentially expensive
    
    Parameters:
    -----------------------------------
    X: (n_samples, n_features)
    
    centers_: (n_components, n_features)
    
    covars_: shape depends on cov_type, now supports
    - "tied_diag": (n_f,), the same dignoal cov mx for all the components, 
    - "tied_full": (n_f, n_f), the same full cov mx for all the components, 
    - "diag": (n_c, n_f), dignoal cov mx for every component
    - "full": (n_c, n_f, n_f), full cov mx for every component
    """
    if scales is None: 
        scales = compute_scales_from_covars(covars_, cov_type)
    if X_scaled is None:
        X_scaled = scale_X(X, scales, cov_type)        
    # scale centers
    centers_scaled = scale_centers(centers_, scales, cov_type)
    # output
    if cov_type == 'tied_diag':
        return .5*cdist(X_scaled, centers_scaled)**2 + np.log(scales).sum()
    elif cov_type == 'tied_full':
        return .5*cdist(X_scaled, centers_scaled) ** 2 + np.log(np.diag(scales)).sum()
    elif cov_type == "diag":
        return .5 * ((X_scaled - centers_scaled[:, np.newaxis, :])**2).sum(axis=-1).T + np.log(scales).sum(axis=-1)
    elif cov_type == 'full':
        return .5 * ((X_scaled - centers_scaled[:, np.newaxis, :])**2).sum(axis=-1).T + np.log(np.diagonal(scales, offset=0, axis1=1, axis2=2)).sum(axis=-1)        
        
        
def compute_covars_from_data(X, cov_type = "tied_diag", labels_=None):
    if cov_type == "tied_diag":
        return X.var(axis=0, ddof=1)
    elif cov_type == "tied_full":
        return np.cov(X.T)
    else:
        raise Exception("Unsupported cov type!!")


def get_correct_covars_shape(n_c, n_f, cov_type = "tied_diag"):
    if cov_type == 'tied_diag':
        return (n_f,)
    elif cov_type == 'tied_full':
        return (n_f, n_f)
    elif cov_type == "diag":
        return (n_c, n_f)
    elif cov_type == 'full':
        return (n_c, n_f, n_f)
    else:
        raise Exception("Unsupported cov type!!")

        
        
#######################

def empirical_trans_mx(labels_, n_components = 2, return_counts = False):
    """
    compute the empirical transition counts / prob mx.
    
    labels must be in 0, 1, ...,  K-1
    """
    # count transitions
    count_mx = np.zeros((n_components, n_components), dtype=int)
    for i in range(n_components):
        # idx prev states
        idx = np.where(labels_[:-1]==i)[0]
        # idx next states
        idx_next_state = idx+1
        # count next states
        states, counts = np.unique(labels_[idx_next_state], return_counts=True)
        count_mx[i, states] = counts
    if return_counts:
        return count_mx
    prob_mx = np.full_like(count_mx, np.nan, dtype=float)#np.eye(n_components)
    total_nums = count_mx.sum(axis=1)
    idx = total_nums>0
    prob_mx[idx] = count_mx[idx] / total_nums[idx][:, np.newaxis]
    return prob_mx



def scale_params(means_, covars_, transmat_, scale1, scale2):
    """
    scale the params of mean, cov mx and trans prob mx, by time scale.
    e.g. scale1=20, scale2 = 1, monthly to daily
    """
    means_new = means_ * scale2 / scale1
    covars_new = covars_ * scale2 / scale1
    transmat_new = expm(logm(transmat_) * scale2 / scale1)
    return means_new, covars_new, transmat_new


def invariant_dist_transmat_(transmat_):
    """
    compute the invariant distribution of a transition proba matrix
    """
    # decomposition
    eigval, eigvec = eig(transmat_.T)
    # select the cols, real parts
    eigvec_val1 = np.real(eigvec[:, np.isclose(eigval, 1)])
    # all nonneg
    eigvec_val1 = eigvec_val1[:, (eigvec_val1 >= 0).all(axis=0)]
    return eigvec_val1[:, 0]/eigvec_val1[:, 0].sum()



#######################
## features
#######################

# zheng features

# gpt4 through chatgpt plus wrote this.
def custom_slice(X, left, right):
    """
    i-th row is X[i+left:i+right+1]
    """
    X = X.astype(float)
    n = len(X)
    width = right - left + 1
    Y = np.empty((n, width))

    # Pad the original array with NaNs
    X_padded = np.pad(X, pad_width=((abs(left), abs(right)), (0, 0)), mode='constant', constant_values=np.nan)

    # Shift the indices according to the given left and right values
    shifted_start = abs(left) + left
    shifted_end = abs(left) + n + left

    # Use strides to get the desired shape and fill the values
    strides = X_padded.strides
    Y = as_strided(X_padded[shifted_start:shifted_end], shape=(n, width), strides=(strides[0], strides[0]))
    return Y


def extract_mean_std_from_sliding_window(X, left, right):
    X_window = pd.DataFrame(custom_slice(X, left, right))
    return pd.DataFrame({"mean":X_window.mean(axis=1), "std":X_window.std(axis=1, ddof=1)}).to_numpy()


def custom_abs_diff(X, k):
    X_df = pd.DataFrame(X)
    return abs((X_df.shift(-k) - X_df.shift(-k+1)).to_numpy())

def feature_engineer_zheng(X, backward = True):
    """
    X is 2-dim.
    
    extract:
    - x_t
    - |x_t-x_{t-1}|
    - |x_{t-1}-x_{t-2}|
    and for each window, its six features
    - mean, 
    - std, 
    - left mean, 
    - left std, 
    - right mean, 
    - right std.    
    """
    res = [X]
    if backward:
        res += [custom_abs_diff(X, 0), custom_abs_diff(X, -1)]
        for l in [6, 14]:
            for start, end in [[-l+1, 0], [-l+1, -int(l/2)], [-int(l/2)+1, 0]]:
                res.append(extract_mean_std_from_sliding_window(X, start, end))
        return np.concatenate(res, axis=1)
    if not backward:
        res += [custom_abs_diff(X, 0), custom_abs_diff(X, 1)]
        for l in [5, 13]:
            temp = int(l/2)
            for start, end in [[-temp, temp], [-temp, 0], [0, temp]]:
                res.append(extract_mean_std_from_sliding_window(X, start, end))
        return np.concatenate(res, axis=1)
    
    
def feature_engineer_zheng_batch(Xs, backward = True):
    """
    Xs is 3-dim.
    extract zheng features for a batch of sequences.
    
    extract:
    - x_t
    - |x_t-x_{t-1}|
    - |x_{t-1}-x_{t-2}|
    and for each window, its six features
    - mean, 
    - std, 
    - left mean, 
    - left std, 
    - right mean, 
    - right std.    
    """
    return np.array([feature_engineer_zheng(X, backward) for X in Xs])


def ewma_mean_std(X, halflife):
    X_ser = pd.Series(X.squeeze())
    X_ewm = X_ser.ewm(halflife=halflife)
    return pd.DataFrame({"mean": X_ewm.mean(), "std": X_ewm.std()}).to_numpy()

def feature_engineer_ewm(X, halflife_list = [1, 2, 5, 10, 20]):
    res = [X, custom_abs_diff(X, 0), custom_abs_diff(X, -1)]
    for halflife in halflife_list:
        res.append(ewma_mean_std(X, halflife))
    return np.concatenate(res, axis=1)

def feature_engineer_ewm_batch(Xs, halflife_list = [1, 2, 5, 10, 20]):
    """
      
    """
    return np.array([feature_engineer_ewm(X, halflife_list) for X in Xs])









# def extract_windows(X, left=5, right=0):
#     """
#     extract a window around each number, with number of elements specified on the left or right.
#     """
#     n_s = len(X)
#     X_window = np.full((n_s, left+right+1), np.nan)
#     for i in range(left, n_s-right):
#         X_window[i] = X[(i-left):(i+right+1), :].squeeze()
#     return X_window

# def extract_mean_std_from_window(X_window):
#     """
#     From a window, extract the six features:
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.
#     """
#     n_window = X_window.shape[1]
#     res = []
#     res.append(X_window.mean(axis=1))
#     res.append(X_window.std(axis=1))
#     N1, N2 = int((n_window+1)/2), int(n_window/2)
#     res.append(X_window[:, :N1].mean(axis=1))
#     res.append(X_window[:, :N1].std(axis=1))
#     res.append(X_window[:, N2:].mean(axis=1))
#     res.append(X_window[:, N2:].std(axis=1))
#     return np.array(res).T

# def compute_abs_diff(X):
#     """
#     compute x_t, |x_t-x_{t-1}| and |x_{t-1}-x_{t-2}|
#     """
#     roll1, roll2 = shift(X, (1, 0), cval=np.nan), shift(X, (2, 0), cval=np.nan)
#     return np.concatenate([X, abs(X-roll1), abs(roll1-roll2)], axis=1)


# def extract_features_zheng(X, window_list = [[5, 0], [13, 0]]):
#     """
#     X is 2-dim.
    
#     extract:
#     - x_t
#     - |x_t-x_{t-1}|
#     - |x_{t-1}-x_{t-2}|
#     and for each window, its six features
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.    
#     """
#     X_windows = [extract_windows(X, left, right) for left, right in window_list]
#     X_feats_windows = [extract_mean_std_from_window(X_window) for X_window in X_windows]
#     return np.concatenate([compute_abs_diff(X)] + X_feats_windows, axis=1)


# def extract_features_zheng_batch(Xs, window_list = [[5, 0], [13, 0]]):
#     """
#     Xs is 3-dim.
#     extract zheng features for a batch of sequences.
    
#     extract:
#     - x_t
#     - |x_t-x_{t-1}|
#     - |x_{t-1}-x_{t-2}|
#     and for each window, its six features
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.    
#     """
#     return np.array([extract_features_zheng(X, window_list) for X in Xs])




# def extract_windows(X, left=5, right=0):
#     """
#     extract a window around each number, with number of elements specified on the left or right.
#     """
#     n_s = len(X)
#     X_window = np.empty((n_s, left+right+1)) * np.nan
#     for i in range(left, n_s-right):
#         X_window[i] = X[(i-left):(i+right+1), :].squeeze()
#     return X_window

# def extract_mean_std_from_window(X_window):
#     """
#     From a window, extract the six features:
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.
#     """
#     n_window = X_window.shape[1]
#     res = []
#     res.append(X_window.mean(axis=1))
#     res.append(X_window.std(axis=1))
#     N1, N2 = int((n_window+1)/2), int(n_window/2)
#     res.append(X_window[:, :N1].mean(axis=1))
#     res.append(X_window[:, :N1].std(axis=1))
#     res.append(X_window[:, N2:].mean(axis=1))
#     res.append(X_window[:, N2:].std(axis=1))
#     return np.array(res).T

# def compute_abs_diff(X):
#     """
#     compute x_t, |x_t-x_{t-1}| and |x_{t-1}-x_{t-2}|
#     """
#     roll1, roll2 = shift(X, (1, 0), cval=np.nan), shift(X, (2, 0), cval=np.nan)
#     return np.concatenate([X, abs(X-roll1), abs(roll1-roll2)], axis=1)


# def extract_features_zheng(X, window_list = [[5, 0], [13, 0]]):
#     """
#     X is 2-dim.
    
#     extract:
#     - x_t
#     - |x_t-x_{t-1}|
#     - |x_{t-1}-x_{t-2}|
#     and for each window, its six features
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.    
#     """
#     X_windows = [extract_windows(X, left, right) for left, right in window_list]
#     X_feats_windows = [extract_mean_std_from_window(X_window) for X_window in X_windows]
#     return np.concatenate([compute_abs_diff(X)] + X_feats_windows, axis=1)
#     # X_window1 = extract_windows(X)
#     # X_window2 = extract_windows(X, 13, 0)
#     # return np.concatenate([compute_abs_diff(X), extract_mean_std_from_window(X_window1), extract_mean_std_from_window(X_window2)], axis=1)

# def extract_features_zheng_batch(Xs):
#     """
#     Xs is 3-dim.
#     extract zheng features for a batch of sequences.
    
#     extract:
#     - x_t
#     - |x_t-x_{t-1}|
#     - |x_{t-1}-x_{t-2}|
#     and for each window, its six features
#     - mean, 
#     - std, 
#     - left mean, 
#     - left std, 
#     - right mean, 
#     - right std.    
#     """
#     return np.array([extract_features_zheng(X) for X in Xs])



# def empirical_trans_mx(labels, n_components = 2, return_counts = False):
#     """
#     compute the empirical transition counts / prob mx.
    
#     labels must be in 0, 1, ...,  K-1
#     """
#     # count transitions
#     count_mx = np.zeros((n_components, n_components), dtype=int)
#     for i in range(n_components):
#         # idx prev states
#         idx = np.where(labels[:-1]==i)[0]
#         # idx next states
#         idx_next_state = idx+1
#         # count next states
#         states, counts = np.unique(labels[idx_next_state], return_counts=True)
#         count_mx[i, states] = counts
#     if return_counts:
#         return count_mx
#     prob_mx = np.eye(n_components)
#     total_nums = count_mx.sum(axis=1)
#     idx = total_nums>0
#     prob_mx[idx] = count_mx[idx] / total_nums[idx][:, np.newaxis]
#     return prob_mx


