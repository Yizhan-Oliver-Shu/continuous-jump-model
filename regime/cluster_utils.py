# basic
import pandas as pd
import numpy as np
from tqdm import tqdm

# import the following helpers for any clustering related tasks
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.utils import _is_arraylike_not_scalar

from itertools import permutations

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# mpl.style.use("seaborn-v0_8")
# plt.rcParams['figure.figsize'] = [9, 9]
# plt.rcParams['axes.titlesize'] = 20
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['xtick.labelsize'] = 15
# plt.rcParams['ytick.labelsize'] = 15
# plt.rcParams['legend.fontsize'] = 18
# plt.rcParams['font.size'] = 30

# # warnings
# pd.options.mode.chained_assignment = None
# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)



# helper functions
def plot_2d_data_clusters(X, y, ax=None):
    """
    plot 2d data, colored by clusters.
    
    Parameters:
    ------------------------
    X:
    
    y:
        cluster labels
    """
    if X.shape[1] != 2:
        raise Exception("Can only plot 2d data!")
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    return ax


# def is_same_clustering(labels1, labels2, n_clusters):
#     """
#     check whether two groupings are the same under permutation. Can only accept labels within {0, 1, ..., n_clusters-1}
#     """
#     # check whether the two labels are within {0, 1, ..., n_clusters-1}
#     labels1, labels2 = np.array(labels1, dtype=np.int32), np.array(labels2, dtype=np.int32)
#     if min(labels1) < 0 or max(labels1) > n_clusters - 1 or min(labels2) < 0 or max(labels2) > n_clusters - 1:
#         raise Exception("only accept label within 0, 1, ..., n_clusters-1.")
#     return _is_same_clustering(labels1, labels2, n_clusters) and _is_same_clustering(labels2, labels1, n_clusters)


def _is_map_from_left_to_right(labels_left, labels_right):
    """
    check whether the map from the left labels to the right is indeed a map.
    if either labels is None, return False
    """
    if labels_left is None or labels_right is None:
        return False
    unique_left = np.unique(labels_left)
    for label in unique_left:
        if len(np.unique(labels_right[labels_left==label])) != 1:
            return False
    return True

def is_same_clustering(labels1, labels2):
    """
    check whether two clustering results are the same, under permutation.
    not the same result if the unique labels of the two results don't match.
    
    if either input is None, return False
    """
    return _is_map_from_left_to_right(labels1, labels2) and _is_map_from_left_to_right(labels2, labels1)

# def map_clustering_to_arange(labels):
#     """
#     if a clustering result is not 
#     """
#     labels = np.array(labels)
#     # unique labels
#     classes = np.unique(labels)
#     if np.array_equal(classes, np.arange(0, len(classes))):
#         return labels
#     res = labels.copy()
#     for i, class_ in enumerate(classes):
#         res[res==class_] = i
#     return res

def generate_imbalanced_data():
    """
    generate an adversely chosen random seed for k-means.
    """
    n_samples = 1500
    n_clusters = 3

    # generate random data
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=170)
    # make it imbalanced
    X = np.vstack(
        (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
    )  # Unevenly sized blobs
    y = [0] * 500 + [1] * 100 + [2] * 10
    return X, y


def cast_ser_into_arr(ser):
    """
    cast a pd Series into a 2-d arr of (len, 1)
    """
    return ser.values[:, np.newaxis]

def labels_arr_into_ser(labels_, ret_ser):
    """
    turns an array of labels into series, using the index of ret_ser
    """
    return pd.Series(labels_, index = ret_ser.index)
    

    
def check_arr_shape(arr, shape, action = "exception", string="cov mx"):
    """
    Check the shape of an object.
    if arr is indeed an array of the desired shape, return True.
    Otherwise you can return False, or raise an exception.
    
    Parameters:
    ------------------------
    arr:
    
    shape: tuple
        the desired shape
    action: string, default = "exception"
        - if 'exception', raise exception. 
        - if 'false', return False.
    string:
        used in the exception message.
    """
    if _is_arraylike_not_scalar(arr) and arr.shape == shape:
        return True
    # action
    if action == "exception":
        raise Exception(f"Unexpected shape of {string}!")
    elif action == "false":
        return False
    else:
        raise Exception("Unsupported type of action!")




def generate_2d_TPM(p, q):
    """
    generate a 2d transition proba mx. p, q are the probability of staying in state 0 & 1.
    """
    return np.array([[p, 1-p], [1-q, q]])
    




def permute_arr(arr):
    """
    given an array, return all the permutations of its labels.
    
    Parameters:
    ---------------------
    arr: array
    
    yield: array
    
    """
    labels, index = np.unique(arr, return_inverse=True)
    arr_permuted = np.empty_like(arr)
    range_len = range(len(labels))
    for perm in permutations(labels):
        for i in range_len:
            arr_permuted[index==i] = perm[i]
        yield arr_permuted.copy()
        
def classification_score_permutation(scorer, y_true, y_pred):
    """
    compute a classification score under all permutations of y_pred, and outputs the highest one.
    """
    scores = []
    for y_pred_perm in permute_arr(y_pred):
        scores.append(scorer(y_true, y_pred_perm))
    return max(scores)


def balanced_accuracy_score_cust(y_true, y_pred):
    """
    a customized function to compute balanced accuracy.
    """
    all_labels, counts = np.unique(y_true, return_counts=True)
    recall = np.empty(len(all_labels))
    for i, label in enumerate(all_labels):
        recall[i] = ((y_true == label) & (y_pred == label)).sum()/counts[i]
    return recall.mean()