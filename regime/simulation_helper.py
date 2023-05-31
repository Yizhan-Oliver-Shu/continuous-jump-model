import os
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import BaseHMM
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import check_random_state
from sklearn.model_selection import ParameterGrid
from itertools import product, permutations, chain
from tqdm import tqdm, trange
import time, math
from datetime import timedelta


from regime.stats import *
from regime.cluster_utils import *


########################################################
#    functions summary

############# data generation ##############

############# estimation  #############

############# scoring  #############



########################################################


##########################################
## save helpers
##########################################

def check_dir_exist(file_name):
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname): 
        os.makedirs(dirname)  
        print(f"created folder: {dirname}.")
    return 


def generate_file_name(path, folder, key_data, name, key_len=None, key_feat=None, key_model=None, job_id=None, suffix="npy"):
    file_name = f"{path[folder]}/{key_data}/{name}"
    if key_len is None:
        return file_name + "." + suffix
    file_name += f"_{key_len}"
    if key_feat is None:
        return file_name + "." + suffix
    file_name += f"_{key_feat}"
    if key_model is None:
        return file_name + "." + suffix
    file_name += f"_{key_model}"
    if job_id is None:
        return file_name + "." + suffix
    return file_name + f"_{job_id}" + "." + suffix

def save_file(arr, path, folder, key_data, name=None, key_len=None, key_feat=None, key_model=None, job_id=None, suffix="npy"):
    """
    save a file, or a dict of files.
    """
    if isinstance(arr, dict):
        for name_, arr_ in arr.items():
            save_file(arr_, path, folder, key_data, name_, key_len, key_feat, key_model, job_id, suffix)
        return
    file_name = generate_file_name(path, folder, key_data, name, key_len, key_feat, key_model, job_id, suffix)
    check_dir_exist(file_name)
    if suffix == "npy":
        np.save(file_name, arr)
        print(f"shape of the saved {name}: {arr.shape}.")
    elif suffix == "h5":
        arr.to_hdf(file_name, name, "w")
        print(f"save {name} to hdf: {arr.shape}")
    elif suffix == "csv":
        arr.to_csv(file_name)
        print(f"save {name} to csv: {arr.shape}")
    else: 
        raise NotImplementedError()
    return 

def load_file(path, folder, key_data, name, key_len=None, key_feat=None, key_model=None, job_id=None, suffix="npy", **kwargs):
    """
    load a file.
    """
    file_name = generate_file_name(path, folder, key_data, name, key_len, key_feat, key_model, job_id, suffix)
    if suffix == "npy":
        return np.load(file_name)
    elif suffix == "h5":
        return pd.read_hdf(file_name)
    elif suffix == "csv":
        return pd.read_csv(file_name, **kwargs)
    else: 
        raise NotImplementedError()

def print_seconds(x):
    x = math.ceil(x)
    return str(timedelta(seconds=x))


##############################
## data generation
##############################

def generate_key_data(data="all", **kwargs):
    """
    data can be a string of key, a list, or "all".
    if data == "t", needs a kwargs "dof", which can be either an int, or a list of ints.
    """
    all_keys = [2, 3, "t", "NB"]    #"Onat"
    scale_list = ["daily", "weekly", "monthly"]
    if isinstance(data, list):
        return list(chain.from_iterable([generate_key_data(key, **kwargs) for key in data]))
    if data == "all":
        data = all_keys
        return generate_key_data(all_keys, **kwargs)
    if data == 2 or data == 3:
        return [f"{data}-state-{scale}" for scale in scale_list]
    if data == "t":
        dof = kwargs["dof"]
        if isinstance(dof, list):
            return list(chain.from_iterable([generate_key_data("t", dof=dof_) for dof_ in dof]))
        return [f"t-{dof}-{scale}" for scale in scale_list]
    if data == 'Onat':
        return [f"Onat-{i}" for i in range(1, 3)]
    if data == "NB":
        return [f"NB-{scale}" for scale in scale_list]
    raise NotImplementedError()


def load_hardy_params(scale = "monthly", num_state = 2):
    """
    load the parameters of HMM from the classical Hardy paper. returns means_, covars_, transmat_.
    """
    if scale == "monthly":
        if num_state == 2:
            means_ = np.array([[.0123], [-.0157]])
            covars_ = np.array([[.0347], [.0778]])**2
            p, q = .9629, .7899
            transmat_ = generate_2d_TPM(p, q)   
            return means_, covars_, transmat_
        if num_state == 3:
            means_ = np.array([[.0123], [0.], [-.0157]])
            covars_ = np.array([[.0347], [.05], [.0778]])**2
            transmat_ = np.array([[.9629, .0185, .0186], [.0618, .8764, .0618], [.1051, .1050, .7899]])
            return means_, covars_, transmat_            
    if scale == "weekly":
        return scale_params(*load_hardy_params(num_state = num_state), 20, 5)
    if scale == "daily":
        return scale_params(*load_hardy_params(num_state = num_state), 20, 1)

def get_GaussianHMM_model(means_, covars_, transmat_, startprob_=None, random_state=None):
    """
    get a GaussianHMM model with the given params. The instance won't update any parameter if fit method is called. 
    """
    n_c, n_f = means_.shape
    random_state = check_random_state(random_state)
    
    if startprob_ is None:
        startprob_ = invariant_dist_transmat_(transmat_)
        
    hmm_model = GaussianHMM(n_components=n_c, random_state=random_state, params="", init_params="")
    hmm_model.n_features = n_f
    hmm_model.means_ = means_
    hmm_model.covars_ = covars_
    hmm_model.transmat_ = transmat_
    hmm_model.startprob_ = startprob_
    return hmm_model

def get_HMM_instance_for_sampling(means_, covars_, transmat_, startprob_=None, emission = "Gaussian", dof_ = None, random_state=None):
    n_c = len(transmat_)
    random_state = check_random_state(random_state)
    if startprob_ is None:
        startprob_ = invariant_dist_transmat_(transmat_)
    
    if emission == "Gaussian":
        model = GaussianHMM_1d_for_sample(n_c, random_state)
    elif emission == "t":
        model = tHMM_1d_for_sample(n_c, random_state)
    else:
        raise NotImplementedError()
    
    model.means_ = means_
    model.covars_ = covars_
    model.transmat_ = transmat_
    model.startprob_ = startprob_
    if emission == "t":
        model.dof_ = dof_
    return model

def sample_from_model(model, n_t, n_s):
    """
    generate a batch of sequences from a model, by calling the `sample` method of the model instance.
    """
    XZ_list = [model.sample(n_samples=n_s) for _ in range(n_t)] #(X:(n_s, n_f), Z:(n_s,))
    return np.array([XZ[0] for XZ in XZ_list]), np.array([XZ[1] for XZ in XZ_list]) # Xs: (n_t, n_s, n_f), Zs: (n_t, n_s)

def simulate_data(model, len_list, n_t, n_b, key_data=None, path=None):
    """
    model can be a dict, under which situation len_list is also a dict.
    """
    if isinstance(model, dict):
        for key_data_, model_ in tqdm(list(model.items())):
            simulate_data(model_, len_list[key_data_], n_t, n_b, key_data_, path)
        return 
    # model is a model instance
    for n_s in len_list:
        Xs, Zs = sample_from_model(model, n_t, n_s+2*n_b)
        save_file(Xs, path, "data", key_data, "Xs", n_s, "raw")
        Xs, Zs = Xs[:, n_b:-n_b], Zs[:, n_b:-n_b]
        save_file(Xs, path, "data", key_data, "Xs", n_s, "HMM")
        save_file(Zs, path, "data", key_data, "Zs", n_s)
    return 


def makedir(path, folder, key_data):
    """
    key_data can be a list.
    """
    if isinstance(key_data, list):
        for key_data_ in key_data: makedir(path, folder, key_data_)
        return 
    check_dir_exist(f"{path[folder]}/{key_data}/")
    return 


######################################
## HMM class only for fast sampling
######################################
class HMM_for_sample(BaseHMM):
    """
    A base class for HMM only used for fast sampling.
    means_, covars_ would be squeezed when being inputted. _check method does nothing.
    """
    def __init__(self, n_components, random_state):
        super().__init__(n_components=n_components, random_state=random_state,)
    
    @property
    def means_(self):
        return self._means_
    
    @means_.setter
    def means_(self, means_):
        self._means_ = means_.squeeze()
        
    @property
    def covars_(self):
        return self._covars_
    
    @covars_.setter
    def covars_(self, covars_):
        self._covars_ = covars_.squeeze()    
        
    def _check(self):
        return 
    
class tHMM_1d_for_sample(HMM_for_sample):
    """
    a class for 1d t-HMM that can sample very fastly, but can only do sampling.
    """
    def __init__(self, n_components, random_state):
        super().__init__(n_components, random_state)

    def _generate_sample_from_state(self, state, random_state):
        return [self._means_[state] + np.sqrt(self._covars_[state] * (self.dof_-2)/self.dof_) * random_state.standard_t(self.dof_)]
    
class GaussianHMM_1d_for_sample(HMM_for_sample):
    """
    a class for 1d GaussianHMM that can sample very fastly, but can only do sampling.
    """
    def __init__(self, n_components, random_state):
        super().__init__(n_components, random_state)

    def _generate_sample_from_state(self, state, random_state):
        return [self._means_[state] + np.sqrt(self._covars_[state]) * random_state.standard_normal()]

#################################
## HSMM
#################################

def compute_p_nbinom(p_geo, n_nbinom):
    return 1 / (1 + (1 - p_geo) / (p_geo * n_nbinom))

class GaussianHSMM_1d_for_sample(HMM_for_sample):
    """
    a class for 1d HSMM that can sample very fastly, but can only do sampling.
    """
    def __init__(self, n_components, random_state):
        super().__init__(n_components, random_state)
        if n_components != 2:
            raise NotImplementedError()

    def _compute_p_nbinom(self):
        self.p_nbinom_ = compute_p_nbinom(extract_off_diagonal(self.transmat_), self.n_shape_)
        
    def sample_Z(self, n_samples=1, currstate=None):
        random_state = check_random_state(self.random_state)
        #
        self._compute_p_nbinom()
        Z = np.full((n_samples,), -1, dtype=np.int32)
        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > random_state.rand()).argmax()
        length = 0
        while (length < n_samples):
            nbinom_sample = random_state.negative_binomial(self.n_shape_[currstate], self.p_nbinom_[currstate]) + 1
            Z[length:max(n_samples, length+nbinom_sample)] = currstate
            length += nbinom_sample
            currstate = 1-currstate
        return Z
    
    def sample(self, n_samples=1, currstate=None):
        random_state = check_random_state(self.random_state)
        Z = self.sample_Z(n_samples, currstate)
        X = np.full((n_samples, 1), 1000.)
        for i in np.unique(Z):
            X[Z==i, :] = self.means_[i] + np.sqrt(self._covars_[i]) * random_state.randn((Z==i).sum(), 1)
        return X, Z
    

    
def get_HSMM_instance_for_sampling(means_, covars_, transmat_, startprob_=None, n_shape_=None, random_state=None):
    n_c = len(transmat_)
    random_state = check_random_state(random_state)
    if startprob_ is None:
        startprob_ = invariant_dist_transmat_(transmat_)
    
    model = GaussianHSMM_1d_for_sample(n_c, random_state)
    
    model.means_ = means_
    model.covars_ = covars_
    model.transmat_ = transmat_
    model.startprob_ = startprob_
    model.n_shape_ = n_shape_
    return model




#################################
## Feature engineering
#################################

def filter_list_contains_key(lst, key):
    """
    key can be a list, where the name must contain all the elements in the key list.
    """
    def func(x):
        if isinstance(key, str):
            return key in x
        elif isinstance(key, list):
            for key_ in key: 
                if key_ not in x: return False
            return True
    return list(filter(func, lst))  

def filter_filenames_in_folder(folder, key):
    """
    key can be a list
    """
    filenames = os.listdir(folder)
    return filter_list_contains_key(filenames, key)

def raise_str_to_list(x):
    if isinstance(x, str): return [x]
    return x

def feature_engineer(key_feat, key_data, n_b, path, n_s_list = None):
    """
    key_data can be a list.
    
    n_s_list, or subsetting the dataset, can only be used if not in batch.
    """
    if isinstance(key_feat, list) or isinstance(key_data, list):
        key_feat, key_data = raise_str_to_list(key_feat), raise_str_to_list(key_data)
        for key_feat_, key_data_ in tqdm(list(product(key_feat, key_data))):
            feature_engineer(key_feat_, key_data_, n_b, path)
        return 

    function_dict = {"zhengB": lambda x: feature_engineer_zheng_batch(x, True),
                    "zhengF": lambda x: feature_engineer_zheng_batch(x, False),
                    "ewm": feature_engineer_ewm_batch}
    if key_feat not in function_dict.keys():
        raise NotImplementedError("feature not supported yet") 
    
    # key_feat, key_data are single key
    folder = f"{path['data']}/{key_data}"
    if n_s_list is not None:
        filenames = [f"Xs_{n_s}_raw.npy" for n_s in n_s_list]
    else:
        filenames = filter_filenames_in_folder(folder, "raw")
    for filename in filenames:
        Xs_raw = np.load(f"{folder}/{filename}")
        Xs_feat = function_dict[key_feat](Xs_raw)[:, n_b:-n_b]
        # save results
        save_file(Xs_feat, path, 'data', key_data, "Xs", int(filename.split('_')[1]), key_feat)         
    return 



############################
## estimation
############################

def extract_off_diagonal(arr):
    """
    for a 2 by 2 mx or a batch of them, flatten and extracts all the off-diagonal elements.
    
    Parameters:
    ----------------------------
    arr: array (n_c, n_c) or (n_t, n_c, n_c).
    """
    if arr.ndim == 2:
        n_c = len(arr)
        return arr.flatten()[np.arange(n_c**2) % (n_c+1) != 0]
    elif arr.ndim == 3:
        n_t, n_c, _ = arr.shape
        arr_copy = arr.reshape((n_t, -1))
        return arr_copy[:, np.arange(n_c**2) % (n_c+1) != 0]
    
def combine_model_param_estimation(means_arr, covars_arr, transmat_arr):
    """
    combine all the model parameters. can be in a batch.
    
    Parameters:
    ----------------------------
    means_arr: array (n_c) or (n_t, n_c).   
    
    covars_arr: array (n_c) or (n_t, n_c). 
    
    transmat_arr: array (n_c, n_c) or (n_t, n_c, n_c).
    """
    transmat_off = extract_off_diagonal(transmat_arr)
    return np.concatenate((means_arr, np.sqrt(np.maximum(covars_arr, 1e-6)), transmat_off), axis=-1)


# our models
def model_fit_batch(model, Xs, Zs, align=True):
    """
    fit a model on a batch of data. save model parameters, and fitted labels & proba.
    need the true labels to do alignments (i.e. the permutation with the highest overall accuracy)

        
    Paramaters:
    ---------------------
    model: a model instance.
    
    Xs: array of size (n_t, n_s, n_f)
        input data
        
    Zs: array (n_t, n_s)
        true labels
        
    Returns:
    -------------------------
    model_params_arr: (n_t, n_c**2 + n_c)
    
    labels_arr: array of size (n_t, n_s)
    
    proba_arr: array of size (n_t, n_s, n_c)
    """
    n_t, n_s, _ = Xs.shape
    n_c = model.n_components

    res_list = []
    # estimate
    # for i_t in trange(n_t):
    for i_t in range(n_t):
        X, Z = Xs[i_t], Zs[i_t]
        # fit
        model.fit(X)
        # save dict result
        res_list.append(extract_results_from_model(model, X_=X[:, 0]))
    # dict of results
    res = combine_list_dict(res_list)
    # align with true labels
    if align:
        res = align_estimation_results_batch(Zs, res)
    # combine model params
    model_params_arr = combine_model_param_estimation(res["means_"], res["covars_"], res["transmat_"])
    labels_arr = res["labels_"]; proba_arr = res["proba_"]
    return model_params_arr, labels_arr, proba_arr
    
# helpers
def weighted_mean_vol(X, proba_):
    n_c = proba_.shape[1]
    means_, covars_ = np.full(n_c, np.nan), np.full(n_c, np.nan)
    total_weight = proba_.sum(0)
    idx = (total_weight>0)
    weighted_sum = X @ proba_
    means_[idx] = weighted_sum[idx] / total_weight[idx]
    weighted_sum_square = ((X[:, np.newaxis] - means_[np.newaxis, :])**2 * proba_).sum(0)
    covars_[idx] = weighted_sum_square[idx] / total_weight[idx]
    return means_, covars_

def raise_labels_to_proba_(labels_, n_c):
    """
    raise one labels_ into a proba_
    labels_: (n_s,)
    """
    n_s = len(labels_)
    proba_ = np.zeros((n_s, n_c))
    np.put_along_axis(proba_, indices=labels_[..., np.newaxis], values=1., axis=-1)
    return proba_  

def raise_labels_to_proba_batch(labels_arr, n_c):
    """
    labels_arr: (n_t, n_s)
    """
    n_t, n_s = labels_arr.shape
    proba_arr = np.zeros((n_t, n_s, n_c))
    np.put_along_axis(proba_arr, indices=labels_arr[..., np.newaxis], values=1., axis=-1)
    return proba_arr    

def extract_results_from_model(model, X_=None):
    """
    extract the estimation results from one model.
    The 1d sequence X is needed to compute the weighted means, covars.
    
    Parameters:
    ----------------------------------------
    model:
    
    X_: array (n_s,). default None.
    
    Returns:
    ---------------------------------------
    result: dict
    """
    n_c = model.n_components
    result = {}
    # proba
    if hasattr(model, "proba_"):
        result["proba_"] = model.proba_
    else:
        result["proba_"] = raise_labels_to_proba_(model.labels_, n_c)
        
    # label
    if hasattr(model, "labels_"):
        result["labels_"] = model.labels_
    else:
        result["labels_"] = model.proba_.argmax(axis=-1).astype(np.int32)
        
    # means covars
    if hasattr(model, "means_"):
        result["means_"] = model.means_
        result["covars_"] = model.covars_
    else:
        # compute weighted average by proba_
        result["means_"], result["covars_"] = weighted_mean_vol(X_, result["proba_"])
        
    # transmat
    if hasattr(model, "transmat_"):
        result["transmat_"] = model.transmat_
    else:
        # empirical
        result["transmat_"] = empirical_trans_mx(result["labels_"], n_c)
    return result

def combine_list_dict(dict_list):
    """
    input is a list of dictionaries, all with the same keys.
    return a dict with the same keys, value is the stacked array.
    """
    keys = dict_list[0].keys()
    res = {key: np.array([dict_[key] for dict_ in dict_list]) for key in keys}
    return res  



def model_fit_batch_with_params(model, Xs, Zs, param_grid=None, align=True):
    """
    fit a model on a batch of data. save model parameters, and fitted labels & proba.
    the model can have a param_grid for hyperparam tuning.
    need the true labels to do alignments (i.e. the permutation with the highest overall accuracy)

        
    Paramaters:
    ---------------------
    model: a model instance.
    
    Xs: array of size (n_t, n_s, n_f)
        input data
        
    Zs: array (n_t, n_s)
        true labels
        
    param_grid: dict, default None
        if None, will call `model_fit_batch` directly.
        
    Returns:
    -------------------------
    model_params_arr: (n_t, n_c**2 + n_c, n_l)
    
    labels_arr: array of size (n_t, n_s, n_l)
    
    proba_arr: array of size (n_t, n_s, n_c, n_l)
    """
    if param_grid is None: # no hyperparams
        return model_fit_batch(model, Xs, Zs, align)
    
    PG = ParameterGrid(param_grid)
    model_params_arr_list, labels_arr_list, proba_arr_list = [], [], []
    for param_ in tqdm(PG):
        model.set_params(**param_)
        model_params_arr, labels_arr, proba_arr = model_fit_batch(model, Xs, Zs, align)
        model_params_arr_list.append(model_params_arr)
        labels_arr_list.append(labels_arr)
        proba_arr_list.append(proba_arr)
    return np.stack(model_params_arr_list, axis=-1), np.stack(labels_arr_list, axis=-1), np.stack(proba_arr_list, axis=-1)

def model_fit_many_datas_models(key_data_list, key_feat_list, model_dict, param_grid, path, job_id, batch_size, n_s_list=None, align=True):
    """
    train a collection of models, w/ hyperparams to tune, on a batch of data from many datasets.
    can specify the seq length to fit, or fit all in the folder.
    """
    key_data_list, key_feat_list = raise_str_to_list(key_data_list), raise_str_to_list(key_feat_list)
    #
    start = job_id * batch_size; end = start + batch_size
    N_combos = len(key_data_list) * len(key_feat_list) * len(model_dict)
    count = 0; time_old = time.time(); total_time=0.
    for key_data, key_feat in product(key_data_list, key_feat_list):
        folder = f"{path['data']}/{key_data}"
        if n_s_list is not None:
            filenames = [f"Xs_{n_s}_{key_feat}.npy" for n_s in n_s_list]
        else:
            filenames = filter_filenames_in_folder(folder, key_feat)
        for key_model, model in model_dict.items():
            count+=1; print(f"{count}/{N_combos} combo starts.")
            for filename in filenames:
                Xs = np.load(f"{folder}/{filename}")[start:end]
                n_s = int(filename.split('_')[1]); Zs = load_file(path, "data", key_data, "Zs", n_s)[start:end]
                # train the model, on a param grid, on a batch of data
                model_params_arr, labels_arr, proba_arr = model_fit_batch_with_params(model, Xs, Zs, param_grid, align)
                # save results
                save_file({"modelParams": model_params_arr,
                          "labels": labels_arr,
                          "proba": proba_arr}, 
                          path, "estimation", key_data, None, n_s, key_feat, key_model, job_id)
            time_now = time.time(); time_this_iter = time_now-time_old; total_time += time_this_iter; time_old = time_now
            print(f"{count}/{N_combos} combo done. Time of this combo: {print_seconds(time_this_iter)}s. Total time: {print_seconds(total_time)}s.")
    return 


def generate_param_grid():
    # lambd_list = 10 ** np.concatenate(([-2.], np.linspace(-1, 5, 13), [6.]))  #[-2.] + list(np.linspace(-1, 5, 13)) + [6.]
    lambd_list = 10 ** np.concatenate(([-2., -1, 0], np.linspace(1, 4, 10), [5., 6]))
    print(lambd_list)
    param_grid = {'jump_penalty': lambd_list}
    return param_grid

####################################################
## scores
####################################################

def accuracy_each_cluster(y_true, y_pred, n_c):
    """
    compute the accuracy under each cluster. if the cluster doesn't appear in the true data, return nan.
    """
    clusters, counts = np.unique(y_true, return_counts=True)
    recall = np.full(n_c, np.nan) #np.empty(n_c) * np.nan
    for i, cluster in enumerate(clusters):
        recall[cluster] = ((y_true == cluster) & (y_pred == cluster)).sum()/counts[i]
    return recall

def scorer_batch(scorer, Zs_true, Zs_pred, *args, has_params = False, idx = None, **kwargs):
    """
    compute scores for a batch of seqs, with possibly some hyparameters.
    
    Parameters:
    --------------------------
    - scorer:
    
    - Zs_true: array of size (n_t, n_s)
    
    - Zs_pred: array of size (n_t, n_s), or (n_t, n_s, n_l) if there is hyperparams.
        
    - has_params: 
        
    - idx: 
        indices of the trials to score. None means no need to subset. If true, select the trials where all of the classes appear in the true data.
    
    Returns:
    --------------------------
    - scores_arr: array of size (n_t, ), (n_t, n_c), or a dim in the last axis representing hyperparams. 
    """
    if idx is None: # no need to sub-index
        if not has_params:  # no hyperparam
            return np.array([scorer(Z_true, Z_pred, *args, **kwargs) for Z_true, Z_pred in zip(Zs_true, Zs_pred)])
        else: # has hyperparams
            n_l = Zs_pred.shape[-1]
            return np.stack([scorer_batch(scorer, Zs_true, Zs_pred[..., i_param], *args, **kwargs) for i_param in range(n_l)], axis=-1)
    # need to subset
    n_t = len(Zs_true)
    if idx is True:
        n_c = len(np.unique(Zs_true))
        idx = get_idx_have_all_clusters(Zs_true, n_c)
    scores_arr_subset = scorer_batch(scorer, Zs_true[idx], Zs_pred[idx], *args, has_params = has_params, **kwargs)
    scores_arr = np.full((n_t,) + scores_arr_subset.shape[1:], np.nan)
    scores_arr[idx] = scores_arr_subset
    return scores_arr


############################
## score helpers
############################

def count_sample_in_each_cluster(Zs, n_c):
    """
    count the number of samples in each cluster for all the trials.
    """
    n_t = len(Zs)
    counts_arr = np.zeros((n_t, n_c), dtype=int)
    for i in range(n_t):
        elements, counts = np.unique(Zs[i], return_counts=True)
        # print(elements, counts)
        counts_arr[i, elements] = counts
    return counts_arr


def get_idx_have_all_clusters(Zs, n_c):
    """
    given a batch of true labels, return the indices for the trials where all the clusters appear in this simulated seq.
    """
    state_counts = count_sample_in_each_cluster(Zs, n_c)
    return (state_counts > 0).all(1) 


#################################
## align labels
#################################

def generate_all_perms_as_arr(n_c):
    return np.array(list(permutations(range(n_c))))

def permute_labels(labels_arr, all_perms):
    """
    return the labels under every permutation. new axis to the last
    """
    n_c = len(all_perms[0])
    labels_all_perms = np.zeros(labels_arr.shape + (len(all_perms),), dtype=np.int32)
    labels_all_perms[..., 0] = labels_arr
    for i_perm, perm in enumerate(all_perms[1:]):
        labels_permuted = labels_all_perms[..., i_perm+1]
        for i_cluster in range(n_c): # permute
            labels_permuted[labels_arr==perm[i_cluster]] = i_cluster
            # labels_permuted[labels_arr==i_cluster] = perm[i_cluster] # labels_permuted[labels_==i]
    return labels_all_perms


def align_estimation_results_batch(Zs_true, res):
    """
    align a batch of estimation results with the true labels, i.e. find the optimal permutation for each sample.
    results include labels_, proba_, means_, covars_, transmat_.
    """
    # n_c, n_t = len(np.unique(Zs_true)), len(Zs_true)
    n_c, n_t = res["proba_"].shape[2], len(Zs_true)
    # all the perms
    all_perms = generate_all_perms_as_arr(n_c) 
    # all the possible perms of labels
    labels_all_perms = permute_labels(res["labels_"], all_perms)
    # score accuracy for each perm
    acc_all_perms = scorer_batch(accuracy_score, Zs_true, labels_all_perms, has_params=True) # of shape (n_t, n_p)
    # best perm for each trial 
    best_perm_idx = acc_all_perms.argmax(-1) # shape (n_t,)
    best_perm = all_perms[best_perm_idx] # (n_t, n_c)
    # take the corresponding perm for labels
    res["labels_"] = np.take_along_axis(labels_all_perms, best_perm_idx[:, np.newaxis, np.newaxis], axis=-1).squeeze(axis=-1)
    res["proba_"] = np.take_along_axis(res["proba_"], best_perm[:, np.newaxis, :], -1)
    res["means_"] = np.take_along_axis(res["means_"], best_perm, -1)
    res["covars_"] = np.take_along_axis(res["covars_"], best_perm, -1)
    res["transmat_"] = np.take_along_axis(res["transmat_"][np.arange(n_t)[:, np.newaxis], best_perm], best_perm[:, np.newaxis, :], -1)
    return res


#################################
## score estimations
#################################

def generate_off_diagonal_idx(n_c):
    res = []
    for i, j in product(range(1, n_c+1), repeat=2):
        if i != j:
            res.append(f"{i}{j}")
    return res

def generate_modelParams_index(n_c):
    index=[]
    index += [f"$\mu_{i}$" for i in range(1, n_c+1)]
    index += [f"$\sigma_{i}$" for i in range(1, n_c+1)]
    index += [f"$\gamma_{{{ij}}}$" for ij in generate_off_diagonal_idx(n_c)]
    return index


def compute_BAC_std_from_acc_arr(acc_arr):
    """
    compute the mean/std of accuracy per class and balanced accuracy.
    
    Parameters:
    ----------------------------------
    acc_arr: size (n_s, n_c)
        the accuracy of k-th class in n-th trial.
        
    Returns:
    ----------------------------------
    acc_mean,
    
    acc_std
    """
    if acc_arr.ndim == 2:  # (n_t, n_c)
        n_c = acc_arr.shape[1]
        acc_cov = pd.DataFrame(acc_arr).cov().to_numpy()
        vec = np.repeat(1/n_c, n_c)
        quad_form = vec @ (acc_cov @ vec)
        return 0 if quad_form<=0 else np.sqrt(quad_form)   #vec @ (acc_cov @ vec) #np.sqrt(vec @ (acc_cov @ vec))
    # ndim == 3
    n_l = acc_arr.shape[-1]
    return np.array([compute_BAC_std_from_acc_arr(acc_arr[..., i_l]) for i_l in range(n_l)])

def generate_summary_df(model_params_arr, scoring_results, param_grid=None):
    if model_params_arr.ndim == 2: # no params
        model_params_arr = model_params_arr[..., np.newaxis]
        for key in scoring_results:
            scoring_results[key] = scoring_results[key][..., np.newaxis]
    means_dict, stds_dict = {}, {}
    #
    n_c = scoring_results["acc"].shape[1]; modelParams_index = generate_modelParams_index(n_c); acc_index = [f"Accuracy {i}" for i in range(1, n_c+1)]
    # model params
    means_dict["model_params"] = pd.DataFrame(np.nanmean(model_params_arr, axis=0), index=modelParams_index)
    stds_dict["model_params"] = pd.DataFrame(np.nanstd(model_params_arr, axis=0), index=modelParams_index)
    # acc
    means_dict["acc"] = pd.DataFrame(np.nanmean(scoring_results["acc"], axis=0), index=acc_index)
    stds_dict["acc"] = pd.DataFrame(np.nanstd(scoring_results["acc"], axis=0), index=acc_index)
    # BAC
    means_dict["BAC"] = pd.DataFrame(means_dict["acc"].mean(0), columns=["BAC"]).T
    stds_dict["BAC"] = pd.DataFrame(compute_BAC_std_from_acc_arr(scoring_results["acc"]), columns=["BAC"]).T
    for name, score_arr in scoring_results.items():
        if name == "acc":
            continue
        means_dict[name] = pd.DataFrame(np.nanmean(score_arr, axis=0), columns=[name]).T
        stds_dict[name] = pd.DataFrame(np.nanstd(score_arr, axis=0), columns=[name]).T
    def combine_dict_to_df(dictionary):
        return pd.concat(dictionary.values(), axis=0)
    means_df, stds_df = combine_dict_to_df(means_dict).T, combine_dict_to_df(stds_dict).T
    set_to_zero = lambda x: 0 if np.isclose(x, 0, atol=1e-10) else x
    four_digits = lambda x: f"{x:.4f}"
    stds_df = stds_df.applymap(set_to_zero)
    return means_df, stds_df, combine_means_std_df(means_df.applymap(four_digits), stds_df.applymap(four_digits), param_grid)

def combine_means_std_df(means_df, stds_df, param_grid = None):
    index, columns = means_df.index, means_df.columns
    df = pd.DataFrame(index=index, columns=columns)
    for idx, col in product(index, columns):
        df.loc[idx, col] = f"{means_df.loc[idx, col]}({stds_df.loc[idx, col]})"
    if param_grid is not None:
        def sci_notation_formatter(x):
            return f"{x:.1e}"
        df.insert(0, "params", [sci_notation_formatter(y) for y in param_grid["jump_penalty"]])
    return df


def load_arr_distributed(path, folder, key_data, name, key_len, key_feat, key_model, number_of_batch, batch_size):
    res = [load_file(path, folder, key_data, name, key_len, key_feat, key_model, job_id)[:batch_size] for job_id in range(number_of_batch)]
    return np.concatenate(res, axis=0)

def scoring_labels_proba_(Zs_true, labels_arr, proba_arr):
    has_params = labels_arr.ndim==3
    # n_c = len(np.unique(Zs_true))
    n_c = proba_arr.shape[2]
    scoring_res = {}
    scoring_res["acc"] = scorer_batch(accuracy_each_cluster, Zs_true, labels_arr, n_c, has_params=has_params)
    if n_c == 2:
        scoring_res["ROC-AUC"] = scorer_batch(roc_auc_score, Zs_true, proba_arr[:, :, 1], has_params=has_params, idx=True)
    else:
        scoring_res["ROC-AUC"] = scorer_batch(roc_auc_score, Zs_true, proba_arr, has_params=has_params, idx=True, average="macro", multi_class="ovo")
    return scoring_res

def score_and_summary_model(path, key_data, key_len, key_feat, key_model, num_of_batch, batch_size, param_grid=None):
    """
    score and summary one model estimation results, from distributed locations.
    """
    # load true labels
    Zs = load_file(path, "data", key_data, "Zs", key_len)[:num_of_batch * batch_size]
    # load estimation results
    estimation_res = {}
    for name in ["modelParams", "labels", "proba"]:
        estimation_res[name] = load_arr_distributed(path, "estimation", key_data, name, key_len, key_feat, key_model, num_of_batch, batch_size)
    # scoring classification accuracy
    scoring_results = scoring_labels_proba_(Zs, estimation_res["labels"], estimation_res["proba"])
    # save scores
    save_file(scoring_results, path, "score", key_data, key_len=key_len, key_feat=key_feat, key_model=key_model)
    # combine model params estimation with accuracy scores
    means_df, stds_df, summary_df = generate_summary_df(estimation_res["modelParams"], scoring_results, param_grid)
    return means_df, stds_df, summary_df


# def score_and_summary_many_models(path, key_data, batch_size, num_of_batch, param_grid=None, model_name=None):
#     """
#     can specify one model_name.
#     key_data can be a list
#     """
#     if isinstance(key_data, list):
#         for key_data_ in key_data: score_and_summary_many_models(path, key_data_, batch_size, num_of_batch, param_grid, model_name)
#         return 
#     key_list = ["labels_", "_0.npy"]
#     folder = f"{path['estimation']}/{key_data}"
#     if model_name is not None:
#         model_name = raise_str_to_list(model_name)
#         filenames = list(chain.from_iterable([filter_filenames_in_folder(folder, key_list + [f"_{model_name_}_"]) for model_name_ in model_name]))
#     else:
#         filenames = filter_filenames_in_folder(folder, key_list)
#     for filename in filenames:
#         _, n_s_str, key_feat, key_model, _ = filename.split("_"); n_s = int(n_s_str)
#         res_df = {}
#         res_df["means"], res_df["stds"], res_df["summary"] = score_and_summary_model(path, key_data, n_s, key_feat, key_model, 
#                                                                                      num_of_batch, batch_size, None if key_feat == "HMM" else param_grid)
#         save_file(res_df, path, "summary", key_data, key_len=n_s, key_feat=key_feat, key_model=key_model, suffix="h5")
#     return 

def score_and_summary_many_models(path, key_data, batch_size, num_of_batch, param_grid=None, model_name=None, n_s_list=None):
    """
    can specify one model_name.
    key_data can be a list
    """
    if isinstance(key_data, list):
        for key_data_ in key_data: score_and_summary_many_models(path, key_data_, batch_size, num_of_batch, param_grid, model_name, n_s_list)
        return 
    key_list = ["labels_", "_0.npy"]
    folder = f"{path['estimation']}/{key_data}"
    # filter model names
    if model_name is not None:
        model_name = raise_str_to_list(model_name)
        filenames = list(chain.from_iterable([filter_filenames_in_folder(folder, key_list + [f"_{model_name_}_"]) for model_name_ in model_name]))
    else:
        filenames = filter_filenames_in_folder(folder, key_list)
    # filter length
    if n_s_list is not None:
        filenames = [filename for filename in filenames if int(filename.split("_")[1]) in n_s_list]
    for filename in filenames:
        _, n_s_str, key_feat, key_model, _ = filename.split("_"); n_s = int(n_s_str)
        res_df = {}
        res_df["means"], res_df["stds"], res_df["summary"] = score_and_summary_model(path, key_data, n_s, key_feat, key_model, 
                                                                                     num_of_batch, batch_size, None if key_feat == "HMM" else param_grid)
        save_file(res_df, path, "summary", key_data, key_len=n_s, key_feat=key_feat, key_model=key_model, suffix="h5")
    return 

###############################################
## generate table-latex
##############################################


def add_cluster(string, cluster):
    return string if not cluster else f"{string}-cluster"


def col_is_acc(x):
    return "Accuracy" in x or x == "BAC" or x=="ROC-AUC"

def combine_summary_means_std_df(means_combined_df, stds_combined_df):
    df = {}
    for model in means_combined_df.index:
        df[model] = means_combined_df.loc[model]
        df[model+'-std'] = stds_combined_df.loc[model]
    df = pd.DataFrame(df)
    df.columns=["" if "std" in x else x for x in df.columns]
    return df.T

def generate_numerical_results_table(path, key_data, n_s, feat_dict, name_dict, cluster=False):
    """
    n_s can be a list.
    """
    if isinstance(n_s, list):
        # return {n_s_: generate_numerical_results_table(path, key_data, n_s_, feat_dict, name_dict, cluster) for n_s_ in n_s}
        return pd.concat([generate_numerical_results_table(path, key_data, n_s_, feat_dict, name_dict, cluster) for n_s_ in n_s])
    # load best_idx
    best_idx_str, summary_str = add_cluster("best-idx", cluster), add_cluster("summary", cluster)
    idx_df = load_file(path, best_idx_str, key_data, "best_idx", suffix="csv", index_col=0)
    for model_name in name_dict:
        if "true" in model_name or "HMM" in model_name: idx_df[model_name] = 0
    means_combined_df = {new_name: load_file(path, summary_str, key_data, "means", n_s, feat_dict[model], model, suffix="h5").iloc[idx_df.loc[n_s, model]] \
                         for model, new_name in name_dict.items()}
    stds_combined_df = {new_name: load_file(path, summary_str, key_data, "stds", n_s, feat_dict[model], model, suffix="h5").iloc[idx_df.loc[n_s, model]] \
                         for model, new_name in name_dict.items()}
    means_combined_df, stds_combined_df = pd.DataFrame(means_combined_df).T, pd.DataFrame(stds_combined_df).T
    # accuracy of estimation
    df_acc_for_rank = {col: (means_combined_df[col]-means_combined_df.loc["true", col]).abs() if not col_is_acc(col) else -means_combined_df[col] for col in means_combined_df.columns}
    df_acc_for_rank = pd.DataFrame(df_acc_for_rank).drop("true")
    best_model_ser = df_acc_for_rank.idxmin()
    means_combined_df, stds_combined_df = means_combined_df.applymap(lambda x: f"{x:.4f}"), stds_combined_df.applymap(lambda x: f"{x:.4f}")
    for col in best_model_ser.index:
        means_combined_df.loc[best_model_ser[col], col] = "\\textbf{" + means_combined_df.loc[best_model_ser[col], col] + "}"
    # combine
    df_ret = combine_means_std_df(means_combined_df, stds_combined_df)
    df_ret = pd.concat([pd.DataFrame("", index=[n_s], columns=df_ret.columns), df_ret])
    return df_ret
#combine_summary_means_std_df(means_combined_df, stds_combined_df) #means_combined_df, stds_combined_df, df_acc_for_rank

#combine_summary_means_std_df(means_combined_df, stds_combined_df) #means_combined_df, stds_combined_df, df_acc_for_rank

def output_column_format(df_ret, space):
    n_col=df_ret.shape[1]
    return ("l @{\hspace{" + space + "}} ") * n_col + "l"

import re
def print_table_to_latex(df, col_space, label, font=(6,8)):
    def output_column_format(df_ret, space):
        n_col=df_ret.shape[1]
        return ("l @{\hspace{" + space + "}} ") * n_col + "l"
    n_col = df.shape[1]
    if n_col > 20:
        raise NotImplementedError()
    if n_col > 10:
        if n_col % 2 == 1:
            df[""] = ""; n_col += 1
        string1, string2 = print_table_to_latex(df.iloc[:, :n_col//2], col_space, label, font), print_table_to_latex(df.iloc[:, n_col//2:], col_space, label, font)
        string = string1+string2
        return re.sub(r"\\bottomrule.*?\\toprule", r"\\bottomrule", string, flags=re.DOTALL)
        
    string = df.style.to_latex(column_format=output_column_format(df, col_space),
            position = "htbp", hrules=True) #, position_float="centering",  caption=label,, label=label
    
    key = "\\begin{tabular}" #"\label{" + label + "}"
    string = string.replace(key, "\n\\begin{adjustwidth}{-10cm}{-10cm} \n\\centering \n{\\fontsize" + \
                            "{" + str(font[0]) + "}" + "{" + str(font[1]) + "}" + "\\selectfont\n" + key)
    
    string = string.replace("\\end{tabular}", "\\end{tabular} \n}\\end{adjustwidth}\n\\caption{"+str(label)+"}\n\\label{"+str(label)+"}")
    return string



# def print_table_to_latex(df, col_space, label, font=(6,8)):
#     def output_column_format(df_ret, space):
#         n_col=df_ret.shape[1]
#         return ("l @{\hspace{" + space + "}} ") * n_col + "l"
#     n_col = df.shape[1]
#     if n_col > 20:
#         raise NotImplementedError()
#     if n_col > 10:
#         if n_col % 2 == 1:
#             df[""] = ""; n_col += 1
#         string1, string2 = print_table_to_latex(df.iloc[:, :n_col//2], col_space, label, font), print_table_to_latex(df.iloc[:, n_col//2:], col_space, label, font)
#         string = string1+string2
#         return re.sub(r"\\bottomrule.*?\\toprule", r"\\bottomrule", string, flags=re.DOTALL)
        
#     string = df.style.to_latex(column_format=output_column_format(df, col_space),
#             position = "htbp", hrules=True, caption=label, label=label) #, position_float="centering"
#     key = "\label{" + label + "}"
#     string = string.replace(key, key + "\n\\begin{adjustwidth}{-10cm}{-10cm} \n\\centering \n{\\fontsize" + \
#                             "{" + str(font[0]) + "}" + "{" + str(font[1]) + "}" + "\\selectfont\n")
#     string = string.replace("\\end{tabular}", "\\end{tabular} \n}\\end{adjustwidth}\n")
#     return string


def save_results_in_latex(path, key_data, n_s_list, feat_dict, name_dict, latex_file_name, cluster=False):
    """
    only do it for one key_data.
    """
    latex_str = add_cluster("latex", cluster)
    df = generate_numerical_results_table(path, key_data, n_s_list, feat_dict, name_dict, cluster)
    string = print_table_to_latex(df, ".2cm", key_data)
    filename = generate_file_name(path, latex_str, key_data, latex_file_name, suffix="latex"); check_dir_exist(filename)
    with open(filename, "w") as f:
        f.write(string)
    return string




###################################################

class Viterbi_wrapper():
    def __init__(self, means_, covars_, transmat_, ):
        self.hmm_instance = get_GaussianHMM_model(means_, covars_, transmat_)
        self.means_ = means_.squeeze()
        self.covars_ = covars_.squeeze()
        self.transmat_ = transmat_
        self.n_components = self.hmm_instance.n_components
        
    def fit(self, X):
        self.labels_ = self.hmm_instance.predict(X)
        self.proba_ = self.hmm_instance.predict_proba(X)