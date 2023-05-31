#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path = "~/Documents/G3_2/regime-identification"
path = expanduser(path)
sys.path.append(path)

path_file = "/scratch/network/yizhans/G3_2/simulation"
path_data = f"{path_file}/data"
path_estimation = f"{path_file}/estimation"
path_score = f"{path_file}/score"


# In[2]:


import numpy as np
from sklearn.metrics import roc_auc_score

from numpy.random import RandomState
random_state = RandomState(0)


# In[3]:


from regime.simulation_helper import *


# # 0-generate-data-estimate-true-model
# 
# In this notebook we systematically generate the simulation data, estimate the labels and probability by the true HMM model, and score them. 

# # Original Jump paper
# ## Vanilla: 9 combinations
# 
# - scale: We use the parameters estimated in the classical Hardy's paper, and convert into three scales: **daily, weekly, monthly**, with decreasing persistency.
# - length: We simulate seqs of different length: 250, 500, 1000.
# 
# For each combo, we simulate `n_t=1000` seqs. The data in each combo are saved in a batch, thus in the shape of `(n_t, n_s, n_f)`. Also since we need to do feature engineering, every seq is 20 periods longer.

# In[4]:


scale_lst = ["daily", "weekly", "monthly"]
n_s_lst = [250, 500, 1000]
key_data_list = [f"{scale}_{n_s}" for scale in scale_lst for n_s in n_s_lst]

n_buffer, n_t, n_c = 20, 1024, 2


# In[5]:


for scale in scale_lst:
    # get a true HMM model
    hmm_true = get_GaussianHMM_model(*load_hardy_params(scale), random_state=random_state)
    for n_s in n_s_lst:
        # generate key for data
        key_data = f"{scale}_{n_s}"
        # simulate X_raw, Z.
        Xs, Zs = sample_from_hmm(hmm_true, n_trials=n_t, n_samples=n_s+n_buffer, random_state=random_state)
        Zs = Zs[:, -n_s:]
        # save raw data
        np_save_print(f"{path_data}/X_raw_{key_data}.npy", Xs, "X raw")
        np_save_print(f"{path_data}/Z_{key_data}.npy", Zs, "Z")        
        # estimate by the true HMM model.
        labels_arr, proba_arr = HMM_estimate_result(hmm_true, Xs[:, -n_s:])
        # save estimation results
        np_save_print(f"{path_estimation}/labels_{key_data}_true.npy", labels_arr, "labels")
        np_save_print(f"{path_estimation}/proba_{key_data}_true.npy", proba_arr, "proba")
        # score the estimation by the true model.
        acc_arr = scorer_batch(accuracy_each_cluster, Zs, labels_arr, )
        idx = get_idx_have_all_clusters(Zs, n_c)
        roc_auc_arr = scorer_batch(roc_auc_score, Zs, proba_arr[:, :, 1], idx_subset=idx)
        # save scores
        np_save_print(f"{path_score}/acc_{key_data}_true", acc_arr, "accuracy score")
        np_save_print(f"{path_score}/roc_auc_{key_data}_true", roc_auc_arr, "roc auc score")
        # print for sanity check
        print(f"{key_data} data. BAC: {np.nanmean(acc_arr, 0).mean()}, roc_auc: {np.nanmean(roc_auc_arr)}")


# In[ ]:




