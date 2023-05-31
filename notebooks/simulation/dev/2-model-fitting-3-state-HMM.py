#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path_repo = expanduser("~/Documents/G3_2/regime-identification"); sys.path.append(path_repo)
path_file = expanduser("/scratch/network/yizhans/G3_2/simulation")
path = {}
for folder in ["data", "estimation", "score", "figure", "latex"]:
    path[folder] = f"{path_file}/{folder}"

job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])


# In[2]:


from numpy.random import RandomState


# In[3]:


from regime.jump import *
from regime.simulation_helper import *


# # 2-model-fitting
# 
# In this notebook we fit the models on two data scenarios: 2-state Hardy parameter and t emission dist.

# In[4]:


n_t, n_buffer = 1024, 20
n_c = 3
batch_size = 32


# # HMM models

# In[5]:


random_state = RandomState(0)
model_hmm_10init = GaussianHMM_model(n_c, n_init=10, init='k-means++', random_state=random_state, n_iter = 300, tol = 1e-4, min_covar=1e-6, covars_prior=1e-6,
                                    transmat_prior = 1.+1e-5)#
model_dict_hmm = {"HMM10init": model_hmm_10init}
# model_hmm_1init = GaussianHMM_model(n_c, n_init=1, init='k-means', random_state=random_state, n_iter = 300, tol = 1e-4, min_covar=1e-6, covars_prior=1e-6)
# model_hmm_1initCovPrior = GaussianHMM_model(n_c, n_init=1, init='k-means', random_state=random_state, n_iter = 300, tol = 1e-4, min_covar=1e-6, covars_prior=1e-4)
# model_dict_hmm = {"true": model_true, "HMM10init": model_hmm_10init, "HMM1init": model_hmm_1init, "HMM1initCovPrior": model_hmm_1initCovPrior}


# In[6]:


key_feat_hmm = "HMM"
param_grid_hmm = None


# In[7]:


model_fit_many_datas_models(generate_key_data(3), key_feat_hmm, model_dict_hmm, param_grid_hmm, path, job_id, batch_size)


# In[ ]:




