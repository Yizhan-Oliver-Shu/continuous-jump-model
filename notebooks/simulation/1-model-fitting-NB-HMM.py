#!/usr/bin/env python
# coding: utf-8

# In[1]:


## timelimit: 00:16:00
import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path_repo = expanduser("~/Documents/G3_2/regime-identification"); sys.path.append(path_repo)
path_file = expanduser("/scratch/network/yizhans/G3_2/simulation")
path = {}
for folder in ["data", "estimation"]:
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


n_c = 2
batch_size = 32


# # True

# In[5]:


for scale in ["daily", "weekly", "monthly"]:
    key_data_list_true = f"NB-{scale}"
    model_dict_true = {"true": Viterbi_wrapper(*load_hardy_params(scale, n_c))}
    model_fit_many_datas_models(key_data_list_true, "HMM", model_dict_true, None, path, job_id, batch_size, align=False)


# # HMM models

# In[6]:


# HMM models

random_state = RandomState(0)

model_hmm = GaussianHMM_model(n_c, 1, "k-means", random_state=random_state, covariance_type='full', 
                          min_covar=1e-6, covars_prior=0., n_iter=300, tol=1e-6, transmat_prior = 1.+1e-8)
model_hmm10init = GaussianHMM_model(n_c, 10, "k-means++", random_state=random_state, covariance_type='full', 
                          min_covar=1e-6, covars_prior=0., n_iter=300, tol=1e-6, transmat_prior = 1.+1e-8)


model_dict_hmm = {"HMM-paper": model_hmm,
                  "HMM10init-paper": model_hmm10init
                 }


# In[7]:


key_feat_hmm = "HMM"
param_grid_hmm = None


# In[8]:


model_fit_many_datas_models(generate_key_data("NB"), key_feat_hmm, model_dict_hmm, param_grid_hmm, path, job_id, batch_size)


# In[ ]:




