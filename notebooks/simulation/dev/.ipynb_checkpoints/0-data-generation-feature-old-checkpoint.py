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


# In[2]:


import numpy as np
from numpy.random import RandomState


# In[3]:


from regime.simulation_helper import *


# In[4]:


n_t, n_buffer = 1024, 20
len_list = [[250, 500, 1000, 2000], [50, 100, 250, 500, 1000], [60, 120, 250, 500]]


# # 0-generate-data
# 
# In this notebook we systematically generate the simulation data. We postpone estimation using the true model to a later notebook, to put estimation all in one place.
# 

# # 2-state models
# 
# - scale: We use the parameters estimated in the classical Hardy's paper, and convert into three scales: **daily, weekly, monthly**, with decreasing persistency.
# - length: We simulate seqs of different length.
# 
# For each combo, we simulate `n_t=1024` seqs. The data in each combo are saved in a batch, thus in the shape of `(n_t, n_s, n_f)`. Also since we need to do feature engineering, every seq is 20 periods longer at both the beginning and the end.

# In[5]:


n_c = 2
random_state = RandomState(0)
key_data_list_2_state = generate_key_data(n_c)
DGP_dict_2_state = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="Gaussian", random_state=random_state) for key_data in key_data_list_2_state}
n_s_dict_2_state = dict(zip(key_data_list_2_state, len_list))


# In[6]:


simulate_data(DGP_dict_2_state, n_s_dict_2_state, n_t, n_buffer, path=path)


# # t-components

# In[7]:


n_c = 2; dof_ = 5
random_state = RandomState(1)
key_data_list_t = generate_key_data("t", dof=dof_)
DGP_dict_t = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="t", dof_=dof_, random_state=random_state) for key_data in key_data_list_t}
n_s_dict_t = dict(zip(key_data_list_t, len_list))


# In[8]:


simulate_data(DGP_dict_t, n_s_dict_t, n_t, n_buffer, path=path)


# # 3-state models

# In[9]:


n_c = 3
random_state = RandomState(10)
key_data_list_3_state = generate_key_data(n_c)
DGP_dict_3_state = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="Gaussian", random_state=random_state) for key_data in key_data_list_3_state}
n_s_dict_3_state = dict(zip(key_data_list_3_state, len_list))


# In[10]:


simulate_data(DGP_dict_3_state, n_s_dict_3_state, n_t, n_buffer, path=path)

