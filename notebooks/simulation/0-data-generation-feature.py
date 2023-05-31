#!/usr/bin/env python
# coding: utf-8

# In[1]:


# timelimit = 00:10:00
import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path_repo = expanduser("~/Documents/G3_2/regime-identification"); sys.path.append(path_repo)
path_file = expanduser("/scratch/network/yizhans/G3_2/simulation")
path = {}
for folder in ["data", "estimation"]:
    path[folder] = f"{path_file}/{folder}"


# In[2]:


import numpy as np
from numpy.random import RandomState


# In[3]:


from regime.simulation_helper import *


# In[4]:


n_t, n_buffer = 1024, 20
len_list = [[250, 500, 1000, 2000], [50, 100, 250, 500, 1000], [60, 120, 250, 500]]


# In[5]:


key_data_dict = {}
DGP_dict = {}
n_s_dict = {}


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

# In[6]:


n_c = 2
data_name = f"{n_c}-state"
random_state = RandomState(0)
key_data_dict[data_name] = generate_key_data(n_c)
DGP_dict[data_name] = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="Gaussian", random_state=random_state) for key_data in key_data_dict[data_name]}
n_s_dict[data_name] = dict(zip(key_data_dict[data_name], len_list))


# In[7]:


simulate_data(DGP_dict[data_name], n_s_dict[data_name], n_t, n_buffer, path=path)


# # t-components

# In[8]:


n_c = 2; dof_ = 5
data_name = f"t-{dof_}"
random_state = RandomState(1)
key_data_dict[data_name] = generate_key_data("t", dof=dof_)
DGP_dict[data_name] = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="t", dof_=dof_, random_state=random_state) for key_data in key_data_dict[data_name]}
n_s_dict[data_name] = dict(zip(key_data_dict[data_name], len_list))


# In[9]:


simulate_data(DGP_dict[data_name], n_s_dict[data_name], n_t, n_buffer, path=path)


# # 3-state models

# In[10]:


n_c = 3
data_name = f"{n_c}-state"
random_state = RandomState(10)
key_data_dict[data_name] = generate_key_data(n_c)
DGP_dict[data_name] = {key_data: \
            get_HMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), emission="Gaussian", random_state=random_state) for key_data in key_data_dict[data_name]}
n_s_dict[data_name] = dict(zip(key_data_dict[data_name], len_list))


# In[11]:


simulate_data(DGP_dict[data_name], n_s_dict[data_name], n_t, n_buffer, path=path)


# # Feature engineering

# In[13]:


key_data_all = generate_key_data([2, 3, "t"], dof=5)
feature_engineer("zhengB", key_data_all, n_buffer, path)


# # makedir

# In[19]:


makedir(path, "estimation", key_data_all)


# In[ ]:




