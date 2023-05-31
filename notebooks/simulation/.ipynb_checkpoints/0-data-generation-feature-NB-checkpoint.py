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

# # NB sojourn time

# In[6]:


n_c = 2
data_name = "NB"
random_state = RandomState(0)
key_data_dict[data_name] = generate_key_data("NB")
n_shape_ = np.array([.1, .06])

DGP_dict[data_name] = \
{key_data: get_HSMM_instance_for_sampling(*load_hardy_params(key_data.split("-")[-1], n_c), n_shape_=n_shape_, random_state=random_state) for key_data in key_data_dict[data_name]}
n_s_dict[data_name] = dict(zip(key_data_dict[data_name], len_list))


# In[7]:


simulate_data(DGP_dict[data_name], n_s_dict[data_name], n_t, n_buffer, path=path)


# # Feature engineering

# In[8]:


key_data_all = generate_key_data("NB")
feature_engineer("zhengB", key_data_all, n_buffer, path)


# # makedir

# In[9]:


makedir(path, "estimation", key_data_all)


# In[ ]:




