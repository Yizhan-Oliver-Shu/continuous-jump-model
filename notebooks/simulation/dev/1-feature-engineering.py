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


# In[3]:


from regime.simulation_helper import *
from regime.stats import *


# # 1-feature-engineering
# 
# In this notebook we perform feature engineering.

# #  Zheng features

# In[4]:


n_b = 20


# In[5]:


key_data_list = generate_key_data([2, 3, "t"], dof=5)
key_data_list


# In[ ]:


feature_engineer(["zhengF", "zhengB"], key_data_list, n_b, path)

