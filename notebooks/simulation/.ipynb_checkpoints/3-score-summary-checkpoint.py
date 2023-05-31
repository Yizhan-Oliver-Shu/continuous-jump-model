#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys, os
from os.path import expanduser
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path_repo = expanduser("~/Documents/G3_2/regime-identification"); sys.path.append(path_repo)
path_file = expanduser("/scratch/network/yizhans/G3_2/simulation")
path = {}
for folder in ["data", "estimation", "score", "summary", "figure", "latex"]:
    path[folder] = f"{path_file}/{folder}"


# In[2]:


import numpy as np


# In[3]:


from regime.simulation_helper import *


# In[4]:


batch_size, num_of_batch = 32, 32
param_grid = generate_param_grid()


# In[7]:


score_and_summary_many_models(path, generate_key_data([2, "t"], dof=5), batch_size, num_of_batch, param_grid)


# In[ ]:




