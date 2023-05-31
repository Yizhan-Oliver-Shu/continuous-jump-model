#!/usr/bin/env python
# coding: utf-8

# In[1]:


## timelimit: 02:45:00
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


# # 3-model-fitting
# 
# In this notebook we fit the models on the data scenario: 3-state Hardy parametert.

# In[4]:


n_c = 3
batch_size = 4


# # Our models

# In[5]:


random_state = RandomState(10)
model_discrete = jump_model(n_c, state_type="discrete", random_state=random_state)
model_cont_mode = jump_model(n_c, state_type="cont", grid_size=.05, mode_loss=True, random_state=random_state)
model_cont = jump_model(n_c, state_type="cont", grid_size=.05, mode_loss=False, random_state=random_state)
model_dict_jump = {"discrete": model_discrete, "cont-mode": model_cont_mode, "cont": model_cont}


# In[6]:


param_grid = generate_param_grid()


# In[7]:


key_feat_jump = ["zhengB"]


# In[ ]:


model_fit_many_datas_models(generate_key_data(n_c), key_feat_jump, model_dict_jump, param_grid, path, job_id, batch_size)


# In[ ]:




