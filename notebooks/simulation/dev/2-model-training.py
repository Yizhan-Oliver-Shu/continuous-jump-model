#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import sys, os
from os.path import expanduser
job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
## actions required!!!!!!!!!!!!!!!!!!!! change your folder path 
path = "~/Documents/G3_2/regime-identification"
path = expanduser(path)
sys.path.append(path)

path_file = f"/scratch/network/yizhans/G3_2/simulation"  
path_file = expanduser(path_file)
path_data = f"{path_file}/data"
path_estimation = f"{path_file}/estimation"
path_score = f"{path_file}/score"


# In[2]:


import numpy as np
from tqdm import tqdm
from itertools import permutations
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
import time

from numpy.random import RandomState
random_state = RandomState(0)


# In[12]:


# from regime.cluster_utils import *
# from regime.stats import *
from regime.simulation_helper import *
from regime.jump import *


# # 2-model-training
# 
# In this notebook we train several models, with different hyperparameters, and save the estimation results.

# In[4]:


scale_lst = ["daily", "weekly", "monthly"]
n_s_lst = [250, 500, 1000]
key_data_list = [f"{scale}_{n_s}" for scale in scale_lst for n_s in n_s_lst]

n_buffer, n_t, n_c = 20, 1024, 2
n_batch = 32
key_feat = "zheng"


# In[5]:


model_discrete = jump_model(n_c, state_type="discrete", random_state=random_state)
model_cont_mode = jump_model(n_c, state_type="cont", grid_size=.02, mode_loss=True, random_state=random_state)
model_cont_no_mode = jump_model(n_c, state_type="cont", grid_size=.02, mode_loss=False, random_state=random_state)


# In[6]:


model_dict = {"discrete": model_discrete, "cont_mode": model_cont_mode, "cont_no_mode": model_cont_no_mode}


# In[7]:


# lambd_list = 10 ** np.linspace(0, 8, 9)
lambd_list = 10 ** np.concatenate(([-2.], np.linspace(-1, 5, 13), [6.])) #[-2.] + list(np.linspace(-1, 5, 13)) + [6.]
print(lambd_list)
param_grid = {'jump_penalty': lambd_list}


# In[8]:


n_batch = 32


# In[9]:


key_feat_list=['zheng']


# In[10]:


train_models_datas_params(key_data_list, key_feat_list, model_dict, param_grid, path_data, path_estimation, start = job_id * n_batch, end = (job_id + 1) * n_batch, sub_job_no = job_id)






