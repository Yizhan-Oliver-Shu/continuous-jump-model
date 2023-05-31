def np_save_print(filepath, arr, arr_name="arr"):
    np.save(filepath, arr)
    print(f"shape of the saved {arr_name}: {arr.shape}.")
    return

def save_estimation_results(model_params_arr, labels_arr, proba_arr, path_estimation, key_data, key_feat, key_model, job_no):
    key = f"{key_data}_{key_feat}_{key_model}_{job_no}"
    np_save_print(f"{path_estimation}/modelParams_{key}.npy", model_params_arr, "model params")
    np_save_print(f"{path_estimation}/labels_{key}.npy", labels_arr, "labels")
    np_save_print(f"{path_estimation}/proba_{key}.npy", proba_arr, "proba")    
    return 
   
def sample_from_model(model, n_trials=50, n_samples=1000, random_state=None):
    """
    generate a batch of sequences from a model, by calling the sample method.
    """
    n_f = model.n_features
    #random_state = check_random_state(random_state)
    Xs = np.empty((n_trials, n_samples, n_f))
    Zs = np.empty((n_trials, n_samples), dtype = np.int32)
    for i in tqdm(range(n_trials)): # can be slow for long sequence
        X, Z = model.sample(n_samples=n_samples)
        Xs[i] = X
        Zs[i] = Z
    return Xs, Zs

    
def simulate_and_estimate_true_GaussianHMM_model_hardy(num_state, n_s_lst, n_t, n_buffer, path_data, path_estimation, random_state):
    """
    simulate HMM using hardy parameters, and estimate by the true model.
    
    num_state: 2 or 3.
    """
    key_feat, key_model, job_id = "HMM", "true", 0
    scale_lst = ["daily", "weekly", "monthly"]
    
    for scale in scale_lst:
        # get a true HMM model
        hmm_true = get_GaussianHMM_model(*load_hardy_params(scale, num_state), random_state=random_state)
        for n_s in n_s_lst:
            # generate key for data
            key_data = f"{num_state}State{scale.capitalize()}{n_s}"
            # simulate Xs, Zs.
            Xs, Zs = sample_from_model(hmm_true, n_trials=n_t, n_samples=n_s+2*n_buffer)
            np_save_print(f"{path_data}/Xs_{key_data}_raw.npy", Xs, "Xs raw")
            Xs, Zs = Xs[:, n_buffer:-n_buffer], Zs[:, n_buffer:-n_buffer]
            np_save_print(f"{path_data}/Xs_{key_data}_HMM.npy", Xs, "Xs")
            np_save_print(f"{path_data}/Zs_{key_data}.npy", Zs, "Zs")
            # estimate by the true HMM model.
            model_params_arr, labels_arr, proba_arr = model_true_fit_batch(hmm_true, Xs)
            # save estimation results
            save_estimation_results(model_params_arr, labels_arr, proba_arr, path_estimation, key_data, key_feat, key_model, job_id)   
    return 



def feature_engineer(key_feat, key_data_list, n_buffer, path_data):
    function_dict = {"zhengBackward": lambda x: feature_engineer_zheng_batch(x, True),
                    "zhengForward": lambda x: feature_engineer_zheng_batch(x, False),
                    "ewm": feature_engineer_ewm_batch}
    if key_feat not in function_dict.keys():
        raise NotImplementedError("feature not supported yet") 
    for key_data in key_data_list:
        Xs_raw = np.load(f"{path_data}/Xs_{key_data}_raw.npy")
        Xs_feat = function_dict[key_feat](Xs_raw)[:, n_buffer:-n_buffer]
        # save results
        np_save_print(f"{path_data}/Xs_{key_data}_{key_feat}.npy", Xs_feat)
    return 

# def concat_dict_values_as_list(dictionary):
#     return list(chain.from_iterable(dictionary.values()))


# def generate_key_data_list_dict(n_s_lst = [250, 500, 1000], num_state_list = [2]):
#     scale_lst = ["daily", "weekly", "monthly"]
#     key_data_list_dict = {}
#     for num_state in num_state_list:
#         key_data_list_dict[num_state] = [f"{num_state}State{scale.capitalize()}{n_s}" for scale in scale_lst for n_s in n_s_lst]
#     return key_data_list_dict



# def feature_engineer(key_feat, key_data_list, n_buffer, path_data):
#     if key_feat in ["zhengBackward", "zhengForward"]:
#         backward = ("Backward" in key_feat)
#         for key_data in key_data_list:
#             Xs_raw = np.load(f"{path_data}/Xs_{key_data}_raw.npy")
#             Xs_feat = feature_engineer_zheng_batch(Xs_raw, backward)[:, n_buffer:-n_buffer]
#             # save results
#             np_save_print(f"{path_data}/Xs_{key_data}_{key_feat}.npy", Xs_feat)
#         return 
#     else:
#         raise NotImplementedError("feature not supported yet")  
    

# def generate_summary_df(model_params_arr, scoring_results):
#     def raise_scalar_to_list(x):
#         if np.isscalar(x): return [x] 
#         return x
#     n_c = scoring_results["acc"].shape[1]
#     means_dict, stds_dict = {}, {}
#     #
#     means_dict["model_params"] = pd.DataFrame(np.nanmean(model_params_arr, axis=0), index=generate_summary_index(n_c))
#     stds_dict["model_params"] = pd.DataFrame(np.nanstd(model_params_arr, axis=0), index=generate_summary_index(n_c))
#     #
#     acc_index = [f"Accuracy {i}" for i in range(1, n_c+1)]
#     means_dict["acc"] = pd.DataFrame(np.nanmean(scoring_results["acc"], axis=0), index=acc_index)
#     stds_dict["acc"] = pd.DataFrame(np.nanstd(scoring_results["acc"], axis=0), index=acc_index)
#     #
#     means_dict["BAC"] = means_dict["acc"].mean(0); means_dict["BAC"] = pd.DataFrame(means_dict["BAC"], columns=["BAC"]).T
#     BAC_std = compute_BAC_std_from_acc_arr(scoring_results["acc"]); BAC_std = raise_scalar_to_list(BAC_std)
#     stds_dict["BAC"] = pd.DataFrame(BAC_std, columns=["BAC"]).T
#     for name, score_arr in scoring_results.items():
#         if name == "acc":
#             continue
#         # print(raise_scalar_to_list(np.nanmean(score_arr, axis=0)))
#         means_dict[name] = pd.DataFrame(raise_scalar_to_list(np.nanmean(score_arr, axis=0)), columns=[name]).T
#         stds_dict[name] = pd.DataFrame(raise_scalar_to_list(np.nanstd(score_arr, axis=0)), columns=[name]).T
#     def combine_dict_to_df(dictionary):
#         return pd.concat(dictionary.values(), axis=0)
#     means_df, stds_df = combine_dict_to_df(means_dict).T, combine_dict_to_df(stds_dict).T
#     return means_df, stds_df, combine_means_std_df(means_df, stds_df)  #means_dict, stds_dict


# def combine_means_std_df(means_df, stds_df):
#     index, columns = means_df.index, means_df.columns
#     df = pd.DataFrame(index=index, columns=columns)
#     for idx, col in product(index, columns):
#         df.loc[idx, col] = f"{means_df.loc[idx, col]:.4f} ({stds_df.loc[idx, col]:.4f})"
#     return df



# def align_labels_proba_by_accuracy(Zs_true, proba_arr, labels_arr):
#     """
#     In a clustering problem, any permutation of the labels is a vaid clustering result. 
#     we find the best permutation for each trial. Here best refers to the highest accuracy.
    
#     Parameters:
#     ---------------------
#     labels_arr: arr of shape (n_t, n_s)
#     """
#     n_t, n_s, n_c = proba_arr.shape
#     # all the perms
#     all_perms = generate_all_perms_as_arr(n_c)
#     # all the possible perms of labels
#     labels_all_perms = permute_labels(labels_arr, all_perms)
#     # score accuracy for each perm
#     acc_all_perms = scorer_batch(accuracy_score, Zs_true, labels_all_perms, has_params=True) # of shape (n_t, n_p)
#     # best perm for each trial 
#     best_perm_idx = acc_all_perms.argmax(-1) # shape (n_t,)
#     # take the corresponding perm for labels
#     labels_arr_new = np.take_along_axis(labels_all_perms, best_perm_idx[:, np.newaxis, np.newaxis], axis=-1).squeeze(axis=-1)
#     # do the same for proba_
#     best_perm = all_perms[best_perm_idx]
#     proba_arr_new = np.take_along_axis(proba_arr, best_perm[:, np.newaxis, :], axis=-1)
#     # proba_arr_new = proba_arr[np.arange(n_t)[:, np.newaxis, np.newaxis], np.arange(n_s)[np.newaxis, :, np.newaxis], best_perm[:, np.newaxis, :]]
#     return proba_arr_new, labels_arr_new

# def align_labels_proba_by_accuracy_batch(Zs_true, proba_arr, labels_arr):
#     """
#     In a clustering problem, any permutation of the labels is a vaid clustering result. 
#     we find the best permutation for each trial. Here best refers to the highest accuracy.
    
#     Parameters:
#     ---------------------
#     labels_arr: arr of shape (n_t, n_s, n_l)
#     """
#     proba_arr_new = np.zeros_like(proba_arr)
#     labels_arr_new = np.zeros_like(labels_arr)
#     for i_lamb in range(proba_arr.shape[-1]):
#         proba_arr_new[..., i_lamb], labels_arr_new[..., i_lamb] = align_labels_proba_by_accuracy(Zs_true, proba_arr[..., i_lamb], labels_arr[..., i_lamb])
#     return proba_arr_new, labels_arr_new

# # true model
# def HMM_estimate_result(model, Xs_):
#     """
#     save the estimation result of a HMM estimator.
    
#     Note one dimension is added at axis=0, for consistency with other models
    
#     Paramaters:
#     ---------------------
#     model: an HMM instance from hmmlearn.
    
#     Xs_: array of size (n_t, n_s, n_f)
#         input data
        
#     Returns:
#     -------------------------
#     labels_arr: array of size (n_t, n_s)
    
#     proba_arr: array of size (n_t, n_s, n_c)
#     """
#     n_t, n_s, n_f = Xs_.shape
#     n_c = model.n_components
#     labels_arr = np.empty((n_t, n_s), dtype=int)
#     proba_arr = np.empty((n_t, n_s, n_c))
    
#     # for i_trial in tqdm(range(n_t)):
#     for i_trial in range(n_t):   # really fast, no need to tqdm, unless the state space becomes large.
#         labels_arr[i_trial] = model.predict(Xs_[i_trial])
#         proba_arr[i_trial] = model.predict_proba(Xs_[i_trial])   
#     return labels_arr, proba_arr



# def train_models_datas_params(key_data_list, key_feat_list, model_dict, param_grid, path_data, path_estimation, start = 0, end = -1, sub_job_no = ""):
#     """
#     train a bunch of models, w/ hyperparams to tune, on trials of data.
#     """
#     N_combos = len(key_data_list) * len(key_feat_list) * len(model_dict)
#     count=1
#     time_old = time.time(); total_time=0.
#     for key_data, key_feat in product(key_data_list, key_feat_list):
#         # load data
#         Xs = np.load(f"{path_data}/X_{key_data}_{key_feat}.npy")[start:end]
#         Zs = np.load(f"{path_data}/Z_{key_data}.npy")[start:end]
#         for key_model, model in model_dict.items():
#             # model = model_dict[key_model]
#             # train the model, on a param grid, for several trials of data
#             proba_arr_, labels_arr_ = train_one_model_one_data_batch_params(model, Xs, param_grid)
#             # align the labels
#             proba_arr_, labels_arr_ = align_labels_proba_by_accuracy_batch(Zs, proba_arr_, labels_arr_)
#             # save results
#             np_save_print(f"{path_estimation}/proba_{key_data}_{key_feat}_{key_model}{sub_job_no}.npy", proba_arr_, "proba")
#             np_save_print(f"{path_estimation}/labels_{key_data}_{key_feat}_{key_model}{sub_job_no}.npy", labels_arr_, "labels")
#             time_now = time.time(); time_this_iter = time_now-time_old; total_time += time_this_iter; time_old = time_now
#             print(f"{count}/{N_combos} combos done. Time of this iter: {print_seconds(time_this_iter)}s. Total time: {print_seconds(total_time)}s.")
#             count+=1

# def model_fit_many_datas_models(key_data_list, key_feat_list, model_dict, param_grid, path, job_id, batch_size):
#     """
#     train a collection of models, w/ hyperparams to tune, on a batch of data from many datasets.
#     """
#     def raise_str_to_list(x):
#         if isinstance(x, str):
#             return [x]
#         return x
#     key_data_list, key_feat_list = raise_str_to_list(key_data_list), raise_str_to_list(key_feat_list)
#     start = job_id * batch_size; end = start + batch_size
#     N_combos = len(key_data_list) * len(key_feat_list) * len(model_dict)
#     count = 0
#     time_old = time.time(); total_time=0.
#     for key_data, key_feat in product(key_data_list, key_feat_list):
#         folder = f"{path['data']}/{key_data}"
#         filenames = list(filter(lambda x: key_feat in x, os.listdir(folder)))
#         for filename in filenames:
#             Xs = np.load(f"{folder}/{filename}")[start:end]
#             n_s = int(filename.split('_')[1])
#             Zs = load_arr(path, "data", key_data, "Zs", n_s)[start:end]
#             for key_model, model in model_dict.items():
#                 count+=1; print(f"{count}/{N_combos} combo starts.")
#                 # train the model, on a param grid, on a batch of data
#                 model_params_arr, labels_arr, proba_arr = model_fit_batch_with_params(model, Xs, Zs, param_grid)
#                 # save results
#                 save_arr_dict({"modelParams": model_params_arr,
#                           "labels": labels_arr,
#                           "proba": proba_arr}, 
#                           path, "estimation", key_data, n_s, key_feat, key_model, job_id)
#                 time_now = time.time(); time_this_iter = time_now-time_old; total_time += time_this_iter; time_old = time_now
#                 print(f"{count}/{N_combos} combo done. Time of this combo: {print_seconds(time_this_iter)}s. Total time: {print_seconds(total_time)}s.")
#     return 


# def model_fit_many_datas_models(key_data_list, key_feat_list, model_dict, param_grid, path_data, path_estimation, job_id, batch_size):
#     """
#     train a collection of models, w/ hyperparams to tune, on a batch of data from many datasets.
#     """
#     start = job_id * batch_size; end = start + batch_size
#     N_combos = len(key_data_list) * len(key_feat_list) * len(model_dict)
#     count = 0
#     time_old = time.time(); total_time=0.
#     for key_data, key_feat in product(key_data_list, key_feat_list):
#         # load data
#         Xs = np.load(f"{path_data}/Xs_{key_data}_{key_feat}.npy")[start:end]
#         Zs = np.load(f"{path_data}/Zs_{key_data}.npy")[start:end]
#         for key_model, model in model_dict.items():
#             count+=1; print(f"{count}/{N_combos} combo starts.")
#             # train the model, on a param grid, on a batch of data
#             model_params_arr, labels_arr, proba_arr = model_fit_batch_with_params(model, Xs, Zs, param_grid)
#             # save results
#             save_estimation_results(model_params_arr, labels_arr, proba_arr, path_estimation, key_data, key_feat, key_model, job_id)
#             time_now = time.time(); time_this_iter = time_now-time_old; total_time += time_this_iter; time_old = time_now
#             print(f"{count}/{N_combos} combo done. Time of this combo: {print_seconds(time_this_iter)}s. Total time: {print_seconds(total_time)}s.")
#     return 

# def extract_idx_unique_state(Zs, n_components):
#     """
#     extract the idx of the sequence that only has one state. return as a list
#     """
#     n_t, n_s = Zs.shape
#     idx = []
#     indicator = np.empty((n_components, n_t), dtype=bool)
#     for i in range(n_components):
#         indicator[i] = (Zs==i).all(axis=1)
#         idx.append(np.where(indicator[i])[0])
#     idx.append(np.where(~indicator.any(axis=0))[0])
#     return idx


# def compute_prob_only_one_state(pi, A, T):
#     """
#     compute the probability that all states in the sequence is the same.
#     The last element is the prob of at least two states in the seq.
#     """
#     p = np.diag(A)
#     res = pi * (p ** (T-1))
#     res = np.append(res, 1-res.sum())
#     return res

def save_scoring_results(scoring_results, key, path_score):
    for score_name, scores in scoring_results.items():
        np_save_print(f"{path_score}/{score_name}_{key}.npy", scores, score_name)

# def load_combine_estimation_arrs(name, key, path_estimation, number_of_batch=1, batch_size=32):
#     """
#     load and combine one estimation arrs, from distributed locations.
#     """
#     res = [np.load(f"{path_estimation}/{name}{key}_{i}.npy")[:batch_size] for i in range(number_of_batch)]
#     return np.concatenate(res, axis=0)

# def load_combine_estimation_results(key, path_estimation, number_of_batch=1, batch_size=32):
#     """
#     load and combine estimated {"model_params_", "labels_", "proba_"}, from distributed locations.
#     """
#     estimation_results = {}
#     for name in ["modelParams_", "labels_", "proba_"]:
#         estimation_results[name] = load_combine_estimation_arrs(name, key, path_estimation, number_of_batch, batch_size)
#     return estimation_results["modelParams_"], estimation_results["labels_"], estimation_results["proba_"]
#     #  model_params_, labels_, proba_


def score_and_summary_model(key, path_data, path_estimation, path_score, number_of_batch=1, batch_size=32, param_grid=None):
    """
    score and summary one model estimation results, from distributed locations.
    """
    key_data = key.split("_")[0]
    # load true labels
    Zs = np.load(f"{path_data}/Zs_{key_data}.npy")[:number_of_batch * batch_size]  
    # load estimation results
    model_params_, labels_, proba_ = load_combine_estimation_results(key, path_estimation, number_of_batch, batch_size)
    # scoring classification accuracy
    scoring_results = scoring_labels_proba_(Zs, labels_, proba_)
    # save scores
    save_scoring_results(scoring_results, key, path_score)
    # combine model params estimation with accuracy scores
    means_df, stds_df, summary_df = generate_summary_df(model_params_, scoring_results, param_grid)
    return means_df, stds_df, summary_df
def score_and_summary_many_models(key_data_list, key_feat_list, key_model_list, path_data, path_estimation, path_score, path_figure, number_of_batch=1, batch_size=32, param_grid=None):
    for key_data, key_feat, key_model in product(key_data_list, key_feat_list, key_model_list):
        key = f"{key_data}_{key_feat}_{key_model}"
        means_df, stds_df, summary_df = score_and_summary_model(key, path_data, path_estimation, path_score, number_of_batch, batch_size, param_grid)
        means_df.to_hdf(f"{path_score}/means_{key}.h5", "means", "w")
        stds_df.to_hdf(f"{path_score}/stds_{key}.h5", "stds", "w")
        summary_df.to_hdf(f"{path_score}/summary_{key}.h5", "summary", "w")
        save_df_as_fig(summary_df, f"{path_figure}/summary_{key}.jpeg")
    return 


def generate_file_name(path, folder, key_data, name, key_len, key_feat=None, key_model=None, job_id=None):
    file_name = f"{path[folder]}/{key_data}/{name}_{key_len}"
    if key_feat is None:
        return file_name + ".npy"
    file_name += f"_{key_feat}"
    if key_model is None:
        return file_name + ".npy"
    file_name += f"_{key_model}"
    if job_id is None:
        return file_name + ".npy"
    return file_name + f"_{job_id}" + ".npy"


def feature_engineer(key_feat, key_data, n_b, path, n_s_list = None):
    """
    key_data can be a list
    """
    if isinstance(key_data, list):
        for key_data_ in key_data:
            feature_engineer(key_feat, key_data_, n_b, path)
        return 
    if isinstance(key_feat, list):
        for key_feat_ in key_feat:
            feature_engineer(key_feat_, key_data, n_b, path)
        return    

    function_dict = {"zhengB": lambda x: feature_engineer_zheng_batch(x, True),
                    "zhengF": lambda x: feature_engineer_zheng_batch(x, False),
                    "ewm": feature_engineer_ewm_batch}
    if key_feat not in function_dict.keys():
        raise NotImplementedError("feature not supported yet") 
    
    # key_feat, key_data are single key
    folder = f"{path['data']}/{key_data}"
    if n_s_list is not None:
        filenames = [generate_file_name(path, "data", key_data, "Xs", n_s, "raw") for n_s in n_s_list]
    else:
        filenames = filter_filenames_in_folder(folder, "raw")
    for filename in filenames:
        Xs_raw = np.load(f"{folder}/{filename}")
        Xs_feat = function_dict[key_feat](Xs_raw)[:, n_b:-n_b]
        # save results
        np_save_print(f"{folder}/{filename.replace('raw', key_feat)}", Xs_feat, "Xs features")            
    return 



def combine_summary_means_std_df(means_combined_df, stds_combined_df):
    df = {}
    for model in means_combined_df.index:
        df[model] = means_combined_df.loc[model]
        df[model+'-std'] = stds_combined_df.loc[model]
    df = pd.DataFrame(df)
    df.columns=["" if "std" in x else x for x in df.columns]
    return df.T

def generate_numerical_results_table(path, key_data, n_s, feat_dict, name_dict, best_idx_df):
    """
    n_s can be a list.
    """
    if isinstance(n_s, list):
        return {n_s_: generate_numerical_results_table(path, key_data, n_s_, feat_dict, name_dict, best_idx_df) for n_s_ in n_s}
    means_combined_df = {name_dict[model]: load_file(path, "score", key_data, "means", n_s, feat_dict[model], model, suffix="h5").iloc[best_idx_df.loc[model, key_data]] for model in name_dict}
        # pd.read_hdf(generate_file_name(path, "score", key_data, "means", n_s, feat_dict[model], model).replace(".npy", ".h5")).iloc[best_idx_df.loc[model, key_data]] for model in name_dict}
    stds_combined_df = {name_dict[model]: load_file(path, "score", key_data, "stds", n_s, feat_dict[model], model, suffix="h5").iloc[best_idx_df.loc[model, key_data]] for model in name_dict}
        # pd.read_hdf(generate_file_name(path, "score", key_data, "stds", n_s, feat_dict[model], model).replace(".npy", ".h5")).iloc[best_idx_df.loc[model, key_data]] for model in name_dict}
    means_combined_df, stds_combined_df = pd.DataFrame(means_combined_df).T, pd.DataFrame(stds_combined_df).T
    # accuracy of estimation
    df_acc_for_rank = {col: (means_combined_df[col]-means_combined_df.loc["true", col]).abs() if not col_is_acc(col) else -means_combined_df[col] for col in means_combined_df.columns}
    df_acc_for_rank = pd.DataFrame(df_acc_for_rank).drop("true")
    best_model_ser = df_acc_for_rank.idxmin()
    means_combined_df, stds_combined_df = means_combined_df.applymap(lambda x: f"{x:.4f}"), stds_combined_df.applymap(lambda x: f"({x:.4f})")
    for col in best_model_ser.index:
        means_combined_df.loc[best_model_ser[col], col] = "\\textbf{" + means_combined_df.loc[best_model_ser[col], col] + "}"
    # combine
    return combine_summary_means_std_df(means_combined_df, stds_combined_df) #means_combined_df, stds_combined_df, df_acc_for_rank

def turn_df_dict_to_latex(df_dict, key_data):
    string = f"% {key_data}\n" + " \\begin{table}[htbp]\n \\begin{adjustwidth}{-1in}{-1in} " + \
    "\\centering\n {\\scriptsize \n \\begin{tabular}{lllllllllll} \\toprule\n"
    string += list(df_dict.values())[0].style.to_latex().split("\n")[1]
    for n_s in df_dict.keys():
        string +=  "\n\\midrule \n" + f"{n_s}" + " &&&&&&&&&&\\\\ \n"
        string += "".join(df_dict[n_s].style.to_latex().split("\n")[2:-2])
    string += "\\bottomrule\n\\end{tabular}}\n \\end{adjustwidth}\n \\caption{"+ f"{key_data}" + "} \\label{tab:"+ f"{key_data}"+"} \\end{table}"
    return string

def save_results_in_latex(path, key_data, n_s_list, feat_dict, name_dict, best_idx_df):
    df_dict = generate_numerical_results_table(path, key_data, n_s_list, feat_dict, name_dict, best_idx_df)
    string = turn_df_dict_to_latex(df_dict, key_data)
    with open(f"{path['latex']}/{key_data}", "w") as f:
        f.write(string)
    return string



# def np_save_print(file_name, arr, arr_name="arr"):
#     """
#     save one file and print its shape. If the folder doesn't exist, creat one.
#     """
#     check_dir_exist(file_name)
#     np.save(file_name, arr)
#     print(f"shape of the saved {arr_name}: {arr.shape}.")
#     return

# def save_arr(arr, path, folder, key_data, name, key_len, key_feat=None, key_model=None, job_id=None):
#     """
#     save an arr
#     """
#     file_name = generate_file_name(path, folder, key_data, name, key_len, key_feat, key_model, job_id)
#     np_save_print(file_name, arr, name)
#     return
# def load_arr(path, folder, key_data, name, key_len, key_feat=None, key_model=None, job_id=None):
#     """
#     save an arr
#     """
#     file_name = generate_file_name(path, folder, key_data, name, key_len, key_feat, key_model, job_id)
#     return np.load(file_name)


# def save_arr_dict(arr_dict, path, folder, key_data, key_len, key_feat=None, key_model=None, job_id=None):
#     """
#     save a dict of arrs.
#     """
#     for name, arr in arr_dict.items():
#         save_arr(arr, path, folder, key_data, name, key_len, key_feat, key_model, job_id)


def simulate_data_estimate_true_model(model, len_list, n_t, n_b, fit_true_model=True, key_data=None, path=None):
    """
    model can be a dict, under which situation len_list is also a dict.
    """
    if isinstance(model, dict):
        for key_data_, model_ in tqdm(list(model.items())):
            simulate_data_estimate_true_model(model_, len_list[key_data_], n_t, n_b, fit_true_model, key_data_, path)
        return 
    # model is a model instance
    for n_s in len_list:
        Xs, Zs = sample_from_model(model, n_t, n_s+2*n_b)
        save_file(Xs, path, "data", key_data, "Xs", n_s, "raw")
        Xs, Zs = Xs[:, n_b:-n_b], Zs[:, n_b:-n_b]
        save_file(Xs, path, "data", key_data, "Xs", n_s, "HMM")
        save_file(Zs, path, "data", key_data, "Zs", n_s)      
        if fit_true_model:
            model_params_arr, labels_arr, proba_arr = model_true_fit_batch(model, Xs)
            save_file({"modelParams": model_params_arr,
                          "labels": labels_arr,
                          "proba": proba_arr}, 
                          path, "estimation", key_data, key_len=n_s, key_feat="HMM", key_model="true", job_id=0,)
    return 

def filter_filenames_in_folder(folder, key):
    filenames = os.listdir(folder)
    return list(filter(lambda x: key in x, filenames))


# true model
def model_true_fit_batch(model, Xs):
    """
    fit the true model on a batch of data. save model parameters, and fitted labels & proba
    There is no need to fit the model on the data.
        
    Paramaters:
    ---------------------
    model: a model instance.
    
    Xs: array of size (n_t, n_s, n_f)
        input data
        
    Returns:
    -------------------------
    model_params_arr: (n_t, n_c**2 + n_c)
    
    labels_arr: array of size (n_t, n_s)
    
    proba_arr: array of size (n_t, n_s, n_c)
    """
    n_t, n_s, _ = Xs.shape
    n_c = model.n_components
    # model parameters, true values
    model_params = combine_model_param_estimation(model.means_.squeeze(), model.covars_.squeeze(), model.transmat_)
    model_params_arr = np.repeat(model_params[np.newaxis, ...], n_t, 0)
    # fitted labels & proba
    labels_arr = np.empty((n_t, n_s), dtype=np.int32)
    proba_arr = np.empty((n_t, n_s, n_c))
    
    # for i_trial in tqdm(range(n_t)):
    for i_trial in range(n_t):   # really fast, no need to tqdm, unless the state space becomes large.
        X = Xs[i_trial]
        labels_arr[i_trial] = model.predict(X)
        proba_arr[i_trial] = model.predict_proba(X)   
    return model_params_arr, labels_arr, proba_arr


# def save_scoring_results(scoring_results, path, key_data, key_len, key_feat, key_model):
#     for score_name, scores_arr in scoring_results.items():
#         save_file(scores_arr, path, "score", key_data, score_name, key_len, key_feat, key_model)


def score_and_summary_many_models(path, key_data, batch_size, num_of_batch, param_grid=None):
    """
    
    """
    if isinstance(key_data, list):
        for key_data_ in key_data: score_and_summary_many_models(path, key_data_, batch_size, num_of_batch, param_grid)
        return 
    filenames = filter_filenames_in_folder(f"{path['estimation']}/{key_data}", "labels_")
    filenames = [x for x in filenames if x.endswith("_0.npy")]
    for filename in filenames:
        _, n_s_str, key_feat, key_model, _ = filename.split("_"); n_s = int(n_s_str)
        res_df = {}
        res_df["means"], res_df["stds"], res_df["summary"] = score_and_summary_model(path, key_data, n_s, key_feat, key_model, 
                                                                                     num_of_batch, batch_size, None if key_feat == "HMM" else param_grid)
        for name, df in res_df.items():
            save_file(df, path, "score", key_data, name, n_s, key_feat, key_model, suffix="h5")
            # df.to_hdf(generate_file_name(path, "score", key_data, name, n_s, key_feat, key_model).replace("npy", "h5"), name, "w")
        figure_filename = generate_file_name(path, "figure", key_data, "summary", n_s, key_feat, key_model, suffix="jpeg")#.replace("npy", "jpeg")
        check_dir_exist(figure_filename); save_df_as_fig(res_df["summary"], figure_filename)
    return 