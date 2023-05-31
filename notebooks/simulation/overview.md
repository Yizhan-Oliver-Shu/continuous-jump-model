# Overview of Simulation Study

## structure of notebooks

| notebook                  | folder                           |
|---------------------------|----------------------------------|
| 0-data-generation-feature | data, estimation                 |
| 1-model-fitting-**          | data, estimation             |
| 2-score-summary           | data, estimation, score, summary |
| 3-figure-from-summary | summary, figure, summary-cluster, figure-cluster|
| 4-latex-generation | summary, best-idx, latex


There are 4 main parts:

-   data generation
-   estimation
-   scoring
-   summary

## 1. Data Generation

Each scenario is given a key `key_data` describing the DGP and seq length. Use the function `sample_from_model(model, n_trials, n_samples)` to sample sequences.

Data: `{path[folder]}/{key_data}/{name}_{key_len}_{key_feat}_{key_model}_{job_id}`.

-   `Zs_{key_data}.npy: (n_t, n_s)`. The true labels.

The observed sequences are generally named by `Xs_{key_data}_{key_feat}.npy: (n_t, n_s, 1)`,

-   `Xs_{key_data}_raw.npy: (n_t, n_s+n_b, 1)` is the raw data. `n_b` is for feature engineering later.
-   `Xs_{key_data}_HMM.npy: (n_t, n_s, 1)` neglects the buffer samples, and is used for fitting true or estimated HMMs.

True model estimation is also done whenever needed; see below.

## 2. Model Estimation/Fitting

Given a batch of data `key_data` & `key_feat`, and a model `key_model`, use the function `model_fit_batch(model, Xs, Zs)` to fit a model on a batch of data. Three arrays are saved as estimation results:

-   model parameters: `(n_t, n_c**2+n_c)`.
-   estimated labels: `(n_t, n_s)`.
-   estimated proba: `(n_t, n_s, n_c)`.

For models with hyperparams to tune, use the function \`\` to fit the model with every possible hyperparam combination on a batch of data.

After fitting, save:

-   estimated model parameters: `means_arr: (n_t, n_c)`, `covars_arr: (n_t, n_c)`, `transmat_arr: (n_t, n_c, n_c)`. Combine them into `model_params_arr: (n_t, n_c**2+n_c)`. Take sqrt of `covars_arr` to get `stds_arr`.
-   estimated labels: `(n_t, n_s)`.
-   estimated proba: `(n_t, n_s, n_c)`.

Note:

-   For the true model, estimated model params are the true value. The model doesn't need fitting on the data.
-   For the other models, we need to consider all permutations and retain the permutation with the highest overall accuracy.
-   For models with hyperparams, use the function \`\` to fit it on a batch of data. The dimension is added to the last axis.

The estimation results are saved in:

-   `model_params_{key_data}_{key_feat}_{key_model}_{job_id}.npy: (n_t, n_c**2+n_c, n_l)`.
-   `labels_{key_data}_{key_feat}_{key_model}_{job_id}.npy: (n_t, n_s, n_l)`.
-   `proba_{key_data}_{key_feat}_{key_model}_{job_id}.npy: (n_t, n_s, n_c, n_l)`.

## 3. Scoring and summary

Whenever a batch of random samples are generated, we use the true model to estimate the sample, and score the true model. Use function `true_model_fit` to fit the true model on a batch of data.

Estimation:

-   `labels_{key_data}_HMM_true.npy: (n_t, n_s)`.
-   `proba_{key_data}_HMM_true.npy: (n_t, n_s, n_c)`.

In general estimations are named by e.g. `labels_{key_data}_{key_feat}_{key_model}.npy: (n_t, n_s, n_l)`.

## Model fitting

Later we use function `function` to fit a model with hyperparams on a batch of data. In general estimations are named by

-   `params`
-   `labels_{key_data}_{key_feat}_{key_model}.npy: (n_t, n_s, n_l)`.
-   `proba_{key_data}_{key_feat}_{key_model}.npy: (n_t, n_s, n_c, n_l)`.

## Scoring

Now we compute the following scores from `labels_` and `proba_`,

-   `acc_arr: (n_t, n_c)`: accuracy per class.
-   `roc_auc_arr: (n_t, )`: ROC-AUC, for the trials with all the classes.
-   `BAC`: average of accuracy per class.

## scoring summary

combines parameter estimation and accuracy scores.

## Model fitting

Models to compare:

-   true HMM model,
-   discrete jump models,
-   continuous jump models, w/ mode loss,
-   continuous jump models, w/o mode loss,
-   estimated HMM model (`cov_type`: diag vs full ?),
-   sparse jump models (to be done).

# shape of data

-   estimation result. For models with hyperparameters to tune, we put the dimension of hyperparams to the end of the shape.
    -   `{path_estimation}/labels_{key_data}_true`, `(n_t, n_s)`.
    -   `{path_estimation}/proba_{key_data}_true`, `(n_t, n_s, n_c)`.
    -   `{path_estimation}/labels_{key_data}_{key_model}_{key_feat}`, `(n_t, n_s, n_l)`.
    -   `{path_estimation}/proba_{key_data}_{key_model}_{key_feat}`, `(n_t, n_s, n_l)`.
-   scores:
    -   `{path_score}/acc_{key_data}_true`, `(n_t, n_c)`.
    -   `{path_score}/roc_auc_{key_data}_true`, `(n_t)`.
    -   `{path_score}/acc_{key_data}_{key_model}_{key_feat}`, `(n_t, n_c, n_l)`.
    -   `{path_score}/roc_auc_{key_data}_{key_model}_{key_feat}`, `(n_t, n_l)`.

# Input Parameters

From three aspects:

-   simulated data: means, covars, transmat, (startprob,) n_s, n_t.
-   feature: `zheng` or `ewma`.
-   model: `true`, `discrete`, `cont_mode`, `cont_no_mode`.

# workflow

## 0-generate-data-estimate-true-model

-   generate `key_data` (e.g. `daily_1000`).
-   get a true HMM model.
-   simulate `X_raw_{key_data}`, `Z_{key_data}`.
-   estimate by the true HMM model. `labels_{key_data}_true`, `proba_{key_data}_true`.
-   score the estimation by the true model.

## 1-feature-engineering

-   create features for all the trials.

## 2-model-training
