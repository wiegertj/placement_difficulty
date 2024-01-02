import math
import pickle
import sys

import shap
import lightgbm as lgb
import os
import optuna
import random
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback


def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe


def light_gbm_regressor(rfe=False, rfe_feature_n=15, shapley_calc=True, targets=[]):
    df_pars_top = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features", "pars_top_features.csv"))
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset_noboot_no_filter.csv"))
    print(df.shape)

    df = df.merge(df_pars_top, on=["dataset"], how="inner")
    print(df.shape)

    df = df[df["no_top_boot"] <= 200]
    print(df.shape)
    df = df.drop_duplicates(subset=['dataset', "sampleId"], keep='first')
    print(df.shape)
    intervals = np.arange(0, 1.1, 0.1)

    for i in range(len(intervals) - 1):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]

        # Filter DataFrame for values within the current interval
        interval_mask = (df['entropy'] >= interval_start) & (df['entropy'] < interval_end)
        elements_in_interval = df[interval_mask].shape[0] / df.shape[0]

        print(f'Interval [{interval_start}, {interval_end}): {elements_in_interval} elements')

    df.drop(columns=["lwr_drop", "branch_dist_best_two_placements", "current_closest_taxon_perc_ham", "percentile"],
                    # "mean_a", "max_a", "min_a", "std_a", "mean_b", "max_b", "min_b", "std_b",
                     #"mean_a_good", "std_eig_sim", "max_a_good", "min_a_good", "std_a_good", "mean_b_good", "max_b_good", "min_b_good", "std_b_good","percentile"],
            inplace=True)
    print("Median Entropy: ")
    print(df["entropy"].median())
    df.columns = df.columns.str.replace(':', '_')
    # final boot
    #df = df[["dataset", "entropy", "sampleId","max_rf_tree",
    #"sk_sup_tree",
    #"transversion_count_rel5",
    #"sk_clo_sim",
    #"kur_kmer_sim",
    #"avg_entropy_msa",
    #"std_fraction_char_rests8",
    #"mean_rf_tree",
    #"min_fraction_char_rests5",
    #"sk_kmer_sim",
    #"std_kmer_sim",
    #"cumSum_abs_max_query",
    #"std_fraction_char_rests7",
    #"mean_kmer_sim",
    #"mean_sup_tree"]]

    #  final noboot
    df =df[["dataset", "entropy", "sampleId",'mean_kmer_sim', 'std_kmer_sim', 'sk_kmer_sim', 'kur_kmer_sim',
    'frac_inv_sites_msa9', 'transversion_count_rel7',
    'std_fraction_char_rests7', 'transversion_count_rel5',
    'min_fraction_char_rests5', 'std_length', 'sk_clo_sim',
    'kur_kmer_sim25', 'max_subst_freq', 'avg_rel_rf_no_boot', 'no_top_boot', #"min_loglik"
            #"mean_loglik", "min_loglik", "max_loglik", "std_loglik", "skw_loglik", "kurt_loglik",

    ]]

    #df = df[[]]
    ####  boot
    #df = df[["dataset", "entropy", "max_rf_tree", "sampleId",
     #        "mean_sup_tree",
    #         "avg_rel_rf_no_boot",
     #        "transversion_count_rel5",
      #       "sk_sup_tree",
       #      "kur_kmer_sim",
        ##     "no_top_boot",
           #  "std_fraction_char_rests7",
          #   "max_subst_freq",
            # "sk_clo_sim",
        #     "min_a_min_b",
         #    "transversion_count_rel7",
          #   "std_fraction_char_rests8",
           #  "min_fraction_char_rests5",
         #    "avg_fraction_char_rests5",
          #   "sk_kmer_sim",
           #  "std_kmer_sim",
     #        "mean_kmer_sim",
      #       "diff_match_counter_parta_w",
       #      "avg_subst_freq",
        #     "abs_weighted_distance_major_modes_supp",
         #    "diff_match_counter_parta",
          #   "transversion_count_rel8",
           #  "match_rel_8",
          #   "diff_match_counter_partb",
           #  "diff_match_counter_partb_w",
            # "kur_clo_sim",
         #    "avg_entropy_msa",
          #   "mean_a_mean_b",
           #  "cv_ham_dist"]]

    #column_name_mapping = {
     #   "max_rf_tree": "max_distance_bootstrap_trees",
      #  "mean_sup_tree": "mean_bootstrap_support_tree",
       # "avg_rel_rf_no_boot": "mean_rf_distance_parsimony",
   #     "transversion_count_rel5": "transversions_inv_sites_t5",
    #    "sk_sup_tree": "skewness_bootstrap_support_tree",
     #   "kur_kmer_sim": "kurtosis_kmer_similarity",
      #  "no_top_boot": "no_unique_topos_bootstraps",
  #      "std_fraction_char_rests7": "std_fraction_non_major_residues_t7",
   #     "max_subst_freq": "max_parsimony_subs",
    #    "sk_clo_sim": "skewness_closeness_centrality",
     #   "min_a_min_b": "min_ham_dist_central_split",
      #  "transversion_count_rel7": "transversion_count_t7",
   #     "std_fraction_char_rests8": "std_fraction_non_major_residues_t8",
    #    "min_fraction_char_rests5": "min_fraction_non_major_residues_t5",
     #   "avg_fraction_char_rests5": "mean_fraction_non_major_residues_t5",
      #  "sk_kmer_sim": "skewness_kmer_similarity",
       # "std_kmer_sim": "std_kmer_similarity",
  #      "mean_kmer_sim": "mean_kmer_similarity",
   #     "diff_match_counter_parta_w": "impure_sites_match_counter_w",
    #    "avg_subst_freq": "mean_parsimony_subs",
     #   "kur_clo_sim": "kurtosis_closeness_similarity",
      #  "cv_ham_dist": "coefficient_variation_hamming_dist"
    #}













    ### no boot
    #df = df[["avg_rel_rf_no_boot","dataset", "entropy", "sampleId",
     #   "min_fraction_char_rests5",
    #    "no_top_boot",
    #    "transversion_count_rel5",
    #    "kur_kmer_sim",
    #    "std_length",
     #   "min_a_min_b",
     #   "transversion_count_rel7",
    #    "max_subst_freq",
    #    "std_fraction_char_rests8",
    #    "std_fraction_char_rests7",
     #   "sk_clo_sim",
     #   "sk_kmer_sim",
     #   "std_kmer_sim",
     #   "diff_match_counter_parta_w",
    #    "avg_entropy_msa",
    #    "spec_n1_query",
    #    "mean_kmer_sim",
    #    "kur_kmer_sim25",
    #    "cumSum_abs_max_query",
     #   "frac_inv_sites_msa9",
     #   "diff_match_counter_parta",
    #   "rel_std_kmer_sim10",
    #    "std_fraction_char_rests5",
     #   "std_perc_hash_lcs",
    #    "match_rel_8",
    #    "approxEntropy_ape_query",
   #     "mean_a_mean_b",
    #    "avg_rel_rf_boot",
    #    "rel_gap_over_diff_sites_thresh_w"]]
    group_tree_space = ["mean_nrf_parsimony_trees", "no_topologies_parsimony_bootstrap"]
    group_inv_sites = ["inv_site_std_frac_query_msa_t7", "transversion_frac_query_msa_t7",
                       "transversion_frac_query_msa_t5",
                       "min_frac_query_msa_t5", "inv_site_matches_query_msa_t9"]
    group_sim_qs_msa = ["kurtosis_15mer_similarity", "skewness_15mer_similarity", "std_15mer_similarity",
                        "mean_15mer_similarity", "kurtosis_25mer_similarity_perc_hash"]
    group_tree_msa = ["max_parsimony_subst_freq", "std_branch_length", "skewness_closeness_centrality"]

    column_name_mapping = {"avg_rel_rf_no_boot": "mean_nrf_parsimony_trees",
    "min_fraction_char_rests5": "min_frac_query_msa_t5",
    "no_top_boot": "no_topologies_parsimony_bootstrap",
    "transversion_count_rel5": "transversion_frac_query_msa_t5",
    "kur_kmer_sim": "kurtosis_15mer_similarity",
    "std_length": "std_branch_length",
    "min_a_min_b": "min_ham_dist_central_split",
    "transversion_count_rel7": "transversion_frac_query_msa_t7",
    "max_subst_freq": "max_parsimony_subst_freq",
    "std_fraction_char_rests8": "std_frac_query_residue_msa_t8",
    "std_fraction_char_rests7": "inv_site_std_frac_query_msa_t7",
    "sk_clo_sim": "skewness_closeness_centrality",
    "sk_kmer_sim": "skewness_15mer_similarity",
    "std_kmer_sim": "std_15mer_similarity",
    "diff_match_counter_parta_w": "impure_sites_match_counter_w",
    "avg_entropy_msa": "mean_site_entropy_msa",
    "spec_n1_query": "spec_n1_query",
    "mean_kmer_sim": "mean_15mer_similarity",
    "kur_kmer_sim25": "kurtosis_25mer_similarity_perc_hash",
    "cumSum_abs_max_query": "cumSum_abs_max_query",
    "frac_inv_sites_msa9": "inv_site_matches_query_msa_t9",
    "diff_match_counter_parta": "impure_sites_match_counter",
    "rel_std_kmer_sim10": "std_10mer_similarity_perc_hash",
    "std_fraction_char_rests5": "std_frac_query_residue_msa_t5",
    "std_perc_hash_lcs": "std_perc_hash_lcs",
    "match_rel_8": "matching_sites_inv_t8",
    "approxEntropy_ape_query": "approx_entropy_ape_query",
    "mean_a_mean_b": "mean_ham_dist_central_split",
    "avg_rel_rf_boot": "mean_rf_parsimony_trees",
    "rel_gap_over_diff_sites_thresh_w": "gaps_frac_over_impure_sites"}

    pickle_version = pickle.format_version

    print("Pickle Version:", pickle_version)

    group_tree_space = ["mean_nrf_parsimony_trees", "avg_rel_rf_no_boot"]
    group_inv_sites = ["std_frac_query_residue_msa_t7", "transversion_frac_query_msa_t7", "transversion_frac_query_msa_t5",
                       "min_frac_non_major_residues_t5", "frac_inv_sites_msa9"]
    group_sim_qs_msa = ["kurtosis_15mer_similarity", "skewness_15mer_similarity", "std_15mer_similarity", "mean_15mer_similarity", "kurtosis_25mer_similarity_perc_hash"]
    group_tree_msa = ["max_parsimony_subst_freq", "std_branch_length", "skewness_closeness_centrality"]

    # Rename the columns in the DataFrame
    df = df.rename(columns=column_name_mapping)

    print(df.columns)
    print(df.shape)

    df["group"] = df['dataset'].astype('category').cat.codes.tolist()
    if targets == []:
        target = "entropy"
    else:
        target = targets

    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    y_test = test[target]

    # X = df.drop(axis=1, columns=target)
    # y = df[target]

    # X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, df["group"], test_size=0.2,
    #                                                                              random_state=12)
    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)), squared=False)
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))
    print("Baseline prediting mean MAE: " + str(mean_absolute_error(y_test, np.zeros(len(y_test)) + mean(y_test))))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)

    mae_mean = mean_absolute_error(y_test, np.zeros(len(y_test)) + mean(y_train))

    mdae_mean = median_absolute_error(y_test, np.zeros(len(y_test)) + mean(y_train))

    mbe_mean = MBE(y_test, np.zeros(len(y_test)) + mean(y_train))

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=20, min_samples_split=10,
                                      min_samples_leaf=20)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n,
                  step=0.05)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'sampleId', 'group']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'sampleId', 'group']).columns[rfe.support_]
        selected_features = selected_features.append(pd.Index(['group']))

        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset', 'sampleId'])
        X_test = X_test.drop(axis=1, columns=['dataset', 'sampleId'])

    def objective(trial):

        params = {
            'objective': 'regression',
            'metric': 'l1',
            'num_iterations': trial.suggest_int('num_iterations', 10, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.5),
            #'max_depth': -1,
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 500),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0, 1.0),
            "verbosity": -1
            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]
            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            model = lgb.train(params, train_data)
            val_preds = model.predict(X_val)
            val_score = mean_absolute_error(y_val, val_preds)
            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_params["objective"] = "regression"
    best_params["metric"] = "l1"
    best_params["boosting_type"] = "gbdt"
    #best_params["bagging_freq"] = 0
    #best_params["max_depth"] = -1

    best_score = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best MAPE training: {best_score}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model = lgb.train(best_params, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", "bad_no_filter_TEST_rmse_r1.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)

    y_pred = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae:.2f}")

    mdae = median_absolute_error(y_test, y_pred)
    print(f"MDAE on test set: {mdae}")

    mbe = MBE(y_test, y_pred)
    print(f"MBE on test set: {mbe}")

    # Create a DataFrame with the current metrics
    metrics_dict = {'RMSE': [rmse], 'MAE': [mae], 'MDAE': [mdae], 'MBE': [mbe], 'RMSE_MEAN': [rmse_mean],
                    'MAE_MEAN': [mae_mean], 'MDAE_MEAN': [mdae_mean], 'MBE_MEAN': [mbe_mean]}
    metrics_df = pd.DataFrame(metrics_dict)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "diff_guesser_noboot_new_no_filter_TEST_rmse_final.csv")):
        metrics_df.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                    "diff_guesser_noboot_new_no_filter_TEST_rmse_final.csv")), index=False)
    else:
        metrics_df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "diff_guesser_noboot_new_no_filter_TEST_rmse_final.csv"),
                          index=False,
                          mode='a', header=False)

    residuals = y_test - y_pred



    feature_importance = final_model.feature_importance(importance_type='gain')

    importance_df = pd.DataFrame(
        {'Feature': X_train.drop(axis=1, columns=["group"]).columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    total_sum = importance_df['Importance'].sum()
    importance_df['Normalized_Importance'] = importance_df['Importance'] / total_sum

    scaler = MinMaxScaler()
    # importance_df['Importance'] = scaler.fit_transform(importance_df[['Importance']])
    # importance_df = importance_df.nlargest(35, 'Importance')

    importance_df.to_csv("diff_guesser_importances_noboot_new_no_filter_TEST.csv")



    name = "8000"
    if rfe:
        name = name + "_rfe_" + str(rfe_feature_n)

    plot_filename = os.path.join(os.pardir, "data/prediction", "feature_importances_no_filter" + name + ".png")

    print("Feature Importances (Normalized):")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    X_test_["prediction"] = y_pred
    X_test_["entropy"] = y_test
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", "prediction_results_noboot_no_filter" + name + ".csv"))

    if shapley_calc:
        # X_test = X_test_[(abs(X_test_['entropy'] - X_test_['prediction']) < 0.05) & (
        #       (X_test_['entropy'] < 0.1) | (X_test_['entropy'] > 0.9))]
        X_test = X_test_
        X_test = X_test.sort_values(by="entropy")
        explainer = shap.Explainer(final_model,
                                   X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]),
                                   check_additivity=False)
        shap_values = explainer(X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]),
                                check_additivity=False)

        shap_values_group_tree_space = shap_values[:, [feature in group_tree_space for feature in X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]).columns]].sum(
            axis=1)
        shap_values_group_inv_sites = shap_values[:, [feature in group_inv_sites for feature in X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]).columns]].sum(
            axis=1)
        shap_values_group_sim_qs_msa = shap_values[:, [feature in group_sim_qs_msa for feature in X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]).columns]].sum(
            axis=1)
        shap_values_group_tree_msa = shap_values[:, [feature in group_tree_msa for feature in X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]).columns]].sum(
            axis=1)

        # Displaying the results
        #for i in range(len(X_test)):
         #   print(f"Instance {i + 1} - Group Tree Space Sum SHAP Values: {shap_values_group_tree_space[i]}")
          #  print(f"Instance {i + 1} - Group Inv Sites Sum SHAP Values: {shap_values_group_inv_sites[i]}")
           # print(f"Instance {i + 1} - Group Sim QS MSA Sum SHAP Values: {shap_values_group_sim_qs_msa[i]}")
            #print(f"Instance {i + 1} - Group Tree MSA Sum SHAP Values: {shap_values_group_tree_msa[i]}")
            #print("------------------------")
        # Create the waterfall plot
        from matplotlib import pyplot as plt
        shap.summary_plot(shap_values, X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]),
                          plot_type="violin", show=False)
        plt.tight_layout()  # Adjust layout to prevent overlapping elements

        plt.savefig(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + "shap_final.png"))

        plt.figure(figsize=(10, 6))


        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm_0.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[1500], max_display=10, show=False)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm_1500.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[-300], max_display=10, show=False)  # Limit the display to 10 features

        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm-300.png")

        shap_df = pd.DataFrame(
            np.c_[shap_values.base_values, shap_values.values],
            columns=["bv"] + list(
                X_test.drop(columns=["entropy", "prediction", "group", "sampleId", "dataset"]).columns)
        )

        # Save the DataFrame to a CSV file
        shap_df.to_csv('shap_values_noboot_no_filter_TEST.csv', index=False)




for i in range(0, 10):
    light_gbm_regressor(rfe=False, shapley_calc=True, targets=[])
