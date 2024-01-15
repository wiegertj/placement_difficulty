import os
import statistics
import subprocess
import sys

import ete3
import pandas as pd
from Bio import SeqIO, AlignIO
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingRegressor

link = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts/filtered_ebg_test.csv"
diff = "/hits/fast/cme/wiegerjs/placement_difficulty/data/treebase_difficulty_new.csv"
df = pd.read_csv(link)
df2 = pd.read_csv(diff)
df2["name"] = df2["name"].str.replace(".phy", "")
df = df.merge(df2, left_on="msa_name", right_on="name")

#idx = df.groupby('msa_name')['effect'].nlargest(10).index.get_level_values(1)
#idx2 = df.groupby('msa_name')['effect'].nsmallest(5).index.get_level_values(1)

# Extract the corresponding rows
#result_df = df.loc[idx.union(idx2)]
result_df = df
result_df = result_df.drop_duplicates(subset=['taxon', 'msa_name'])
result_new = []

print(result_df)
print(result_df.shape)

for index, row in result_df.iterrows():
    taxon = row["taxon"]
    msa_name = row["msa_name"]
    sbs_path_filtered = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_{taxon}.raxml.support"
    try:
        sbs_tree_filtered = ete3.Tree(sbs_path_filtered, format=0)
        print(sbs_tree_filtered)
    except Exception as e:
        continue

    sbs_tree_unfiltered = os.path.join(os.pardir, "data/raw/reference_tree/") + msa_name + ".newick.raxml.support"
    try:
        sbs_tree_unfiltered = ete3.Tree(sbs_tree_unfiltered, format=0)
    except Exception as e:
        continue

    leaf_node = sbs_tree_unfiltered.search_nodes(name=taxon)[0]
    leaf_node.delete()
    leaf_names = sbs_tree_unfiltered.get_leaf_names()
    print(sbs_tree_unfiltered)

    sum_support_filter = 0.0
    sum_support_unfilter = 0.0

    sum_support_filter_list = []
    sum_support_unfilter_list = []

    max_sum_support_filter = 0.0
    max_sum_support_unfilter = 0.0

    # Sum up the support values for newick_tree_original_copy
    for node in sbs_tree_unfiltered.traverse():
        if node.support is not None and not node.is_leaf():
            sum_support_unfilter += node.support
            sum_support_unfilter_list.append(node.support)
            max_sum_support_unfilter += 100

    # Sum up the support values for newick_tree_tmp
    for node in sbs_tree_filtered.traverse():
        if node.support is not None and not node.is_leaf():
            print(node.support)

            sum_support_filter += node.support
            sum_support_filter_list.append(node.support)

            max_sum_support_filter += 100

    print(sum_support_filter)
    print(sum_support_unfilter)
    elementwise_difference = [a - b for a, b in zip(sum_support_filter_list, sum_support_unfilter_list)]

    result_new.append((sum_support_filter, sum_support_unfilter, sum_support_filter / sum_support_unfilter, taxon,
                       msa_name, row['effect'], max_sum_support_unfilter, sum_support_filter / max_sum_support_unfilter,
                       sum_support_unfilter / max_sum_support_unfilter,
                       row["uncertainty_pred"] / row["max_uncertainty"], row["sequence_length"],
                       min(elementwise_difference), max(elementwise_difference),
                       statistics.stdev(elementwise_difference), skew(elementwise_difference),
                       kurtosis(elementwise_difference), row["difficulty"], row["min_1"], row["max_1"], row["mean_1"], row["std_1"], row["skew_1"], row["kurt_1"])
                      )

    print(
        f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_filter / sum_support_unfilter}")

df_final = pd.DataFrame(result_new, columns=["new_support_bs", "old_support_bs", "ratio", "taxon", "msa_name", "effect",
                                             "max_support", "new_ratio", "old_ratio", "uncertainty", "sequence_length",
                                             "min", "max", "std", "skw", "kurt", "difficulty", "min_1", "max_1","mean_1", "std_1", "skew_1", "kurt_1"])
query_df = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/query_features.csv")
query_df["msa_name"] = query_df["dataset"]
query_df["taxon"] = query_df["sampleId"]
df_final = df_final.merge(query_df, on=["msa_name", "taxon"], how="inner")
#df_final.drop(columns=["dataset", "taxon", "sampleId", "msa_name"], inplace=True)

#msa_df = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/msa_features.csv")
#msa_df["msa_name"] = msa_df["dataset"]
#df_final = df_final.merge(pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/msa_features.csv"), on=["msa_name"], how="inner")


print(df_final.sort_values("uncertainty"))
print(df_final[["ratio", "effect"]])
print(df_final["ratio"].mean())
print(df_final["ratio"].median())
print(statistics.mean(df_final["new_ratio"] - df_final["old_ratio"]))
print(statistics.mean(df_final["ratio"] - df_final["effect"]))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, median_absolute_error

df = df_final[["ratio", "effect", "uncertainty", "max_support", "sequence_length", "difficulty", "min_1", "max_1","mean_1", "std_1", "skew_1", "kurt_1",
               "gap_fraction","longest_gap_rel","frac_inv_sites_msa7","frac_inv_sites_msa8","frac_inv_sites_msa9","frac_inv_sites_msa95","frac_inv_sites_msa3","frac_inv_sites_msa1","match_rel_7","match_rel_8","match_rel_9","match_rel_95","match_rel_3","match_rel_1","match_rel_gap","match_rel_2","match_rel_4","match_rel_6","match_rel_5","transition_count_rel9","transversion_count_rel9","max_fraction_char_rests9","min_fraction_char_rests9","avg_fraction_char_rests9","std_fraction_char_rests9","skw_fraction_char_rests9","kur_fraction_char_rests9","transition_count_rel8","transversion_count_rel8","max_fraction_char_rests8","min_fraction_char_rests8","avg_fraction_char_rests8","std_fraction_char_rests8","skw_fraction_char_rests8","kur_fraction_char_rests8","transition_count_rel7","transversion_count_rel7","max_fraction_char_rests7","min_fraction_char_rests7","avg_fraction_char_rests7","std_fraction_char_rests7","skw_fraction_char_rests7","kur_fraction_char_rests7","transition_count_rel5","transversion_count_rel5","max_fraction_char_rests5","min_fraction_char_rests5","avg_fraction_char_rests5","std_fraction_char_rests5","skw_fraction_char_rests5","kur_fraction_char_rests5","gap_positions_0","gap_positions_1","gap_positions_2","gap_positions_3","gap_positions_4","gap_positions_5","gap_positions_6","gap_positions_7","gap_positions_8","gap_positions_9","approxEntropy_ape_query","cumSum_p_query","cumSum_abs_max_query","cumSum_mode_query","spec_p_query","spec_d_query","spec_n1_query","matrix_p_query","complex_p_query","complex_xObs_query","run_pi_query","run_vObs_query","run_one_p_query","run_one_x0bs_query","run_one_mean_query","run_one_std_query","run_one_min_query","run_one_max_query","randex-4_query","randex-3_query","randex-2_query","randex-1_query","randex1_query","randex2_query","randex3_query","randex4_query","g_fraction_query","a_fraction_query","t_fraction_query","c_fraction_query","rest_fraction_query","aa_stat_min_query","aa_stat_max_query","aa_stat_std_query","aa_stat_mean_query","min_gap_query","max_gap_query","mean_gap_query","cv_gap_query","sk_gap_query","kur_gap_query"]]

df['target'] = (df['ratio'] > 1.05).astype(int)
print(df["target"].value_counts())

# Print the number of NaN values for each column
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"Column '{column}': {nan_count} NaN values")

df.fillna(-1, inplace=True)

# Features (X) and target variable (y)
X = df[["effect", "uncertainty", "max_support", "sequence_length", "difficulty", "min_1", "max_1","mean_1", "std_1", "skew_1", "kurt_1",
               "gap_fraction","longest_gap_rel","frac_inv_sites_msa7","frac_inv_sites_msa8","frac_inv_sites_msa9","frac_inv_sites_msa95","frac_inv_sites_msa3","frac_inv_sites_msa1","match_rel_7","match_rel_8","match_rel_9","match_rel_95","match_rel_3","match_rel_1","match_rel_gap","match_rel_2","match_rel_4","match_rel_6","match_rel_5","transition_count_rel9","transversion_count_rel9","max_fraction_char_rests9","min_fraction_char_rests9","avg_fraction_char_rests9","std_fraction_char_rests9","skw_fraction_char_rests9","kur_fraction_char_rests9","transition_count_rel8","transversion_count_rel8","max_fraction_char_rests8","min_fraction_char_rests8","avg_fraction_char_rests8","std_fraction_char_rests8","skw_fraction_char_rests8","kur_fraction_char_rests8","transition_count_rel7","transversion_count_rel7","max_fraction_char_rests7","min_fraction_char_rests7","avg_fraction_char_rests7","std_fraction_char_rests7","skw_fraction_char_rests7","kur_fraction_char_rests7","transition_count_rel5","transversion_count_rel5","max_fraction_char_rests5","min_fraction_char_rests5","avg_fraction_char_rests5","std_fraction_char_rests5","skw_fraction_char_rests5","kur_fraction_char_rests5","gap_positions_0","gap_positions_1","gap_positions_2","gap_positions_3","gap_positions_4","gap_positions_5","gap_positions_6","gap_positions_7","gap_positions_8","gap_positions_9","approxEntropy_ape_query","cumSum_p_query","cumSum_abs_max_query","cumSum_mode_query","spec_p_query","spec_d_query","spec_n1_query","matrix_p_query","complex_p_query","complex_xObs_query","run_pi_query","run_vObs_query","run_one_p_query","run_one_x0bs_query","run_one_mean_query","run_one_std_query","run_one_min_query","run_one_max_query","randex-4_query","randex-3_query","randex-2_query","randex-1_query","randex1_query","randex2_query","randex3_query","randex4_query","g_fraction_query","a_fraction_query","t_fraction_query","c_fraction_query","rest_fraction_query","aa_stat_min_query","aa_stat_max_query","aa_stat_std_query","aa_stat_mean_query","min_gap_query","max_gap_query","mean_gap_query","cv_gap_query","sk_gap_query","kur_gap_query"]]

y = df['ratio']
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Split the data into training and testing sets
import lightgbm as lgb

# Initialize the LightGBM Regressor
regressor = lgb.LGBMRegressor()

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.01,0.1],
    'max_depth': [10, 20],
    "verbosity": [-1],
    'min_child_samples': [5, 10],  # LightGBM's equivalent of min_samples_leaf
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Perform 20 random holdouts
num_holdouts = 5
mae_scores = []
median_ae_scores = []
baseline_mae_scores = []
import numpy as np
for _ in range(num_holdouts):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())

    # Calculate and store baseline mean absolute error
    baseline_mae_scores.append(mean_absolute_error(y_test, baseline_pred))

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print('Best Hyperparameters:', grid_search.best_params_)

    # Get the feature importances and sort them in ascending order
    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted_features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1])

    # Print the sorted feature importances
    print(f'Holdout  - Sorted Feature Importances (Ascending):')
    for feature, importance in sorted_features:
        print(f'{feature}: {importance:.4f}')

    # Predict on the test set using the best model
    y_pred = grid_search.predict(X_test)

    # Calculate and store mean absolute error and median absolute error
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    median_ae_scores.append(median_absolute_error(y_test, y_pred))

# Print average performance metrics over holdouts
print(f'Average Mean Absolute Error over {num_holdouts} holdouts: {sum(mae_scores) / num_holdouts:.2f}')
import numpy as np
print(f'Average Median Absolute Error over {num_holdouts} holdouts: {np.median(median_ae_scores):.2f}')
print(f'Average Baseline Mean Absolute Error over {num_holdouts} holdouts: {sum(baseline_mae_scores) / num_holdouts:.2f}')
