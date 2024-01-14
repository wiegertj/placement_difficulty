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

idx = df.groupby('msa_name')['effect'].nlargest(10).index.get_level_values(1)
idx2 = df.groupby('msa_name')['effect'].nsmallest(5).index.get_level_values(1)

# Extract the corresponding rows
result_df = df.loc[idx.union(idx2)]
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
    sbs_tree_unfiltered = ete3.Tree(sbs_tree_unfiltered, format=0)

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
                       statistics.stdev(elementwise_difference), skew(elementwise_difference)
                       , kurtosis(elementwise_difference), row["difficulty"]))

    print(
        f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_filter / sum_support_unfilter}")

df_final = pd.DataFrame(result_new, columns=["new_support_bs", "old_support_bs", "ratio", "taxon", "msa_name", "effect",
                                             "max_support", "new_ratio", "old_ratio", "uncertainty", "sequence_length",
                                             "min", "max", "std", "skw", "kurt", "difficulty"])
print(df_final.sort_values("uncertainty"))
print(df_final[["ratio", "effect"]])
print(df_final["ratio"].mean())
print(df_final["ratio"].median())
print(statistics.mean(df_final["new_ratio"] - df_final["old_ratio"]))
print(statistics.mean(df_final["ratio"] - df_final["effect"]))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, median_absolute_error

df = df_final[["ratio", "effect", "uncertainty", "max_support", "sequence_length", "difficulty"]]

df['target'] = (df['ratio'] > 1.05).astype(int)
print(df["target"].value_counts())

# Features (X) and target variable (y)
X = df[['effect', 'uncertainty', 'max_support', "sequence_length", "difficulty"]]
y = df['ratio']
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

regressor = GradientBoostingRegressor()

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

# Perform 20 random holdouts
num_holdouts = 20
mae_scores = []
median_ae_scores = []

for _ in range(num_holdouts):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print('Best Hyperparameters:', grid_search.best_params_)

    # Predict on the test set using the best model
    y_pred = grid_search.predict(X_test)

    # Calculate and store mean absolute error and median absolute error
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    median_ae_scores.append(median_absolute_error(y_test, y_pred))
import numpy as np
# Print average performance metrics over holdouts
print(f'Average Mean Absolute Error over {num_holdouts} holdouts: {sum(mae_scores) / num_holdouts:.2f}')
print(f'Average Median Absolute Error over {num_holdouts} holdouts: {np.median(median_ae_scores):.2f}')
