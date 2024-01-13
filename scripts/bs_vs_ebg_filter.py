import os
import statistics
import subprocess
import sys

import ete3
import pandas as pd
from Bio import SeqIO, AlignIO
from scipy.stats import kurtosis, skew
link = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts/filtered_ebg_test.csv"
df = pd.read_csv(link)

idx = df.groupby('msa_name')['effect'].nlargest(3).index.get_level_values(1)

# Extract the corresponding rows
result_df = df.loc[idx]
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

    result_new.append((sum_support_filter, sum_support_unfilter, sum_support_filter / sum_support_unfilter,taxon, msa_name, row['effect'], max_sum_support_unfilter, sum_support_filter/max_sum_support_unfilter,
                       sum_support_unfilter/max_sum_support_unfilter, row["uncertainty_pred"] / row["max_uncertainty"], row["sequence_length"], min(elementwise_difference), max(elementwise_difference), statistics.stdev(elementwise_difference), skew(elementwise_difference)
                       , kurtosis(elementwise_difference)))


    print(f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_filter / sum_support_unfilter}")

df_final = pd.DataFrame(result_new, columns=["new_support_bs", "old_support_bs", "ratio","taxon", "msa_name", "effect", "max_support", "new_ratio", "old_ratio", "uncertainty", "sequence_length", "min", "max", "std", "skw", "kurt"])
print(df_final.sort_values("uncertainty"))
print(df_final[["ratio", "effect"]])
print(df_final["ratio"].mean())
print(df_final["ratio"].median())
print(statistics.mean(df_final["new_ratio"] - df_final["old_ratio"]))
print(statistics.mean(df_final["ratio"] - df_final["effect"]))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, classification_report
df = df_final[["ratio", "effect", "uncertainty", "max_support", "sequence_length", "min", "max", "std", "skw", "kurt"]]

df['target'] = (df['ratio'] > 1).astype(int)

# Features (X) and target variable (y)
X = df[['effect', 'uncertainty', 'max_support', "sequence_length",  "min", "max", "std", "skw", "kurt"]]
y = df['target']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [1, 2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8]
}
grid_search = GridSearchCV(classifier, param_grid, cv=40, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print('Best Hyperparameters:', grid_search.best_params_)

# Predict on the test set using the best model
y_pred = grid_search.predict(X_test)

classifier.fit(X_train, y_train)

# Print the feature importances
feature_importances = classifier.feature_importances_
print('Feature Importances:')
for feature, importance in zip(X.columns, feature_importances):
    print(f'{feature}: {importance:.4f}')

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))