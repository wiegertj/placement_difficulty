import os
import statistics
import subprocess
import sys

import ete3
import pandas as pd
from Bio import SeqIO, AlignIO

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

    max_sum_support_filter = 0.0
    max_sum_support_unfilter = 0.0


    # Sum up the support values for newick_tree_original_copy
    for node in sbs_tree_unfiltered.traverse():
        if node.support is not None and not node.is_leaf():
            sum_support_unfilter += node.support
            max_sum_support_unfilter += 100

    # Sum up the support values for newick_tree_tmp
    for node in sbs_tree_filtered.traverse():
        if node.support is not None and not node.is_leaf():
            print(node.support)

            sum_support_filter += node.support
            max_sum_support_filter += 100

    print(sum_support_filter)
    print(sum_support_unfilter)

    result_new.append((sum_support_filter, sum_support_unfilter, sum_support_filter / sum_support_unfilter,taxon, msa_name, row['effect'], max_sum_support_unfilter, sum_support_filter/max_sum_support_unfilter,
                       sum_support_unfilter/max_sum_support_unfilter, row["uncertainty_pred"] / row["max_uncertainty"]))


    print(f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_filter / sum_support_unfilter}")

df_final = pd.DataFrame(result_new, columns=["new_support_bs", "old_support_bs", "ratio","taxon", "msa_name", "effect", "max_support", "new_ratio", "old_ratio", "uncertainty"])
print(df_final.sort_values("uncertainty"))
print(df_final[["ratio", "effect"]])
print(df_final["ratio"].mean())
print(df_final["ratio"].median())
print(statistics.mean(df_final["new_ratio"] - df_final["old_ratio"]))
print(statistics.mean(df_final["ratio"] - df_final["effect"]))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, classification_report
df = df_final[["ratio", "effect", "uncertainty"]]

df['target'] = (df['ratio'] > 1).astype(int)

# Features (X) and target variable (y)
X = df[['effect', 'uncertainty']]
y = df['target']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier()

cv_scores = cross_val_score(classifier, X, y, cv=10, scoring='accuracy')

# Print cross-validation scores
print('Cross-validation scores:', cv_scores)
print('Mean accuracy:', cv_scores.mean())