from ete3 import Tree
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import os


def count_supporting_branches(tree_path, threshold):
    with open(tree_path, "r") as support_file:
        tree_str = support_file.read()
        tree = Tree(tree_str)

        count = 0
        for node in tree.traverse():
            if node.support is not None and node.support >= threshold:
                count += 1

    return count / len(list(tree.traverse()))


def calculate_support_statistics(support_file_path):
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)

    support_values = []
    for node in phylo_tree.traverse():
        if node.support is not None:
            support_values.append(node.support)

    # Calculate statistics
    if len(support_values) == 0:
        return {
            "min_support": None,
            "max_support": None,
            "mean_support": None,
            "skewness": None,
            "kurtosis": None,
        }

    min_support = np.min(support_values)
    max_support = np.max(support_values)
    mean_support = np.mean(support_values)
    std_support = np.std(support_values)

    skewness = skew(support_values)
    kurt = kurtosis(support_values)

    return min_support, max_support, mean_support, std_support, skewness, kurt


def compute_rf_distance_statistics(support_file_path, reference_tree_path):
    reference_tree = Tree(reference_tree_path)
    rf_distances = []
    print(support_file_path)
    with open(support_file_path, "r") as support_file:
        for line in support_file:
            #print(line)
            bootstrap_tree = Tree(line.strip())
            results_distance = reference_tree.compare(bootstrap_tree, unrooted=True)
            print(results_distance)
            #rf_distance = reference_tree.robinson_foulds(bootstrap_tree, unrooted_trees=True)[0]
            rf_distances.append(results_distance["norm_rf"])

    print(rf_distances)

    min_rf = min(rf_distances)
    max_rf = max(rf_distances)
    mean_rf = np.mean(rf_distances)
    std_dev_rf = np.std(rf_distances)
    skewness_rf = skew(rf_distances)
    kurtosis_rf = kurtosis(rf_distances)

    return min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
filenames = filenames[:2]
results = []
counter = 0

for file in filenames:
    counter +=1
    print(counter)
    bootstrap_file = os.path.join(os.pardir, "data/raw/reference_tree/") + file + ".raxml.support"
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)

    if not os.path.exists(bootstrap_file):
        print("Skipped, no bootstrap found: " + file)
        continue

    min_support, max_support, mean_support, std_support, skewness, kurt = calculate_support_statistics(bootstrap_file)
    min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf = compute_rf_distance_statistics(bootstrap_file, tree_path)

    results.append(
        (file, min_support, max_support, mean_support, std_support, skewness, kurt, min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf))

df = pd.DataFrame(results, columns=["dataset", "min_support", "max_support", "mean_support", "std_support", "skewness", "kurt", "min_rf", "max_rf", "mean_rf", "std_dev_rf", "skewness_rf", "kurtosis_rf"
                                       ])
df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"), index=False)