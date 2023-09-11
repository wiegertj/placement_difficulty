import statistics

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


def nearest_sequence_features(support_file_path, taxon_name):
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)

    farthest_leaf = phylo_tree.get_farthest_leaf()

    # Calculate the depth of the tree as the distance from the root to the farthest leaf
    depth = phylo_tree.get_distance(farthest_leaf, topology_only=True)

    # Print or use the depth value as needed
    print("Tree Depth:", depth)

    target_node = phylo_tree.search_nodes(name=taxon_name)[0]
    support_values = []

    current_node = target_node
    while current_node:
        if current_node.support is not None:
            support_values.append(current_node.support)
        current_node = current_node.up

    min_support = min(support_values) / 100
    max_support = max(support_values) / 100
    mean_support = np.mean(support_values)
    std_support = np.std(support_values)
    skewness = skew(support_values)
    kurt = kurtosis(support_values, fisher=True)
    depth = len(support_values) / phylo_tree.depth()

    return min_support, max_support, mean_support, std_support, skewness, kurt, depth


def calculate_support_statistics(support_file_path):
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)

    support_values = []
    for node in phylo_tree.traverse():
        if node.support is not None:
            support_values.append(node.support)

    min_support = np.min(support_values)
    max_support = np.max(support_values)
    mean_support = np.mean(support_values)
    std_support = np.std(support_values)

    skewness = skew(support_values)
    kurt = kurtosis(support_values, fisher=True)

    return min_support, max_support, mean_support, std_support, skewness, kurt


def compute_rf_distance_statistics(bootstrap_path, reference_tree_path):
    reference_tree = Tree(reference_tree_path)
    rf_distances = []
    print(bootstrap_path)
    with open(bootstrap_path, "r") as support_file:
        for line in support_file:
            bootstrap_tree = Tree(line.strip())
            results_distance = reference_tree.compare(bootstrap_tree, unrooted=True)
            rf_distances.append(results_distance["norm_rf"])

    min_rf = min(rf_distances)
    max_rf = max(rf_distances)
    mean_rf = np.mean(rf_distances)
    std_dev_rf = np.std(rf_distances)
    skewness_rf = skew(rf_distances)
    kurtosis_rf = kurtosis(rf_distances, fisher=True)

    if skewness_rf == np.nan:
        skewness_rf = 0

    if kurtosis_rf == np.nan:
        kurtosis_rf = 0

    return min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
results = []
counter = 0

for file in filenames:
    counter += 1
    print(counter)

    bootstrap_path = os.path.join(os.pardir, "data/raw/msa",
                                  file.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    if not os.path.exists(bootstrap_path):
        print("Skipped, no bootstrap found: " + file)
        continue

    support_path = os.path.join(os.pardir, "data/raw/reference_tree/") + file + ".raxml.support"
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)

    distance_file = os.path.join(os.pardir, "data/processed/features",
                                 file.replace(".newick", "") + "16p_msa_perc_hash_dist.csv")
    df_distances = pd.read_csv(distance_file)

    result_columns_nearest = ['sampleId', 'dataset', 'min_support_nearest', 'max_support_nearest', 'mean_support_nearest', 'std_support_nearest', 'skewness_nearest',
                      'kurtosis_nearest', 'depth_nearest']
    results_df_nearest = pd.DataFrame(columns=result_columns_nearest)

    for index, row in df_distances.iterrows():
        taxon_name = row['current_closest_taxon_perc_ham']
        sampleId = row['sampleId']
        datset = row['dataset']
        min_support, max_support, mean_support, std_support, skewness, kurt, depth = nearest_sequence_features(
            bootstrap_path,
            taxon_name)
        results_df = results_df.append({
            'sampleId': sampleId,
            'dataset': datset,
            'min_support': min_support,
            'max_support': max_support,
            'mean_support': mean_support,
            'std_support': std_support,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'depth': depth
        }, ignore_index=True)

    results_df_nearest.to_csv(os.path.join(os.pardir, "data/processed/features",
                                 file.replace("_reference.fasta", "") +"nearest_bootstrap.csv"))

    #min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf = compute_rf_distance_statistics(bootstrap_path,
                                                                                                 #  tree_path)

    #results.append(
     #   (file, min_support, max_support, mean_support, std_support, skewness, kurt, min_rf, max_rf, mean_rf, std_dev_rf,
      #   skewness_rf, kurtosis_rf))

#df = pd.DataFrame(results,
 #                 columns=["dataset", "min_sup_tree", "max_sup_tree", "mean_sup_tree", "std_sup_tree", "sk_sup_tree",
  #                         "kurt_support",
   #                        "min_rf_tree", "max_rf_tree", "mean_rf_tree", "std_rf_tree", "sk_rf_tree", "kur_rf_tree"
    #                       ])
#df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"), index=False)
