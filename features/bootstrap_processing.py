import statistics
import warnings

from ete3 import Tree
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

    farthest_leaf, tree_depth = phylo_tree.get_tree_root().get_farthest_leaf(topology_only=True)
    total_tree_length = sum(node.dist for node in phylo_tree.traverse())

    target_node = phylo_tree.search_nodes(name=taxon_name)[0]
    support_values = []
    branch_lengths = []

    current_node = target_node
    while current_node:
        if current_node.support is not None:
            support_values.append(current_node.support)
        branch_lengths.append(current_node.dist / total_tree_length)
        current_node = current_node.up

    min_support_ = min(support_values) / 100
    max_support_ = max(support_values) / 100
    mean_support_ = statistics.mean(support_values) / 100
    std_support_ = statistics.stdev(support_values) / 100

    skewness_ = skew(support_values)
    kurt_ = kurtosis(support_values, fisher=True)
    depth_ = (len(support_values) - 1) / tree_depth

    min_branch_len_nearest = min(branch_lengths)
    max_branch_len_nearest = max(branch_lengths)
    mean_branch_len_nearest = statistics.mean(branch_lengths)
    std_branch_len_nearest = statistics.stdev(branch_lengths)
    sk_branch_len_nearest = skew(branch_lengths)
    kurt_branch_len_nearest = kurtosis(branch_lengths, fisher=True)

    return min_support_, max_support_, mean_support_, std_support_, skewness_, kurt_, depth_, min_branch_len_nearest, max_branch_len_nearest, mean_branch_len_nearest, std_branch_len_nearest, sk_branch_len_nearest, kurt_branch_len_nearest


def calculate_support_statistics(support_file_path):
    print("Calc support")
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
    import matplotlib.pyplot as plt

    intervals = [(i * 10, (i + 1) * 10) for i in range(10)]

    # Calculate the values falling into each interval
    interval_counts = [np.sum((value >= start) & (value <= end) for value in support_values) for start, end in
                       intervals]

    # Find the two intervals with the most values
    top_intervals = np.argsort(interval_counts)[-2:]

    # Calculate the interval distance between the two most frequent intervals
    abs_distance_major_modes_supp = abs(intervals[top_intervals[1]][1] - intervals[top_intervals[0]][0])
    distance_major_modes_supp = intervals[top_intervals[1]][1] - intervals[top_intervals[0]][0]

    # Calculate the percentage difference in value counts for both intervals
    percentage_difference = abs(interval_counts[top_intervals[0]] - interval_counts[top_intervals[1]]) / len(support_values)

    # Calculate the weighted interval distance using the percentage difference
    weighted_distance_major_modes_supp = distance_major_modes_supp / percentage_difference if percentage_difference > 0 else 1
    abs_weighted_distance_major_modes_supp = abs_distance_major_modes_supp / percentage_difference if percentage_difference > 0 else 1


    print("Top Intervals with the Most Values:")
    for i in top_intervals:
        print(f"Interval Range: {intervals[i]}, Value Count: {interval_counts[i]}")
    print("Support number: " + str(len(support_values)))
    print("Distance between modes:", distance_major_modes_supp)
    print("Abs Distance between modes:", abs_distance_major_modes_supp)
    print("Percentage Difference in Data Points:", percentage_difference)
    print("Weighted Distance between modes:", weighted_distance_major_modes_supp)
    print("Abs Weighted Distance between modes:", abs_weighted_distance_major_modes_supp)


    if (skewness >= 2) and (len(support_values) >= 50):
        import matplotlib.pyplot as plt
        print("printed")
        # Assuming 'support_values' is your list of values
        plt.figure(figsize=(8, 4))  # Create a new figure

        # Create a histogram
        plt.hist(support_values, bins=20, edgecolor='k')  # You can adjust the number of bins as needed

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Support Values - Skewness large: ' + str(skewness) + " " + support_file_path)

        # Save the plot as a figure (e.g., 'fig.png')
        plt.savefig('sk_supp_big.png')
    if (skewness <= 0.2) and len(support_values) >= 50:
        import matplotlib.pyplot as plt
        print("printed")
        # Assuming 'support_values' is your list of values
        plt.figure(figsize=(8, 4))  # Create a new figure

        # Create a histogram
        plt.hist(support_values, bins=20, edgecolor='k')  # You can adjust the number of bins as needed

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Support Values - Skewness small: ' + str(skewness) + " " + support_file_path)

        # Save the plot as a figure (e.g., 'fig.png')
        plt.savefig('sk_supp_small.png')

    kurt = kurtosis(support_values, fisher=True)

    return min_support, max_support, mean_support, std_support, skewness, kurt, distance_major_modes_supp, abs_distance_major_modes_supp, weighted_distance_major_modes_supp,abs_weighted_distance_major_modes_supp


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
    try:
        mean_rf = np.mean(rf_distances)
    except np.core._exceptions._UFuncNoLoopError:
        print("Problem occured")
        return 0,0,0,0,0,0
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
    if file == "20632_1.newick":
        continue
    counter += 1
    print(counter)
    print(file)
    bootstrap_path = os.path.join(os.pardir, "data/raw/msa",
                                  file.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    if not os.path.exists(bootstrap_path):
        print("Skipped, no bootstrap found: " + file)
        continue

    support_path = os.path.join(os.pardir, "data/raw/reference_tree/") + file + ".raxml.support"
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)

    distance_file = os.path.join(os.pardir, "data/processed/features",
                                 file.replace(".newick", "") + "16p_msa_perc_hash_dist.csv")
    if not os.path.exists(distance_file):
        print("Distance file not found ... " + file)
        continue

    try:
        df_distances = pd.read_csv(distance_file)
    except pd.errors.EmptyDataError:
        print("Empty distance dataframe found ... skipped")
        continue

    #if not os.path.exists(os.path.join(os.pardir, "data/processed/features",
     #                                      file.replace(".newick", "") + "_nearest_bootstrap.csv")):
    result_columns_nearest = ['sampleId', 'dataset', 'min_support_nearest', 'max_support_nearest',
                              'mean_support_nearest', 'std_support_nearest', 'skewness_nearest',
                              'kurtosis_nearest', 'depth_nearest', "min_branch_len_nearest", "max_branch_len_nearest",
                              "mean_branch_len_nearest", "std_branch_len_nearest", "sk_branch_len_nearest",
                              "kurt_branch_len_nearest"]
    results_df_nearest = pd.DataFrame(columns=result_columns_nearest)
    float_columns = [
        'min_support_nearest', 'max_support_nearest', 'mean_support_nearest',
        'std_support_nearest', 'skewness_nearest', 'kurtosis_nearest',
        'depth_nearest', 'min_branch_len_nearest', 'max_branch_len_nearest',
        'mean_branch_len_nearest', 'std_branch_len_nearest', 'sk_branch_len_nearest',
        'kurt_branch_len_nearest'
    ]

    results_df_nearest[float_columns] = results_df_nearest[float_columns].astype(float)

    # Initialize an empty list to store dictionaries
    results_data = []

    for index, row in df_distances.iterrows():
        taxon_name = row['current_closest_taxon_perc_ham']
        sampleId = row['sampleId']
        datset = row['dataset']

        min_support, max_support, mean_support, std_support, skewness, kurt, depth, min_branch_len_nearest, max_branch_len_nearest, mean_branch_len_nearest, std_branch_len_nearest, sk_branch_len_nearest, kurt_branch_len_nearest = nearest_sequence_features(
            support_path,
            taxon_name)

        # Append a dictionary to the list
        results_data.append({
            'sampleId': sampleId,
            'dataset': datset,
            'min_support_nearest': min_support,
            'max_support_nearest': max_support,
            'mean_support_nearest': mean_support,
            'std_support_nearest': std_support,
            'skewness_nearest': skewness,
            'kurtosis_nearest': kurt,
            'depth_nearest': depth,
            "min_branch_len_nearest": min_branch_len_nearest,
            "max_branch_len_nearest": max_branch_len_nearest,
            "mean_branch_len_nearest": mean_branch_len_nearest,
            "std_branch_len_nearest": std_branch_len_nearest,
            "sk_branch_len_nearest": sk_branch_len_nearest,
            "kurt_branch_len_nearest": kurt_branch_len_nearest
        })

    # Create a DataFrame from the list of dictionaries
    results_df_nearest = pd.DataFrame(results_data)

    results_df_nearest.to_csv(os.path.join(os.pardir, "data/processed/features",
                                           file.replace(".newick", "") + "_nearest_bootstrap.csv"))
    #else:
     #   print("Found nearest bootstrap results for " + file)

    min_rf, max_rf, mean_rf, std_dev_rf, skewness_rf, kurtosis_rf = compute_rf_distance_statistics(bootstrap_path,
    tree_path)


    min_support, max_support, mean_support, std_support, skewness, kurt, distance_major_modes_supp, abs_distance_major_modes_supp, weighted_distance_major_modes_supp,abs_weighted_distance_major_modes_supp = calculate_support_statistics(support_path)

    results.append(
    (file, min_support, max_support, mean_support, std_support, skewness, kurt, distance_major_modes_supp, abs_distance_major_modes_supp, weighted_distance_major_modes_supp,abs_weighted_distance_major_modes_supp, min_rf, max_rf, mean_rf, std_dev_rf,
    skewness_rf, kurtosis_rf))

df = pd.DataFrame(results,
             columns=["dataset", "min_sup_tree", "max_sup_tree", "mean_sup_tree", "std_sup_tree", "sk_sup_tree",
                     "kurt_support", "distance_major_modes_supp", "abs_distance_major_modes_supp", "weighted_distance_major_modes_supp","abs_weighted_distance_major_modes_supp",
                    "min_rf_tree", "max_rf_tree", "mean_rf_tree", "std_rf_tree", "sk_rf_tree", "kur_rf_tree"
                   ])
df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"), index=False)
