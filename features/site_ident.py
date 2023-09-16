import math
import statistics
import sys
import warnings
from collections import Counter

from Bio.Align import MultipleSeqAlignment

from Bio import AlignIO
from ete3 import Tree
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from Bio.Align import AlignInfo

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

def calculate_imp_site(support_file_path, msa_filepath, name):
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)

        # Initialize variables to store the branch with the least support
        min_support = float('inf')  # Initialize with a high value
        min_support_branch = None

        # Iterate through all branches in the tree
        for node in phylo_tree.traverse("postorder"):
            if node.support is not None and not node.is_root() and not node.is_leaf():
                if node.support < min_support:
                    min_support = node.support
                    min_support_branch = node
        # Initialize lists to store the bipartition
        list_a = []
        list_b = []

        # Split the tree at the branch with the least support
        if min_support_branch is not None:
            for leaf in phylo_tree:
                if leaf in min_support_branch:
                    list_a.append(leaf.name)
                else:
                    list_b.append(leaf.name)

        alignment = AlignIO.read(msa_filepath, 'fasta')
        alignment_a = MultipleSeqAlignment([])
        alignment_b = MultipleSeqAlignment([])
        for record in alignment:
            if record.id in list_a:
                alignment_a.append(record)
            elif record.id in list_b:
                alignment_b.append(record)
        freqs_b = []
        freqs_a = []

        for i in range(len(alignment_a[0])):
            column_a = alignment_a[:, i]
            column_b = alignment_b[:, i]

            combined_values = column_a + column_b
            all_keys = set(combined_values)

            counter_a = Counter({key: 0 for key in all_keys})
            counter_b = Counter({key: 0 for key in all_keys})

            counter_a.update(column_a)
            counter_b.update(column_b)

            sorted_keys = sorted(all_keys)

            counter_a = Counter({key: counter_a[key] for key in sorted_keys})
            counter_b = Counter({key: counter_b[key] for key in sorted_keys})

            freqs_a.append(counter_a)
            freqs_b.append(counter_b)

        kl_divergence_results = []
        smoothing_value = 1e-10

        for site_freq_a, site_freq_b in zip(freqs_a, freqs_b):
            total_count_a = sum(site_freq_a.values())
            total_count_b = sum(site_freq_b.values())
            normalized_freq_a = {k: (v + smoothing_value) / (total_count_a + smoothing_value * len(site_freq_a)) for
                                 k, v in site_freq_a.items()}
            normalized_freq_b = {k: (v + smoothing_value) / (total_count_b + smoothing_value * len(site_freq_b)) for
                                 k, v in site_freq_b.items()}

            site_freq_a_array = np.array(list(normalized_freq_a.values()))
            site_freq_b_array = np.array(list(normalized_freq_b.values()))

            kl_divergence_value = entropy(site_freq_a_array, site_freq_b_array)

            kl_divergence_results.append(kl_divergence_value)

        min_kl_divergence = min(kl_divergence_results)
        max_kl_divergence = max(kl_divergence_results)

        # Normalize the list to the range [0, 1]
        normalized_kl_divergence_results = [(x - min_kl_divergence) / (max_kl_divergence - min_kl_divergence) for x in
                                            kl_divergence_results]
        binary_results = [1 if value > 0.5 else 0 for value in normalized_kl_divergence_results]

        threshold = sorted(normalized_kl_divergence_results)[-int(0.2 * len(normalized_kl_divergence_results))]
        #print(threshold)
        # Set values greater than or equal to the threshold to 1, and the rest to 0
        binary_results_threshold = [1 if value >= threshold else 0 for value in normalized_kl_divergence_results]
        support_kl_div_filtered_1_frac = sum(binary_results) / len(binary_results) # how much of the msa is difficult
        print(support_kl_div_filtered_1_frac)
        support_kl_div_filtered_1_frac_thresh = sum(binary_results_threshold) / len(binary_results_threshold)  # how much of the msa is difficult
        results_final = []

        for record in alignment:
            queryseq = record.seq
            non_gap_count = 0
            for char in queryseq:
                if char not in ["-", "N"]:
                    non_gap_count += 1
            gap_count = len(queryseq) - non_gap_count

            gap_match_counter = 0
            non_gap_match_counter = 0

            for i in range(len(alignment_a[0])):
                if (binary_results[i] == 1) and (queryseq[i] in ["-", "N"]):
                    gap_match_counter += 1
                elif (binary_results[i] == 1) and (queryseq[i] not in ["-", "N"]):
                    non_gap_match_counter += 1

            if sum(binary_results) != 0:
                gaps_over_diff_sites_frac = gap_match_counter / sum(binary_results) # how many of diff sites are gaps
                non_gaps_over_diff_sites_frac = non_gap_match_counter / sum(binary_results) # how many of diff sites are non gaps
            else:
                gaps_over_diff_sites_frac = 0
                non_gaps_over_diff_sites_frac = 0

            if non_gap_count != 0:
                rel_non_gap_over_diff_sites = non_gap_match_counter / non_gap_count # how much of the actual sequence without gaps lies over diff sites
            else:
                rel_non_gap_over_diff_sites = 0

            if gap_count != 0:
                rel_gap_over_diff_sites = gap_match_counter / gap_count # how much of the gaps lies over diff site
            else:
                rel_gap_over_diff_sites = 0







            gap_match_counter_thresh = 0
            non_gap_match_counter_thresh = 0


            for i in range(len(alignment_a[0])):
                if (binary_results_threshold[i] == 1) and (queryseq[i] in ["-", "N"]):
                    gap_match_counter_thresh += 1
                elif (binary_results_threshold[i] == 1) and (queryseq[i] not in ["-", "N"]):
                    non_gap_match_counter_thresh += 1

            if sum(binary_results_threshold) != 0:
                gaps_over_diff_sites_frac_thresh = gap_match_counter_thresh / sum(binary_results_threshold) # how many of diff sites are gaps
                non_gaps_over_diff_sites_frac_thresh = non_gap_match_counter_thresh / sum(binary_results_threshold) # how many of diff sites are non gaps
            else:
                gaps_over_diff_sites_frac_thresh = 0
                non_gaps_over_diff_sites_frac_thresh = 0

            if non_gap_count != 0:
                rel_non_gap_over_diff_sites_thresh = non_gap_match_counter_thresh / non_gap_count # how much of the actual sequence without gaps lies over diff sites
            else:
                rel_non_gap_over_diff_sites_thresh = 0

            if gap_count != 0:
                rel_gap_over_diff_sites_thresh = gap_match_counter_thresh / gap_count # how much of the gaps lies over diff site
            else:
                rel_gap_over_diff_sites_thresh = 0


            results_final.append((name, record.id, support_kl_div_filtered_1_frac, gaps_over_diff_sites_frac, non_gaps_over_diff_sites_frac, rel_non_gap_over_diff_sites, rel_gap_over_diff_sites,
                            support_kl_div_filtered_1_frac_thresh, gaps_over_diff_sites_frac_thresh, non_gaps_over_diff_sites_frac_thresh, rel_non_gap_over_diff_sites_thresh, rel_gap_over_diff_sites_thresh))

        columns = [
            'sampleId',
            "dataset",
            'support_kl_div_filtered_1_frac',
            'gaps_over_diff_sites_frac',
            'non_gaps_over_diff_sites_frac',
            'rel_non_gap_over_diff_sites',
            'rel_gap_over_diff_sites',
            'support_kl_div_filtered_1_frac_thresh',
            'gaps_over_diff_sites_frac_thresh',
            'non_gaps_over_diff_sites_frac_thresh',
            'rel_non_gap_over_diff_sites_thresh',
            'rel_gap_over_diff_sites_thresh'
        ]
        df = pd.DataFrame(results_final, columns=columns)
        df.to_csv(os.path.join(os.pardir, "data/processed/features",
                                name + "_diff_site_stats.csv"))
        return
def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions p and q.
    """
    kl = sum(p[i] * math.log(p[i] / q[i]) for i in p)
    return kl

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
    print(file)
    bootstrap_path = os.path.join(os.pardir, "data/raw/msa",
                                  file.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    if not os.path.exists(bootstrap_path):
        print("Skipped, no bootstrap found: " + file)
        continue

    support_path = os.path.join(os.pardir, "data/raw/reference_tree/") + file + ".raxml.support"
    msa_path = os.path.join(os.pardir, "data/raw/msa/") + file.replace(".newick", "_reference.fasta")
    calculate_imp_site(support_path, msa_path, file.replace(".newick", ""))

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)
    print("Finished " + file)
    continue
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

    if not os.path.exists(os.path.join(os.pardir, "data/processed/features",
                                           file.replace(".newick", "") + "_nearest_bootstrap.csv")):
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
    else:
        print("Found nearest bootstrap results for " + file)

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
