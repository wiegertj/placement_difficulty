from collections import Counter
from statistics import mean
import pandas as pd
import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from scipy.stats import entropy, skew
from ete3 import Tree
import os


def split_features(tree_path, msa_filepath, dataset):
    with open(tree_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)
        branch_id_counter = 0

        for node in phylo_tree.traverse():
            branch_id_counter += 1
            if not node.is_leaf():
                node.__setattr__("name", branch_id_counter)

        results = []

        for node in phylo_tree.traverse("postorder"):
            if (not node.is_root()) and (not node.is_leaf()):

                # Create Split MSA
                list_a = []
                list_a_dist_branch = []
                list_a_dist_topo = []
                list_b = []
                list_b_dist_branch = []
                list_b_dist_topo = []
                for leaf in phylo_tree.get_leaves():
                    if leaf in node.get_leaves():
                        list_a.append(leaf.name)
                        list_a_dist_branch.append(leaf.get_distance(target=phylo_tree.get_tree_root()))
                        list_a_dist_topo.append(
                            leaf.get_distance(topology_only=True, target=phylo_tree.get_tree_root()))
                    else:
                        list_b.append(leaf.name)
                        list_b_dist_branch.append(leaf.get_distance(target=phylo_tree.get_tree_root()))
                        list_b_dist_topo.append(
                            leaf.get_distance(topology_only=True, target=phylo_tree.get_tree_root()))

                split_len_a_b =  min(len(list_b), len(list_a)) / max(len(list_b), len(list_a))
                split_min_dist_branch_a = min(list_a_dist_branch)
                split_max_dist_branch_a = max(list_a_dist_branch)
                split_mean_dist_branch_a = mean(list_a_dist_branch)
                split_std_dist_branch_a = np.std(list_a_dist_branch)
                split_skew_dist_branch_a = skew(list_a_dist_branch)

                split_min_dist_branch_b = min(list_b_dist_branch)
                split_max_dist_branch_b = max(list_b_dist_branch)
                split_mean_dist_branch_b = mean(list_b_dist_branch)
                split_std_dist_branch_b = np.std(list_b_dist_branch)
                split_skew_dist_branch_b = skew(list_b_dist_branch)

                split_min_ratio_branch = min(split_min_dist_branch_a, split_min_dist_branch_b) / max(
                    split_min_dist_branch_a, split_min_dist_branch_b)
                split_max_ratio_branch = min(split_max_dist_branch_a, split_max_dist_branch_b) / max(
                    split_max_dist_branch_a, split_max_dist_branch_b)
                split_std_ratio_branch = min(split_std_dist_branch_a, split_std_dist_branch_b) / max(
                    split_std_dist_branch_a, split_std_dist_branch_b)
                split_mean_ratio_branch = min(split_mean_dist_branch_a, split_mean_dist_branch_b) / max(
                    split_mean_dist_branch_a, split_mean_dist_branch_b)
                try:
                    split_skw_ratio_branch = min(split_skew_dist_branch_a, split_skew_dist_branch_b) / max(
                        split_skew_dist_branch_a, split_skew_dist_branch_b)
                except ZeroDivisionError:
                    split_skw_ratio_branch = 0

                split_min_dist_topo_a = min(list_a_dist_topo)
                split_max_dist_topo_a = max(list_a_dist_topo)
                split_mean_dist_topo_a = mean(list_a_dist_topo)
                split_std_dist_topo_a = np.std(list_a_dist_topo)
                split_skew_dist_topo_a = skew(list_a_dist_topo)

                split_min_dist_topo_b = min(list_b_dist_topo)
                split_max_dist_topo_b = max(list_b_dist_topo)
                split_mean_dist_topo_b = mean(list_b_dist_topo)
                split_std_dist_topo_b = np.std(list_b_dist_topo)
                split_skew_dist_topo_b = skew(list_b_dist_topo)

                split_min_ratio_topo = min(split_min_dist_topo_a, split_min_dist_topo_b) / max(
                    split_min_dist_topo_a, split_min_dist_topo_b)
                split_max_ratio_topo = min(split_max_dist_topo_a, split_max_dist_topo_b) / max(
                    split_max_dist_topo_a, split_max_dist_topo_b)
                split_std_ratio_topo = min(split_std_dist_topo_a, split_std_dist_topo_b) / max(
                    split_std_dist_topo_a, split_std_dist_topo_b)
                split_mean_ratio_topo = min(split_mean_dist_topo_a, split_mean_dist_topo_b) / max(
                    split_mean_dist_topo_a, split_mean_dist_topo_b)
                try:
                    split_skw_ratio_topo = min(split_skew_dist_topo_a, split_skew_dist_topo_b) / max(
                        split_skew_dist_topo_a, split_skew_dist_topo_b)
                except ZeroDivisionError:
                    split_skw_ratio_topo = 0

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

                entropy_differences = []

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

                for site_freq_a, site_freq_b in zip(freqs_a, freqs_b):
                    total_count_a = sum(site_freq_a.values())
                    total_count_b = sum(site_freq_b.values())
                    try:
                        normalized_freq_a = {k: v / total_count_a for
                                             k, v in site_freq_a.items()}
                    except ZeroDivisionError:
                        normalized_freq_a = {k: 0 for
                                             k, v in site_freq_a.items()}
                    try:
                        normalized_freq_b = {k: v / total_count_b for
                                             k, v in site_freq_b.items()}
                    except:
                        normalized_freq_b = {k: 0 for
                                             k, v in site_freq_b.items()}

                    site_freq_a_array = np.array(list(normalized_freq_a.values()))

                    site_freq_b_array = np.array(list(normalized_freq_b.values()))

                    entropy_a = entropy(site_freq_a_array)
                    entropy_b = entropy(site_freq_b_array)

                    entropy_difference = abs(entropy_a - entropy_b)
                    entropy_differences.append(entropy_difference)

                split_min_entropy_diff = min(entropy_differences)
                split_max_entropy_diff = max(entropy_differences)
                split_std_entropy_diff = np.std(entropy_differences)
                split_mean_entropy_diff = mean(entropy_differences)
                split_skw_entropy_diff = skew(entropy_differences)

                result = (dataset, node.name, split_len_a_b, split_min_entropy_diff, split_max_entropy_diff, split_std_entropy_diff,
                          split_mean_entropy_diff, split_skw_entropy_diff, split_min_ratio_topo, split_max_ratio_topo,
                          split_std_ratio_topo, split_mean_ratio_topo, split_skw_ratio_topo, split_min_ratio_branch,
                          split_max_ratio_branch, split_max_ratio_branch, split_mean_ratio_branch,
                          split_skw_ratio_branch, split_std_ratio_branch)
                results.append(result)

    df_res = pd.DataFrame(results, columns=["dataset", "branchId", "split_len_a_b", "split_min_entropy_diff", "split_max_entropy_diff",
                                            "split_std_entropy_diff", "split_mean_entropy_diff",
                                            "split_skw_entropy_diff", "split_min_ratio_topo", "split_max_ratio_topo",
                                            "split_std_ratio_topo", "split_mean_ratio_topo", "split_skw_ratio_topo",
                                            "split_min_ratio_branch", "split_max_ratio_branch",
                                            "split_max_ratio_branch", "split_mean_ratio_branch",
                                            "split_skw_ratio_branch", "split_std_ratio_branch"])
    return df_res


grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)

loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
filenames = filenames[:506]

counter = 0
df_list = []
for file in filenames:
    counter += 1
    print(counter)
    print(file)


    # support_path = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_1000.raxml.support"
    tree_path = os.path.join(grandir, "data/raw/reference_tree/") + file
    msa_path = os.path.join(grandir, "data/raw/msa/") + file.replace(".newick", "_reference.fasta")
    df_tmp = split_features(tree_path, msa_path, file.replace(".newick", ""))

    if not os.path.isfile(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features.csv")):
        df_tmp.to_csv(os.path.join(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features.csv")), index=False)
    else:
        df_tmp.to_csv(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features.csv"),
                     index=False,
                     mode='a', header=False)

