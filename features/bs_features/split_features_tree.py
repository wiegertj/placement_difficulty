from collections import Counter
from statistics import mean
import pandas as pd
import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from scipy.stats import entropy, skew
from ete3 import Tree
import os
import networkx as nx

def height(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return max(height(node.children[0]), height(node.children[1])) + 1


def imbalance_ratio(node):
    if node is None:
        return 0
    try:
        left_height = height(node.children[0])
        right_height = height(node.children[1])
    except IndexError:
        return 0
    return max(left_height, right_height) / min(left_height, right_height)


def compute_tree_imbalance(tree):
    imbalance_values = []

    def collect_imbalance(node):
        if node is None:
            return
        if not node.is_leaf():
            ir = imbalance_ratio(node)
            imbalance_values.append(ir)
            for child in node.children:
                collect_imbalance(child)

    collect_imbalance(tree)
    return imbalance_values

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
                left_subtree, right_subtree = node.split()

                num_leaves_left = len(left_subtree.get_leaves())
                num_leaves_right = len(right_subtree.get_leaves())
                leaves_ratio = min(num_leaves_left,  num_leaves_right) / max(num_leaves_left,  num_leaves_right)

                num_children_left = sum(1 for child in left_subtree.traverse())  # Number of leaf children
                num_children_right = sum(1 for child in right_subtree.traverse())  # Number of leaf children
                children_ratio = min(num_children_left, num_children_right) / max(num_children_left, num_children_right)

                branch_lengths_left = [node.dist for node in left_subtree.traverse() if not node.is_root()]
                branch_lengths_right = [node.dist for node in right_subtree.traverse() if not node.is_root()]
                sum_bl_left = sum(branch_lengths_left)
                sum_bl_right = sum(branch_lengths_right)
                bl_ratio = min(sum_bl_left, sum_bl_right) / max(sum_bl_right, sum_bl_left)

                irs_left = compute_tree_imbalance(left_subtree)
                irs_right = compute_tree_imbalance(right_subtree)
                irs_ratio = min(irs_left, irs_right) / max(irs_right, irs_left)

                def traverse_and_add_edges(node_, G):
                    for child in node_.children:
                        edge_weight = node_.get_distance(child)
                        G.add_edge(node_.name, child.name, weight=edge_weight)
                        traverse_and_add_edges(child)
                    return G

                G_left = nx.DiGraph()
                G_left = traverse_and_add_edges(left_subtree, G_left)
                closeness_centrality_left = nx.closeness_centrality(G_left, distance='weight')
                mean_clo_sim_left = sum(closeness_centrality_left) / len(closeness_centrality_left)
                std_clo_sim_left = np.std(closeness_centrality_left)
                min_clo_sim_left = min(closeness_centrality_left)
                max_clo_sim_left = max(closeness_centrality_left)
                skw_clo_sim_left = skew(closeness_centrality_left)

                G_right = nx.DiGraph()
                G_right = traverse_and_add_edges(right_subtree, G_right)
                closeness_centrality_right = nx.closeness_centrality(G_right, distance='weight')
                mean_clo_sim_right = sum(closeness_centrality_right) / len(closeness_centrality_right)
                std_clo_sim_right = np.std(closeness_centrality_right)
                min_clo_sim_right = min(closeness_centrality_right)
                max_clo_sim_right = max(closeness_centrality_right)
                skw_clo_sim_right = skew(closeness_centrality_right)

                mean_clo_sim_ratio = min(mean_clo_sim_left, mean_clo_sim_right) / max(mean_clo_sim_left, mean_clo_sim_right)
                std_clo_sim_ratio = min(std_clo_sim_left, std_clo_sim_right) / max(std_clo_sim_left, std_clo_sim_ratio)
                max_clo_sim_ratio = min(max_clo_sim_left, max_clo_sim_right) / max(max_clo_sim_right, max_clo_sim_left)
                min_clo_sim_ratio = min(min_clo_sim_left, min_clo_sim_right) / max(min_clo_sim_right, min_clo_sim_left)
                skw_clo_sim_ratio = min(skw_clo_sim_left, skw_clo_sim_right) / max(skw_clo_sim_left, skw_clo_sim_right)


                result = (dataset, node.name, leaves_ratio, children_ratio, bl_ratio, irs_ratio, mean_clo_sim_ratio, std_clo_sim_ratio,
                          min_clo_sim_ratio,max_clo_sim_ratio,skw_clo_sim_ratio)
                results.append(result)

    df_res = pd.DataFrame(results, columns=["dataset", "branchId", "leaves_ratio", "children_ratio", "bl_ratio", "irs_ratio", "mean_clo_sim_ratio", "std_clo_sim_ratio",
                          "min_clo_sim_ratio","max_clo_sim_ratio","skw_clo_sim_ratio"])
    return df_res


grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)

#loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
targets = pd.read_csv(os.path.join(grandir, "data/processed/target/branch_supports.csv"))
#filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
#filenames = filenames[:507]

#already_processed = pd.read_csv(os.path.join(grandir, "data/processed/features/bs_features/split_features.csv"))["dataset"].unique().tolist()
#targets = targets[~targets['dataset'].isin(already_processed)]
targets["dataset"] = targets["dataset"] + ".newick"
filenames = targets["dataset"].unique().tolist()
counter = 0
df_list = []
for file in filenames:
    counter += 1
    print(counter)
    print(len(filenames))
    print(file)
    if file == "15653_0.newick" or file == "15045_1.newick":
        continue

    # support_path = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_1000.raxml.support"
    tree_path = os.path.join(grandir, "data/raw/reference_tree/") + file
    msa_path = os.path.join(grandir, "data/raw/msa/") + file.replace(".newick", "_reference.fasta")
    df_tmp = split_features(tree_path, msa_path, file.replace(".newick", ""))

    if not os.path.isfile(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features_tree.csv")):
        df_tmp.to_csv(os.path.join(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features_tree.csv")), index=False)
    else:
        df_tmp.to_csv(os.path.join(grandir, "data/processed/features/bs_features",
                             "split_features_tree.csv"),
                     index=False,
                     mode='a', header=False)

