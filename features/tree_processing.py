import statistics
import types
import ete3
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import dendropy
from sklearn.decomposition import PCA


def analyze_newick_tree(newick_tree, tree_file) -> tuple:
    """
    Computes summary statistics for the tree.

            Parameters:
                    :param newick_tree: tree to analyze
                    :param tree_file: needed for name extraction

            Returns:
                    :return tuple: dataset, average branch length, max branch length, min branch length, std branch length, depth
    """

    branch_lengths = [node.dist for node in newick_tree.traverse() if not node.is_root()]
    if tree_file.replace(".newick", "") == "test":
        print(sum(branch_lengths))

    tree_list = dendropy.TreeList()
    tree_list.read(data=newick_tree.write(format=1), schema="newick")

    average_length = np.mean(branch_lengths)
    max_length = np.max(branch_lengths)
    min_length = np.min(branch_lengths)
    std_length = np.std(branch_lengths)
    depth = tree.get_farthest_node()[1]

    tip_branch_lengths = [node.dist for node in tree.iter_leaves()]
    average_branch_length_tips = sum(tip_branch_lengths) / len(tip_branch_lengths)
    max_branch_length_tips = max(tip_branch_lengths)
    if len(tip_branch_lengths) >= 2:
        std_branch_length_tips = statistics.stdev(tip_branch_lengths)
        skew_branch_length_tips = skew(tip_branch_lengths)
        kurtosis_branch_length_tips = kurtosis(tip_branch_lengths, fisher=True)
    else:
        std_branch_length_tips = 0
        skew_branch_length_tips = 0
        kurtosis_branch_length_tips = 0
    all_nodes = tree.traverse()
    inner_nodes = [node.dist for node in all_nodes if not node.is_leaf()]
    # inner_branch_lengths = [node.dist for node in inner_nodes]
    average_branch_length_inner = sum(inner_nodes) / len(inner_nodes)
    min_branch_length_inner = min(inner_nodes)
    max_branch_length_inner = max(inner_nodes)
    if len(inner_nodes) >= 2:
        std_branch_length_inner = statistics.stdev(inner_nodes)
        skew_branch_length_inner = skew(inner_nodes)
        kurtosis_branch_length_inner = kurtosis(inner_nodes, fisher=True)
    else:
        std_branch_length_inner = 0
        skew_branch_length_inner = 0
        kurtosis_branch_length_inner = 0

    irs = compute_tree_imbalance(tree)
    avg_irs = sum(irs) / len(irs)
    if len(irs) >= 2:
        std_irs = statistics.stdev(irs)
        max_irs = max(irs)
        skew_irs = skew(irs)
        kurtosis_irs = kurtosis(irs, fisher=True)
    else:
        skew_irs = 0
        kurtosis_irs = 0
        std_irs = 0
    min_clo_sim, max_clo_sim, mean_clo_sim, std_clo_sim, sk_clo_sim, kur_clo_sim, min_eig_sim, max_eig_sim, mean_eig_sim, std_eig_sim, sk_eig_sim, kur_eig_sim = calculate_all_centrality_measures(newick_tree)

    return tree_file.replace(".newick",
                             ""), average_length, max_length, min_length, std_length, depth, average_branch_length_tips, \
           max_branch_length_tips, std_branch_length_tips, skew_branch_length_tips, kurtosis_branch_length_tips, average_branch_length_inner, \
           min_branch_length_inner, max_branch_length_inner, std_branch_length_inner, skew_branch_length_inner, kurtosis_branch_length_inner, avg_irs, std_irs, max_irs, \
           skew_irs, kurtosis_irs, min_clo_sim, max_clo_sim, mean_clo_sim, std_clo_sim, sk_clo_sim, kur_clo_sim, min_eig_sim, max_eig_sim, mean_eig_sim, std_eig_sim, sk_eig_sim, kur_eig_sim


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


def normalize_branch_lengths(tree):
    total_length = 0.0

    for node in tree.traverse():
        if not node.is_root():
            total_length += node.dist

    for node in tree.traverse():
        if node.up:
            node.dist /= total_length
    return tree


import networkx as nx
from ete3 import Tree


def calculate_summary_stats(centrality_dict):
    values = list(centrality_dict.values())

    return np.min(values), np.max(values), np.mean(values), np.std(values), skew(values), kurtosis(values, fisher=True)


def calculate_all_centrality_measures(ete3_tree):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Traverse the ETE3 tree and add edges to the graph with branch lengths as weights
    def traverse_and_add_edges(node):
        for child in node.children:
            edge_weight = node.get_distance(child)
            G.add_edge(node.name, child.name, weight=edge_weight)
            traverse_and_add_edges(child)

    # Start traversal from the tree root
    traverse_and_add_edges(ete3_tree)

    # Calculate all centrality measures considering edge weights (branch lengths)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    closeness_centrality = nx.closeness_centrality(G, distance='weight')
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')

    min_clo_sim, max_clo_sim, mean_clo_sim, std_clo_sim, sk_clo_sim, kur_clo_sim = calculate_summary_stats(closeness_centrality)
    min_eig_sim, max_eig_sim, mean_eig_sim, std_eig_sim, sk_eig_sim, kur_eig_sim = calculate_summary_stats(eigenvector_centrality)

    return min_clo_sim, max_clo_sim, mean_clo_sim, std_clo_sim, sk_clo_sim, kur_clo_sim, min_eig_sim, max_eig_sim, mean_eig_sim, std_eig_sim, sk_eig_sim, kur_eig_sim

if __name__ == '__main__':

    results = []

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    print(filenames)

    for file in filenames:
        if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
            print("Not found " + file)
            filenames.remove(file)

    for tree_file in filenames:
        with open(os.path.join(os.pardir, "data/raw/reference_tree", tree_file), 'r') as file:
            newick_tree = file.read()

        tree = ete3.Tree(newick_tree)
        tree = normalize_branch_lengths(tree)

        result = analyze_newick_tree(tree, tree_file)
        results.append(result)

    df = pd.DataFrame(results,
                      columns=['dataset', "average_length", "max_length", "min_length", "std_length", "depth",
                               "average_branch_length_tips", "max_branch_length_tips", "std_branch_length_tips",
                               "skew_branch_length_tips",
                               "kurtosis_branch_length_tips", "average_branch_length_inner", "min_branch_length_inner",
                               "max_branch_length_inner", "std_branch_length_inner",
                               "skew_branch_length_inner", "kurtosis_branch_length_inner", "avg_irs", "std_irs",
                               "max_irs", "skew_irs", "kurtosis_irs","min_clo_sim", "max_clo_sim", "mean_clo_sim", "std_clo_sim", "sk_clo_sim", "kur_clo_sim", "min_eig_sim", "max_eig_sim", "mean_eig_sim", "std_eig_sim", "sk_eig_sim", "kur_eig_sim"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"))
