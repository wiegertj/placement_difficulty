import statistics
import types
import ete3
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis

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

    average_length = np.mean(branch_lengths)
    max_length = np.max(branch_lengths)
    min_length = np.min(branch_lengths)
    std_length = np.std(branch_lengths)
    depth = tree.get_farthest_node()[1]

    tip_branch_lengths = [node.dist for node in tree.iter_leaves()]
    average_branch_length_tips = sum(tip_branch_lengths) / len(tip_branch_lengths)
    min_branch_length_tips = min(tip_branch_lengths)
    max_branch_length_tips = max(tip_branch_lengths)
    std_branch_length_tips = statistics.stdev(tip_branch_lengths)
    skew_branch_length_tips = skew(tip_branch_lengths)
    kurtosis_branch_length_tips = kurtosis(tip_branch_lengths, fisher=False)

    all_nodes = tree.traverse()
    inner_nodes = [node for node in all_nodes if not node.is_leaf()]
    print(len(inner_nodes))
    inner_branch_lengths = [node.dist for node in inner_nodes]
    average_branch_length_inner = sum(inner_branch_lengths) / len(inner_nodes)
    min_branch_length_inner = min(inner_branch_lengths)
    max_branch_length_inner = max(inner_branch_lengths)
    std_branch_length_inner = statistics.stdev(inner_branch_lengths)
    skew_branch_length_inner = skew(inner_branch_lengths)
    kurtosis_branch_length_inner = kurtosis(inner_branch_lengths, fisher=False)

    irs = compute_tree_imbalance(tree)
    avg_irs = sum(irs) / len(irs)
    std_irs = statistics.stdev(irs)
    max_irs = max(irs)
    min_irs = min(irs)
    skew_irs = skew(irs)
    kurtosis_irs = kurtosis(irs, fisher=False)


    return tree_file.replace(".newick",
                             ""), average_length, max_length, min_length, std_length, depth, average_branch_length_tips,\
           min_branch_length_tips, max_branch_length_tips, std_branch_length_tips, skew_branch_length_tips, kurtosis_branch_length_tips,average_branch_length_inner,\
           min_branch_length_inner, max_branch_length_inner, std_branch_length_inner, skew_branch_length_inner, kurtosis_branch_length_inner, avg_irs, std_irs, max_irs, min_irs,\
           skew_irs, kurtosis_irs



def height(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return max(height(node.children[0]), height(node.children[1])) + 1


def imbalance_ratio(node):
    if node is None:
        return 0

    left_height = height(node.children[0])
    right_height = height(node.children[1])
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

    for tree_file in filenames:
        with open(os.path.join(os.pardir, "data/raw/reference_tree", tree_file), 'r') as file:
            newick_tree = file.read()

        tree = ete3.Tree(newick_tree)
        tree = normalize_branch_lengths(tree)

        result = analyze_newick_tree(tree, tree_file)
        results.append(result)

    df = pd.DataFrame(results,
                      columns=['dataset', "average_length", "max_length", "min_length", "std_length", "depth", "average_branch_length_tips",\
           "min_branch_length_tips", "max_branch_length_tips", "std_branch_length_tips", "skew_branch_length_tips", "kurtosis_branch_length_tips","average_branch_length_inner",\
           "min_branch_length_inner", "max_branch_length_inner", "std_branch_length_inner", "skew_branch_length_inner", "kurtosis_branch_length_inner", "avg_irs", "std_irs", "max_irs", "min_irs",\
           "skew_irs", "kurtosis_irs"])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"))
