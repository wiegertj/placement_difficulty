import ete3
import numpy as np
import os
import pandas as pd


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

    average_length = np.mean(branch_lengths)
    max_length = np.max(branch_lengths)
    min_length = np.min(branch_lengths)
    std_length = np.std(branch_lengths)
    depth = tree.get_farthest_node()[1]

    return tree_file.replace(".newick", ""), average_length, max_length, min_length, std_length, depth


if __name__ == '__main__':

    results = []

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_list = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

    loo_list= loo_list + ["bv.newick", "tara.newick", "neotrop.newick"]
    print(loo_list)

    for tree_file in loo_list:
        with open(os.path.join(os.pardir, "data/raw/reference_tree", tree_file), 'r') as file:
            newick_tree = file.read()

        tree = ete3.Tree(newick_tree)

        result = analyze_newick_tree(tree, tree_file)
        results.append(result)

    df = pd.DataFrame(results,
                      columns=['dataset', 'avg_blength', 'max_blength', 'min_blength', 'std_blength', 'tree_depth'])
    df.to_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"))
