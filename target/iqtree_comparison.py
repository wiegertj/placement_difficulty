import statistics
import warnings

import ete3
from ete3 import Tree
import pandas as pd
from scipy.stats import skew

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import os


def calculate_support_statistics(support_file_path, dataset_name):
    results = []
    with open(support_file_path, "r") as support_file:
        tree_str = support_file.read()
        phylo_tree = Tree(tree_str)
    branch_id_counter = 0
    for node in phylo_tree.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)



            farthest_topo = phylo_tree.get_farthest_leaf(topology_only=True)[1]
            farthest_branch = phylo_tree.get_farthest_leaf(topology_only=False)[1]
            length_relative = length / farthest_branch
            depth = node.get_distance(topology_only=True, target=phylo_tree.get_tree_root())
            num_children = sum(1 for child in node.traverse())  # Number of leaf children
            results.append((dataset_name, node.name, node.support / 100, length, length_relative,depth, depth / farthest_topo, num_children / branch_id_counter))

    return results


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
loo_selection = loo_selection[:300]
filenames = loo_selection["dataset"].values.tolist()

counter = 0

for file in filenames:
    print(len(filenames))
    print(file)

    support_path = os.path.join(os.pardir, "scripts/") + file + "_1000.raxml.support"
    support_path_iq = os.path.join(os.pardir, "data/raw/msa/") + file + "_reference.fasta.treefile"
    print(support_path_iq)
    counter +=1
    print(counter)

    # Load the Newick trees
    tree = ete3.Tree(support_path, format=1)
    tree_iqtree = ete3.Tree(support_path_iq, format=1)

    # Get the sets of bipartitions for each tree
    bipartitions = set(tree.get_bipartitions())
    print(bipartitions)
    bipartitions_iqtree = set(tree_iqtree.get_bipartitions())

    # Find common bipartitions
    common_bipartitions = bipartitions.intersection(bipartitions_iqtree)

    # Compute the difference in support values for common bipartitions
    for common_bp in common_bipartitions:
        node = tree.get_common_ancestor(common_bp)
        node_iqtree = tree_iqtree.get_common_ancestor(common_bp)

        support = node.support if not node.is_root() else None
        support_iqtree = node_iqtree.support if not node_iqtree.is_root() else None

        support_diff = abs(support - support_iqtree) if support is not None and support_iqtree is not None else None

        print(f"Common Bipartition: {common_bp}, Support Difference: {support_diff}")





