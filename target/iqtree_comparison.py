import statistics
import time
import warnings

import ete3
from ete3 import Tree
import pandas as pd
from scipy.stats import skew

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import os
def get_bipartition(node):
    if not node.is_leaf():
        left_children = sorted([leaf.name for leaf in node.children[0].iter_leaves()])
        right_children = sorted([leaf.name for leaf in node.children[1].iter_leaves()])
        bipartition = (left_children, right_children)
        return bipartition
    return None

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
    try:
        tree = ete3.Tree(support_path, format=0)
        tree_iqtree = ete3.Tree(support_path_iq, format=0)
    except ete3.parser.newick.NewickError:
        print("Tree broken")
        continue

    results = []

    branch_id_counter = 0
    for node in tree.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)

    branch_id_counter = 0
    for node in tree_iqtree.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)

    for node in tree.traverse():
        if not node.is_leaf():
            bipartition = get_bipartition(node)

            if bipartition is not None:
                for node_iq in tree_iqtree.traverse():
                    bipartition_iq = get_bipartition(node_iq)
                    if bipartition_iq is not None:
                        first_match = False
                        second_match = False
                        if (bipartition[0] == bipartition_iq[0]) or (bipartition[0] == bipartition_iq[1]):
                            first_match = True
                        if (bipartition[1] == bipartition_iq[0]) or (bipartition[1] == bipartition_iq[1]):
                            second_match = True
                        if second_match and first_match:
                            print((node.name, node_iq.name, (node.support - node_iq.support)/100))
                            results.append((node.name, node_iq.name, (node.support - node_iq.support)/100))


    print(str(len(results) / len([1 for node in tree.traverse() if not node.is_leaf()])))
    time.sleep(3)




