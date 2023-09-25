import statistics
import warnings
from ete3 import Tree
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
            depth = node.get_distance(topology_only=True, target=phylo_tree.get_tree_root())
            num_children = sum(1 for child in node.traverse())  # Number of leaf children
            results.append((dataset_name, node.name, node.support / 100, length, depth, num_children))

    return results


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
results_final = []
counter = 0

for file in filenames:
    support_path = os.path.join(os.pardir, "scripts/") + file + "_1000.raxml.support"
    print(support_path)

    if os.path.exists(support_path):
        print("Found")
        results_tmp = calculate_support_statistics(support_path, file.replace(".newick", ""))
        results_final.extend(results_tmp)


df_final = pd.DataFrame(results_final, columns=["dataset", "branchId", "support", "length", "depth", "num_children"])
df_final.to_csv(os.path.join(os.pardir, "data/processed/target/branch_supports.csv"))
