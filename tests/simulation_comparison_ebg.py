import warnings

import ete3
from ete3 import Tree
import pandas as pd
from scipy.stats import skew
import os


def get_bipartition(node):
    if not node.is_leaf():
        left_children = sorted([leaf.name for leaf in node.children[0].iter_leaves()])
        right_children = sorted([leaf.name for leaf in node.children[1].iter_leaves()])
        bipartition = (left_children, right_children)
        return bipartition
    return None

folder_path = '/hits/fast/cme/wiegerjs/EBG_simulations/ebg_results/'

# Get a list of folder names in the specified path
folder_names = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

results = []
for folder_name in folder_names:
    abs_path = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace(".phy", "")
    print(abs_path)
    tree_ebg_path = abs_path + f"/{dataset}.phy_median_support_prediction.newick"
    tree_true_path = f"/hits/fast/cme/wiegerjs/EBG_simulations/data/{folder_name}/gtr_g.raxml.bestTree"

    try:
        tree_ebg = ete3.Tree(tree_ebg_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("EBG Tree broken")
        print(tree_ebg_path)
        continue

    try:
        tree_true = ete3.Tree(tree_true_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("True Tree broken")
        print(tree_true_path)
        continue

    branch_id_counter = 0
    for node in tree_ebg.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)

    branch_id_counter = 0
    for node in tree_true.traverse():
        branch_id_counter += 1
        if node.support is not None and not node.is_leaf():
            length = node.dist
            node.__setattr__("name", branch_id_counter)

    for node in tree_ebg.traverse():
        if not node.is_leaf():
            bipartition_ebg = get_bipartition(node)

            if bipartition_ebg is not None:
                bipartition_found = False
                for node_true in tree_true.traverse():
                    if node_true.is_leaf():
                        continue
                    bipartition_true = get_bipartition(node_true)
                    if bipartition_true is not None:
                        first_match = False
                        second_match = False
                        if (bipartition_ebg[0] == bipartition_true[0]) or (bipartition_ebg[0] == bipartition_true[1]):
                            first_match = True
                        if (bipartition_ebg[1] == bipartition_true[0]) or (bipartition_ebg[1] == bipartition_true[1]):
                            second_match = True
                        if second_match and first_match:  # bipartition is in true tree
                            bipartition_found = True
                            if node.support >= 70:  # bipartition is deemed significant
                                # store true positive
                                print((node.name, node_true.name, node.support))
                                results.append((dataset, node.name, node.support))
                            else:  # bipartition is NOT deemed significant
                                # store false negative
                                print((node.name, node_true.name, node.support))
                                results.append((dataset, node.name, node.support))

df_res = pd.DataFrame(results, columns=["dataset", "branchId", "ebg_median"])
df_res.to_csv(os.path.join(os.pardir, "data/ebg_simulation_median.csv"))
