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


folder_path = '/hits/fast/cme/wiegerjs/placement_difficulty/tests'

# Get a list of folder names in the specified path
# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files ending with ".treefile"
tree_files = [file for file in all_files if file.endswith('_final_sbs_support_pandit.raxml.support')]
results = []
for folder_name in tree_files:
    abs_path = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace("_final_sbs_support_pandit.raxml.support", "")
    print(abs_path)
    tree_rb_path = abs_path
    tree_true_path = f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{folder_name}/tree.{folder_name}"


    try:
        tree_rb = ete3.Tree(tree_rb_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("RB Tree broken")
        print(tree_rb_path)
        continue

    try:
        tree_true = ete3.Tree(tree_true_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("True Tree broken")
        print(tree_true_path)
        continue

    branch_id_counter = 0
    for node in tree_rb.traverse():
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

    for node in tree_rb.traverse():
        if not node.is_leaf():
            bipartition_rb = get_bipartition(node)

            if bipartition_rb is not None:
                bipartition_found = False
                for node_true in tree_true.traverse():
                    if node_true.is_leaf():
                        continue
                    bipartition_true = get_bipartition(node_true)

                    if bipartition_true is not None:
                        first_match = False
                        second_match = False
                        print("#" * 50)

                        if (bipartition_rb[0] == bipartition_true[0]) or (bipartition_rb[0] == bipartition_true[1]):
                            first_match = True
                        if (bipartition_rb[1] == bipartition_true[0]) or (bipartition_rb[1] == bipartition_true[1]):
                            second_match = True
                        if second_match and first_match:  # bipartition is in true tree
                            bipartition_found = True
                            results.append((dataset, node_true.name, node.support, 1))

                if not bipartition_found:

                    results.append((dataset, node_true.name, node.support, 0))


df_res = pd.DataFrame(results, columns=["dataset", "branchID_True", "sbs_Support", "inTrue"])
df_res.to_csv(os.path.join(os.pardir, "data/sbs_simulation_pandit.csv"))
