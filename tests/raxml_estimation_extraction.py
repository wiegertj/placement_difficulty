import pandas as pd
import os

# Specify the path to the directory containing your folders
from ete3 import Tree

list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter=0
results = []

# Loop over each subdirectory (folder) within the specified path
for folder_name in list_foldernames:
    folder_path = os.path.abspath(os.path.join(os.pardir, "data/raxml_data",str(folder_name)))
    print(folder_path)
    counter += 1
    print(counter)
    if os.path.isdir(folder_path):
        support_path = os.path.abspath(os.pardir + "/" + str(folder_name) + "/" + str(folder_name) + ".raxml.support")
        print(support_path)

        if not os.path.exists(support_path):
            print("not found")
            continue

        with open(support_path, "r") as support_file:
            tree_str = support_file.read()
            phylo_tree = Tree(tree_str)
        branch_id_counter = 0
        for node in phylo_tree.traverse():
            branch_id_counter += 1
            if node.support is not None and not node.is_leaf():
                length = node.dist
                node.__setattr__("name", branch_id_counter)

                number_nodes = sum([1 for node in phylo_tree.traverse()])
                number_inner_nodes = sum([1 for node in phylo_tree.traverse() if not node.is_leaf()])
                number_children = sum([1 for node in phylo_tree.traverse() if node.is_leaf()])

                farthest_topo = phylo_tree.get_farthest_leaf(topology_only=True)[1]
                farthest_branch = phylo_tree.get_farthest_leaf(topology_only=False)[1]
                length_relative = length / farthest_branch
                depth = node.get_distance(topology_only=True, target=phylo_tree.get_tree_root())
                num_children = sum([1 for child in node.traverse()])  # Number of leaf children
                num_children_inner = sum([1 for child in node.traverse() if not child.is_leaf()])
                num_children_leaf = sum([1 for child in node.traverse() if child.is_leaf()])

                results.append((folder_name, node.name, node.support / 100, length, length_relative, depth,
                                depth / farthest_topo, num_children,
                                num_children / number_nodes, num_children_inner, num_children_inner / num_children,
                                num_children_leaf, num_children_leaf / number_children,
                                num_children_inner / num_children_leaf))

df_final = pd.DataFrame(results, columns=["dataset", "branchId", "support", "length", "length_relative", "depth", "depth_relative","num_children",
                                                "rel_num_children", "num_children_inner", "rel_num_children_inner", "num_children_leaf", "rel_num_children_leaf",
                                                "child_inner_leaf_ratio"])

df_final.to_csv(os.path.join(os.pardir, "data/processed/target/branch_supports_raxml_data.csv"))