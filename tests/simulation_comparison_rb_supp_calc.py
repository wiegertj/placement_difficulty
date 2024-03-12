import subprocess

folder_path = '/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/raxml_rapid_bs_simulations'
import ete3
from ete3 import Tree
import pandas as pd
from scipy.stats import skew
import os
# Get a list of folder names in the specified path
# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files ending with ".treefile"
tree_files = [file for file in all_files if file.endswith('.phy')]
results = []
for folder_name in tree_files:
    abs_path = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace(".phy", "").replace("RAxML_bootstrap.", "")
    print(abs_path)
    tree_rb_path = abs_path
    tree_true_path = f"/hits/fast/cme/wiegerjs/EBG_simulations/data/{dataset}.phy/gtr_g.raxml.bestTree"

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
    prefix = dataset + "_final_rb_support_"
    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_true_path}",
                     f"--bs-trees {abs_path}",
                     "--redo",
                     f"--prefix {prefix}",
                     ]

    subprocess.run(" ".join(raxml_command), shell=True)