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
    dataset = folder_name.replace(".phy", "").replace("RAxML_bootstrap.", "").replace("RAxML_info.", "")
    inference_path = "/hits/fast/cme/wiegerjs/EBG_simulations/raxml_inf_res/" + dataset + ".phy_boot_test_.raxml.bestTree"

    tree_rb_path = abs_path



    try:
        tree_inf = ete3.Tree(inference_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("True Tree broken")
        print(inference_path)
        continue

    prefix = dataset + "_final_rb_support_"
    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_inf}",
                     f"--bs-trees {abs_path}",
                     "--redo",
                     f"--prefix {prefix}",
                     ]

    subprocess.run(" ".join(raxml_command), shell=True)