import subprocess

folder_path = '/hits/fast/cme/wiegerjs/PANDIT/wdir'
import ete3
from ete3 import Tree
import pandas as pd
from scipy.stats import skew
import os
# Get a list of folder names in the specified path
# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files ending with ".treefile"
tree_files = [file for file in all_files if file.__contains__('bootstraps')]
results = []
for folder_name in tree_files:
    abs_path = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace(".raxml.bootstraps", "")
    #inference_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{dataset}_pandit_inference_.raxml.bestTree"
    inference_path = f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{dataset}/tree.{dataset}"

    print(inference_path)
    tree_rb_path = abs_path



    try:
        tree_inf = ete3.Tree(inference_path, format=0)
    except ete3.parser.newick.NewickError as e:
        print("True Tree broken")
        print(inference_path)
        continue

    prefix = dataset + "_final_sbs_support_pandit_v2"
    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {inference_path}",
                     f"--bs-trees {abs_path}",
                     "--redo",
                     f"--prefix {prefix}",
                     ]

    subprocess.run(" ".join(raxml_command), shell=True)