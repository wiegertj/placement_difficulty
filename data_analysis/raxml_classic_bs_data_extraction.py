import time

print("Started")

import pandas as pd

print("Started1")

import subprocess

print("Started2")

import os

print("Started3")

from ete3 import Tree

print("Started4")

from Bio import SeqIO, AlignIO

print("Started5")

filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy",".newick").values.tolist()
results = []
loo_selection_aa = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
filenames_filtered = filenames[:180]
filenames_filtered = filenames_filtered + loo_selection_aa
counter = 0
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    # if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7
    classic_raxml_boots = os.path.join(os.pardir, "data/processed/raxml_rapid_bs_deimos_test/" + tree_filename.replace(".newick",
                                                                                                    "") + "_1000_bs_raxml_classic")
    print(classic_raxml_boots)
    if not os.path.exists(classic_raxml_boots):
        print("support not found")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    output_prefix = tree_filename.split(".")[0] + "_1000_bs_raxml_classic"  # Using the filename as the prefix

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {classic_raxml_boots}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)

    support_tree_path = "/hits/fast/cme/wiegerjs/placement_difficulty/data_analysis/" + tree_filename.replace(".newick",
                                                                                                              "") + "_1000_bs_raxml_classic.raxml.support"
    try:
        with open(support_tree_path, "r") as support_file:
            tree_str = support_file.read()
            tree = Tree(tree_str)  #

            branch_id_counter = 0

            for node in tree.traverse():
                all_supps = []
                all_diff_supps = []
                branch_id_counter += 1
                node.__setattr__("name", branch_id_counter)
                if node.support is not None and not node.is_leaf():
                    results.append((tree_filename.replace(".newick", ""), branch_id_counter, node.support))
    except FileNotFoundError:
        continue

results_df = pd.DataFrame(results, columns=["dataset", "branchId", "support_raxml_classic"])
results_df.to_csv("raxml_classic_supports.csv")
