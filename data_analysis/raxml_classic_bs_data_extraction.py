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

filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy",
                                                                                                       ".newick").values.tolist()

filenames_filtered = filenames
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames_filtered = filenames_filtered[:300]
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    # if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7
    classic_raxml_boots = os.path.join(os.pardir, "target/RAxML_bootstrap." + tree_filename.replace(".newick",
                                                                                                    "") + "_1000_bs_raxml_classic")
    if not os.path.exists(classic_raxml_boots):
        print("support not found")
        continue

    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick" or tree_filename == "20632_1.newick":
        print("skipped too large!")
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


