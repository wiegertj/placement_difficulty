import subprocess
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", ".newick")
filenames = loo_selection["dataset"].values.tolist()

counter = 0
for tree_filename in filenames:
    counter += 1
    print(str(counter) + "/" + str(len(filenames)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    print(tree_path)



    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())



    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    output_prefix = tree_filename.split(".")[0]    # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "data/raw/msa",
                                      tree_filename.replace(".newick", "_reference.fasta")+".raxml.bootstraps")

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {bootstrap_filepath}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)
