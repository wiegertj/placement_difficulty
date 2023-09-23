import subprocess
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)

counter = 0
for tree_filename in filenames:
    counter += 1
    print(counter)
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    if num_leaves >= 500:
        print("Too large, skipped")
        continue


    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    if sequence_length >= 5000:
        print("Too large, skipped")


    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    bootstrap_filepath = os.path.join(os.pardir, "data/raw/msa",
                                      tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    output_prefix = tree_filename.split(".")[0] + "_x"  # Using the filename as the prefix

    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_path}",
        f"--bs-trees {1000}",
        f"--msa {msa_filepath}",
        "--redo"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {bootstrap_filepath}",
                     "--redo"]

    subprocess.run(" ".join(raxml_command), shell=True)
