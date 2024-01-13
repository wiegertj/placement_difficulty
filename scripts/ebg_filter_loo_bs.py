import os
import subprocess
import sys

import ete3
import pandas as pd
from Bio import SeqIO, AlignIO

link = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts/filtered_ebg_test.csv"
df = pd.read_csv(link)

idx = df.groupby('msa_name')['effect'].nlargest(3).index.get_level_values(1)

# Extract the corresponding rows
result_df = df.loc[idx]

print(result_df)
print(result_df.shape)

for index, row in result_df.iterrows():


    taxon = row["taxon"]
    msa_name = row["msa_name"]
    if msa_name == "21191_0":
        continue

    if os.path.exists(os.path.abspath(os.path.join(os.pardir, "scripts",
                                          msa_name + "_" + taxon + ".raxml.bootstraps"))):
        print("skipped")
        continue

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    filepath = os.path.abspath(filepath)

    original_tree_path_tmp = os.path.abspath(os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick"))
    model_path_tmp = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"))

    new_alignment = []
    MSA = AlignIO.read(filepath, 'fasta')


    # Delete one out of MSA
    for record in MSA:
        if record.id != taxon:
            seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")

            new_alignment.append(seq_record)

    output_file = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_ebg_filter_" + taxon + ".fasta")
    output_file = os.path.abspath(output_file)

    with open(output_file, "w") as new_alignment_output:
        SeqIO.write(new_alignment, new_alignment_output, "fasta")

    # Delete one from tree
    original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick")

    tree_path = original_tree_path  # use original tree without reestimation
    print("-------------------------------------------")

    print("Getting original from " + tree_path)
    print("Start without reestimation")

    with open(tree_path, 'r') as file:
        print(tree_path)
        newick_tree = file.read()
        tree = ete3.Tree(newick_tree)

        leaf_names = tree.get_leaf_names()
        leaf_count = len(leaf_names)
        leaf_node = tree.search_nodes(name=taxon)[0]
        leaf_node.delete()
        leaf_names = tree.get_leaf_names()
        leaf_count = len(leaf_names)
        original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree_tmp",
                                          msa_name + "_" + taxon + "_ebg_filter.newick")
        print("Start creating loo tree")
        original_tree_path = os.path.abspath(original_tree_path)
        print("Storing to: " + original_tree_path)

        newick_string = tree.write()
        try:
            with open(original_tree_path, 'w') as file:
                file.write(newick_string)
            print(f"Newick tree has been saved to {original_tree_path}")
        except Exception as e:
            print(f"An error occurred while saving the Newick tree: {str(e)}")
        print("-------------------------------------------")

        bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                          msa_name + "_" + taxon + ".raxml.bootstraps")

        raxml_command = [
            "raxml-ng",
            "--bootstrap",
            "--model", model_path_tmp,
            f"--bs-trees {1000}",
            "--msa", output_file,
            "--redo",
            "--prefix", msa_name + "_" + taxon
        ]

        print(" ".join(raxml_command))
        subprocess.run(" ".join(raxml_command), shell=True)

        print(f"Bootstrap analysis for {original_tree_path} completed.")

        raxml_command = ["raxml-ng",
                         "--support",
                         f"--tree {original_tree_path}",
                         f"--bs-trees {bootstrap_filepath}",
                         "--redo",
                         f"--prefix " + msa_name + "_" + taxon,
                         ]

        subprocess.run(" ".join(raxml_command), shell=True)
