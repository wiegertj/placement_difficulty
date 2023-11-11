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


filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", ".newick")
filenames_aa = loo_selection["dataset"].values.tolist()

duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames_filtered = filenames[:180]
filenames_filtered = filenames_filtered + filenames_aa
print(filenames_filtered)
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7


    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())


    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    model_path = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt"))



    output_prefix = tree_filename.split(".")[0] + "ebg_test"  # Using the filename as the prefix



    start_time = time.time()

    raxml_command = [
        "ebg",
        f"-model {model_path}",
        f"-msa {os.path.abspath(msa_filepath)}",
        f"-tree {os.path.abspath(tree_path)}",
        "-t b",
        f"-o {output_prefix}"
    ]

    #print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)

    end_time = time.time()


    elapsed_time = end_time - start_time
    print("Elapsed time (seconds):", elapsed_time)
    alignment = AlignIO.read(msa_filepath, "fasta")

    num_sequences = len(alignment)

    # Get the length of the alignment
    alignment_length = alignment.get_alignment_length()

    # Print the number of sequences and length of the alignment
    print("Number of Sequences:", num_sequences)
    print("Length of Alignment:", alignment_length)

    data = {
        'dataset': [tree_filename.replace(".newick", "")],
        'elapsed_time': [int(elapsed_time)],
        'num_seq': [num_sequences],
        'len': [alignment_length]
    }

    time_dat = pd.DataFrame(data)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "benchmark_ebg_opt_nomt.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "benchmark_ebg_opt_nomt.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "benchmark_ebg_opt_nomt.csv"),
                        index=False,
                        mode='a', header=False)

#raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
