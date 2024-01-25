import math
import time
import numpy as np
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


filenames_filtered = filenames
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames_filtered = filenames_filtered[:180]
print(filenames_filtered)
loo_selection_aa = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
print(loo_selection_aa)
#filenames_filtered = loo_selection_aa
#moveon = False
#filenames_filtered = filenames_filtered[20:]
for tree_filename in loo_selection_aa: #+ loo_selection_aa:
    counter += 1
    if tree_filename == "13985_6.newick":
        continue

    if tree_filename == "15306_5.newick":
        continue#
    if tree_filename == "10454_1.newick":
        continue#
    if tree_filename == "12329_0.newick":
        continue#
    if tree_filename == "14244_0.newick":
        continue
    if tree_filename == "10436_0.newick":
        continue

    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7


    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())




    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    alignment = AlignIO.read(msa_filepath, "fasta")

    sequence_data = [list(record.seq) for record in alignment]

    alignment_array = np.array(sequence_data)
    no_col = alignment.get_alignment_length()

    set_cols = set()

    for col in range(0, no_col - 1):
        column = "".join(alignment_array[:, col].tolist())
        set_cols.add(column)
    num_sequences = len(alignment)

    # Get the length of the alignment
    alignment_length = alignment.get_alignment_length()
    print("Unique cols: " + str(len(set_cols)))
    print("Number of Sequences:", num_sequences)
    print("Length of Alignment:", alignment_length)

    thread_num = math.ceil((len(set_cols) / 500))
    if thread_num > 60:
        thread_num = 60
    print("Chosen threads:" + str(thread_num))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    model_path = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt"))

    if tree_filename.split(".")[0] == "15669_9":
        continue


    if tree_filename.split(".")[0] == "20675_0":
        continue

    output_prefix = tree_filename.split(".")[0] + "_1000_bs_raxml_classic"  # Using the filename as the prefix



    start_time = time.time()
    print(str(counter) + "/" + str(len(filenames_filtered)))

    raxml_command = [
        "raxmlHPC",
        f"-T 60",
        f"-m PROTGAMMAGTR",
        f"-s {msa_filepath}",
        f"-# autoMRE",
        "-p 12345",
        "-x 12345",
        "-w /hits/fast/cme/wiegerjs/placement_difficulty/data/processed/raxml_rapid_bs_deimos_test",
        f"-n {output_prefix}"    ]

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
        'dataset': tree_filename.replace(".newick", ""),
        'elapsed_time': int(elapsed_time),
        'num_seq': num_sequences,
        'len': alignment_length
    }
    data_res = [data]
    time_dat = pd.DataFrame(data_res)

    #if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
     #                                  "benchmark_rapid_bootstrap_deimos_aa.csv")):
      #  time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
       #                                           "benchmark_rapid_bootstrap_deimos_aa.csv")), index=False)
    #else:
     #   time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
      #                               "benchmark_rapid_bootstrap_deimos_aa.csv"),
       #                 index=False,
        #                mode='a', header=False)

#raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
