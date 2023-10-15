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


filenames_filtered = filenames
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames_filtered = filenames_filtered[:300]
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick" or tree_filename == "20632_1.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    if num_leaves >= 800:
        print("Too large, skipped")
        continue



    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    if sequence_length >= 8000:
        print("Too large, skipped")
        continue
    model_path = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt"))



    output_prefix = tree_filename.split(".")[0] + "_1000_bs_raxml_classic"  # Using the filename as the prefix



    start_time = time.time()

    raxml_command = [
        "raxmlHPC",
        "-T 48"
        f"-m GTRGAMMA",
        f"-s {msa_filepath}",
        f"-# {1000}",
        "-p 12345",
        "-x 12345",
        f"-n {output_prefix}"
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
        'elapsed_time': [int(elapsed_time)],
        'num_seq': [num_sequences],
        'len': [alignment_length]
    }

    time_dat = pd.DataFrame(data)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "pars_boot_times_raxml_classic.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "pars_boot_times_raxml_classic.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "pars_boot_times_raxml_classic.csv"),
                        index=False,
                        mode='a', header=False)

#raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
