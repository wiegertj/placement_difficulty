import time
import pandas as pd
import subprocess
import os
from ete3 import Tree
from Bio import SeqIO, AlignIO

filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", ".newick")
filenames_aa = loo_selection["dataset"].values.tolist()
filenames_filtered = filenames[:180]
counter = 0
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    print(tree_path)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    msa_filepath = os.path.abspath(os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta")))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    os.chdir("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ufboot2_alrt")
    start_time = time.time()
    raxml_command = [
        "/home/wiegerjs/iqtree-2.2.2.6-Linux/bin/iqtree2",
        "-m GTR+G",
        #"-m LG+G",
        "-s " + msa_filepath,
        #"-B 1000",
        "-T AUTO",
        "--threads-max 60",
        "-alrt 1000"
    ]

    # print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)

    subprocess.run(" ".join(raxml_command), shell=True)

    end_time = time.time()

    folder_path = os.path.join(os.pardir, "data/raw/msa")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    files_to_delete = [file for file in file_list if
                       ".log" in file]

    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)

    elapsed_time = end_time - start_time
    print("Elapsed time (seconds):", elapsed_time)
    alignment = AlignIO.read(msa_filepath, "fasta")

    num_sequences = len(alignment)

    alignment_length = alignment.get_alignment_length()

    print("Number of Sequences:", num_sequences)
    print("Length of Alignment:", alignment_length)

    data = {
        'dataset': tree_filename.replace(".newick", ""),
        'elapsed_time': [int(elapsed_time)],
        'num_seq': [num_sequences],
        'len': [alignment_length]
    }

    time_dat = pd.DataFrame(data)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "bootstrap_times_iqtree_alrt.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "bootstrap_times_iqtree_alrt.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "bootstrap_times_iqtree_alrt.csv"),
                        index=False,
                        mode='a', header=False)
