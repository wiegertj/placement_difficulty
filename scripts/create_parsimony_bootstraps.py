import subprocess
import time

import pandas as pd
import os
import numpy as np
from Bio import SeqIO, AlignIO, Seq, SeqRecord, Phylo
import random
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)

results = []
counter = 0
for tree_filename in filenames:
    start_time = time.time()

    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))
    if not os.path.exists(msa_filepath):
        print("Skipped, MSA not found")
        continue

    if os.path.exists(os.path.join(os.pardir, "scripts/") + tree_filename.replace(".newick", "") + "_parsimony_supp199nomodel_.raxml.support"):
        print("Found already, move on")
        continue

    if tree_filename == "20632_1.newick":
        continue

    resampled_msas = []
    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")

    alignment = AlignIO.read(msa_filepath, "fasta")

    sequence_data = [list(record.seq) for record in alignment]

    # Convert the sequence data into a NumPy array
    alignment_array = np.array(sequence_data)
    no_col = alignment.get_alignment_length()
    original_ids = [record.id for record in alignment]

    alignment_length = alignment.get_alignment_length()
    original_sites = list(range(1, alignment_length + 1))

    trees_path = os.path.join(os.pardir, "data/raw/reference_tree/tmp_nomodel",
                              tree_filename.replace(".newick", "_pars_boot.txt"))


    for x in range(200):
        # Initialize an empty array for each replicate
        #replicate_alignment = np.empty((alignment_array.shape[0], alignment_array.shape[1]), dtype=alignment_array.dtype)
        sampled_columns = np.random.choice(alignment_array.shape[1], size=alignment_array.shape[1], replace=True)
        replicate_alignment = alignment_array[:, sampled_columns]

        seq_records = [SeqRecord.SeqRecord(Seq.Seq(''.join(seq)), id=original_ids[i], description="") for i, seq in
                       enumerate(replicate_alignment)]

        # Create a MultipleSeqAlignment object from the SeqRecords
        msa_new = AlignIO.MultipleSeqAlignment(seq_records)

        new_msa_path = os.path.join(os.pardir, "data/raw/msa/tmp_nomodel/", tree_filename.replace(".newick", "") + "_pars_tmp_" + str(x) + ".fasta")
        output_prefix = tree_filename.split(".")[0] + "_parsimony_199tempnomodel_" + str(x)  # Using the filename as the prefix




















        SeqIO.write(msa_new, os.path.join(os.pardir, "data/raw/msa/tmp_nomodel/", tree_filename.replace(".newick", "") + "_pars_tmp_" + str(x) + ".fasta"), "fasta")

        raxml_command = [
            "raxml-ng",
            "--start",
            f"--model GTR+G",
            "--tree pars{1}",
            f"--msa {new_msa_path}",
            "--redo",
            "--log ERROR",
            f"--prefix {output_prefix}",
        ]
        subprocess.run(" ".join(raxml_command), shell=True)


        result_tree_path = os.path.join(os.pardir, "scripts", tree_filename.replace(".newick", "") + "_parsimony_199tempnomodel_" + str(x) + ".raxml.startTree")

        try:
            with open(result_tree_path, 'r') as tree_file:
                newick_tree = tree_file.read()
        except FileNotFoundError:
            continue

        # Delete result_tree_path and new_msa_filepath
        os.remove(new_msa_path)


        # Append the Newick tree to the trees file
        with open(trees_path, 'a') as trees_file:
            # If the file doesn't exist, create it and add the tree
            if not os.path.exists(trees_path):
                trees_file.write(newick_tree)
            else:
                # Append the tree
                trees_file.write(newick_tree)

    folder_path = os.path.join(os.pardir, "scripts")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files that contain "_parsimony_100temp_" in their names
    files_to_delete = [file for file in file_list if (("_parsimony_199tempnomodel_" in file) or (".log" in file) or (".rba" in file) or ("reduced" in file))]

    # Delete the filtered files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)


    print(f"Deleted {len(files_to_delete)} files containing '_parsimony_199tempnomodel_' in their names.")
    output_prefix = tree_filename.split(".")[0] + "_parsimony_supp199nomodel_" + str(x)  # Using the filename as the prefix
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {trees_path}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time (seconds):", elapsed_time)
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
                             "pars_boot_times199_nomodel.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "pars_boot_times199_nomodel.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "pars_boot_times199_nomodel.csv"),
                     index=False,
                     mode='a', header=False)



