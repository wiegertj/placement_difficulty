import subprocess
import pandas as pd
import os
from Bio import SeqIO, AlignIO, Seq, SeqRecord
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
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))
    if not os.path.exists(msa_filepath):
        print("Skipped, MSA not found")
        continue

    resampled_msas = []
    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")

    alignment = AlignIO.read(msa_filepath, "fasta")

    alignment_length = alignment.get_alignment_length()
    original_sites = list(range(1, alignment_length + 1))

    trees_path =  os.path.join(os.pardir, "data/raw/reference_tree/tmp", tree_filename.replace(".newick", "_pars_boot.txt"))

    # Perform resampling 100 times
    for x in range(100):
        sampled_lists_index = random.choices(original_sites, k=alignment_length)
        new_records = []
        for record in alignment:
            new_sequence = Seq.Seq('')
            for site_index in sampled_lists_index:
                new_sequence += record[site_index - 1]  # Adjust for 0-based indexing
            new_seq_record = SeqRecord.SeqRecord(new_sequence, id=record.id, description="")
            new_records.append(new_seq_record)
        msa_new = AlignIO.MultipleSeqAlignment(new_records)
        new_msa_path = os.path.join(os.pardir, "data/raw/msa/tmp/", tree_filename.replace(".newick", "") + "_pars_tmp_" + str(x) + ".fasta")
        output_prefix = tree_filename.split(".")[0] + "_parsimony_100temp_" + str(x)  # Using the filename as the prefix

        SeqIO.write(msa_new, os.path.join(os.pardir, "data/raw/msa/tmp/", tree_filename.replace(".newick", "") + "_pars_tmp_" + str(x) + ".fasta"), "fasta")
        print("Saved new bootstrap")

        raxml_command = [
            "raxml-ng",
            "--start",
            f"--model {model_path}",
            "--tree pars{1}",
            f"--msa {new_msa_path}",
            "--redo",
            f"--prefix {output_prefix}",
        ]
        print(raxml_command)
        subprocess.run(" ".join(raxml_command), shell=True)

        print(f"Parsimony Bootstrap analysis for {tree_filename}" + str(x) + "completed.")

        result_tree_path = os.path.join(os.pardir, "scripts", tree_filename.replace(".newick", "") + "_parsimony_100temp_" + str(x) + ".raxml.startTree")

        with open(result_tree_path, 'r') as tree_file:
            newick_tree = tree_file.read()

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
    files_to_delete = [file for file in file_list if ("_parsimony_100temp_" in file or ".log" in file or ".rba" in file)]

    # Delete the filtered files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)

    print(f"Deleted {len(files_to_delete)} files containing '_parsimony_100temp_' in their names.")
    output_prefix = tree_filename.split(".")[0] + "_parsimony_supp_" + str(x)  # Using the filename as the prefix
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {trees_path}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)
