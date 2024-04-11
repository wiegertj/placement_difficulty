import os

import subprocess
import pandas as pd
import numpy as np
import types
import random
from Bio import AlignIO, SeqIO, SeqRecord, Seq


def list_foldernames(path):
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names


def gather_file_paths(base_path, folder_name):
    msa_path = os.path.join(base_path, folder_name, 'gtr_g_sim_msa.fasta')
    return msa_path

folder_path = '/hits/fast/cme/wiegerjs/PANDIT/data_pandit'

# Get a list of folder names in the specified path

# Call the function to get the list of folder names
folders = list_foldernames(folder_path)
counter = 0
# Iterate over the folders
for folder in folders:
    counter += 1
    print(counter)
    # Get file paths
    msa_path = f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{folder}/data.{folder}_converted.fasta"

    print(f"MSA Path: {msa_path}")




    # Do something with the file paths, e.g., print them
    print(f"Folder: {folder}")
    print(f"MSA Path: {msa_path}")


    command = ["pythia", "--msa", os.path.abspath(msa_path),
               "--raxmlng", "raxml-ng",
               "--removeDuplicates"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(command)
    pythia_output = result.stdout
    pythia_output = result.stdout
    pythia_error = result.stderr  # Capture stderr

    is_index = result.stderr.find("is: ")
    if is_index != -1:
        value_string = result.stderr[is_index + 4:is_index + 8]  # Add 4 to skip "is: "
        extracted_value = float(value_string)
    else:
        print("No match found")
        continue

    last_float_before = extracted_value

    results = []
    results.append((folder, last_float_before))

    df = pd.DataFrame(results, columns=["msa_name", "diff"])
    if not os.path.isfile(
            os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                         "simu_diffs_pandit.csv")):
        df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                               "simu_diffs_pandit.csv"), index=False)
    else:
        df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                               "simu_diffs_pandit.csv"),
                  index=False,
                  mode='a', header=False)
    results = []