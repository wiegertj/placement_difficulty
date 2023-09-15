import math
import random
import types
import Bio
import pandas as pd
import os
import statistics
import numpy as np
from Bio import AlignIO
from Bio.Align import AlignInfo
from scipy.stats import skew, kurtosis
from collections import Counter
import os
import shutil
import dendropy
import ete3
import subprocess
import glob
import pandas as pd
import numpy as np
import types
import random
from Bio import AlignIO, SeqIO
from dendropy.calculate import treecompare

def calculate_site_entropies(msa_filepath):
    alignment = AlignIO.read(msa_filepath, 'fasta')

    # Get the number of sequences and the length of each sequence
    num_sequences = len(alignment)
    if num_sequences >= 500:
        return 0
    sequence_length = alignment.get_alignment_length()

    # Initialize an array to store site entropies
    site_entropies = np.zeros(sequence_length)

    # Iterate over each position in the alignment
    for position in range(sequence_length):
        # Get the residues (characters) at the current position for all sequences
        residues_at_position = alignment[:, position]
        # Count the occurrences of each residue at the current position
        residue_counts = Counter(residues_at_position)

        # Calculate the Shannon entropy for the current position
        entropy = 0.0
        for count in residue_counts.values():
            probability = count / num_sequences
            entropy -= probability * np.log2(probability)

        # Normalize the entropy to the range [0, 1]
        max_entropy = np.log2(len(residue_counts))
        normalized_entropy = entropy / max_entropy

        # Store the normalized entropy in the array
        site_entropies[position] = normalized_entropy

    return site_entropies

def filter_by_entropy_interval(msa_filepath, low_bound, up_bound):
    site_entropies = calculate_site_entropies(msa_filepath)
    try:
        if site_entropies == 0:
            return 0
    except ValueError:
        print("Processing ... ")
    alignment = AlignIO.read(msa_filepath, 'fasta')

    # Create a mask for sites within the specified entropy interval
    mask = (site_entropies >= low_bound) & (site_entropies <= up_bound)
    count_zeros = len(mask) - np.count_nonzero(mask)
    count_ones = np.count_nonzero(mask)

    # Print the counts
    print("Number of 0s:", count_zeros)
    print("Number of 1s:", count_ones)
    records = list(alignment)
    alignment_length = len(alignment[0])
    print(alignment_length)
    # Iterate over the records and remove the specified site from each sequence
    for record in records:
        for site in range(alignment_length):
            if mask[site] == 0:
                record.seq = record.seq[:5] + record.seq[5 + 1:]

    # Create a new alignment from the modified records
    modified_alignment = AlignIO.MultipleSeqAlignment(records)
    print(msa_filepath)
    print(mask)
    print(len(modified_alignment[0]))
    AlignIO.write(modified_alignment, msa_filepath.replace("_reference.fasta", "_filtered_" + str(int(10*low_bound)) + str(int(10*up_bound))), 'fasta')

    return msa_filepath.replace("_reference.fasta", "_filtered_" + str(int(10*low_bound)) + str(int(10*up_bound)))


if __name__ == '__main__':

    module_path = os.path.join(os.pardir, "configs/feature_config.py")

    feature_config = types.ModuleType('feature_config')
    feature_config.__file__ = module_path

    with open(module_path, 'rb') as module_file:
        code = compile(module_file.read(), module_path, 'exec')
        exec(code, feature_config.__dict__)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    # Create bins for the "difficult" column
    bins = [i / 10.0 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    loo_selection['difficult_bin'] = pd.cut(loo_selection['difficult'], bins=bins, right=False)

    # Initialize an empty DataFrame to store the sampled data
    sampled_data = pd.DataFrame()

    # Perform random sampling within each bin
    for bin_label, group in loo_selection.groupby('difficult_bin'):
        if len(group) >= 20:
            sampled_group = group.sample(20, random_state=42)
            sampled_data = pd.concat([sampled_data, sampled_group])
    sampled_data.reset_index(drop=True, inplace=True)
    print(sampled_data.head(10))
    print(sampled_data.tail(10))

    filenames = sampled_data['verbose_name'].str.replace(".phy", "_reference.fasta").tolist()

    for file in filenames:
        if not os.path.exists(os.path.join(os.pardir, "data/raw/msa", file)):
            print("Not found: " + file)
            filenames.remove(file)

    if feature_config.INCUDE_TARA_BV_NEO:
        filenames = filenames + ["bv_reference.fasta", "neotrop_reference.fasta", "tara_reference.fasta"]

    results = []
    counter = 0
    for file in filenames:
        counter += 1
        print(str(counter) + "/" + str(len(filenames)))
        filepath = os.path.join(os.pardir, "data/raw/msa", file)
        isAA = False
        datatype = loo_selection[loo_selection["verbose_name"] == file.replace("_reference.fasta", ".phy")].iloc[0][
            "data_type"]
        if datatype == "AA" or datatype == "DataType.AA":
            isAA = True

        filepath_modified = filter_by_entropy_interval(filepath, 0.1, 0.9)
        if filepath_modified == 0:
            print("Skipped, too large")
            continue
        print(file)
        print("Start raxml")

        command = ["raxml-ng", "--search", "--msa", filepath_modified, "--model",
                   "GTR+G", "tree", "pars{50}, rand{50}", "--redo"]
        print(command)

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print("RAxML-ng process completed successfully.")
                print("Output:")
                print(stdout)

            else:
                print("RAxML-ng process failed with an error.")
                print("Error Output:")
                print(stderr)
                continue

        except FileNotFoundError:
            print("RAxML-ng executable not found. Please make sure RAxML-ng is installed and in the system")


