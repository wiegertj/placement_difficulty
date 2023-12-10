import os

import subprocess
import pandas as pd
import numpy as np
import types
import random
from Bio import AlignIO, SeqIO, SeqRecord, Seq





def remove_gaps(sequence):
    return sequence.replace('-', '')


def calculate_bsd_aligned(tree1, tree2):
    branch_lengths1 = [branch.length for branch in tree1]
    branch_lengths2 = [branch.length for branch in tree2]

    score = np.sum(np.abs(branch_lengths1 - branch_lengths2)) / (np.sum(branch_lengths1) + np.sum(branch_lengths2))
    return score


loo_selection = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/loo_selection.csv")
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
already = pd.read_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs.csv"))
already = already["msa_name"].unique().tolist()

filtered_filenames = filenames
msa_counter = 0
print(len(filenames))

for msa_name in filtered_filenames:
    msa_counter += 1

    if msa_name in already:
        continue
    results = []
    print(str(msa_counter) + "/" + str(len(filtered_filenames)))
    print(msa_name)
    msa_filepath = os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/raw/msa", msa_name + "_reference.fasta")
    alignment = AlignIO.read(msa_filepath, "fasta")
    original_ids = [record.id for record in alignment]
    sequence_data = [list(record.seq) for record in alignment]
    alignment_array = np.array(sequence_data)

    raxml_path = "/home/wiegerjs/bin/raxml-ng-mpi"
    command = ["pythia", "--msa", os.path.abspath(msa_filepath),
               "--raxmlng", raxml_path]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
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

    #if last_float_before <= 0.4:
     #   continue

    for x in [(0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.45),
              (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8),
              (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)
              ]:
        lower_bound = x[0]
        upper_bound = x[1]
        print(int(0.1 * alignment_array.shape[1]) - 1)
        from scipy.stats import entropy

        # Calculate normalized entropy for each column
        column_entropies = np.apply_along_axis(lambda col: entropy(col) / np.log(col.shape[0]), axis=0,
                                               arr=alignment_array)

        # Find columns with normalized entropy between 0.95 and 1
        selected_columns = np.where((column_entropies >= lower_bound) & (column_entropies <= upper_bound))[0]

        # Print selected columns (optional)
        print("Selected Columns:", selected_columns)

        # Delete selected columns
        alignment_array_tmp = np.delete(alignment_array, selected_columns, axis=1)

        # Print the modified array shape
        print("Original Shape:", alignment_array.shape)
        print("Modified Shape:", alignment_array_tmp.shape)

        print(alignment_array_tmp.shape)
        seq_records = [SeqRecord.SeqRecord(Seq.Seq(''.join(seq)), id=original_ids[i], description="") for i, seq in
                       enumerate(alignment_array_tmp)]
        msa_new = AlignIO.MultipleSeqAlignment(seq_records)

        new_msa_path = os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/raw/msa/tmp_nomodel/",
                                    msa_name + "_pars_tmp_" + str(lower_bound) + str(upper_bound) + ".fasta")
        SeqIO.write(msa_new, os.path.abspath(new_msa_path),
                    "fasta")

        command = ["pythia", "--msa", os.path.abspath(new_msa_path),
                   "--raxmlng", raxml_path]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        pythia_output = result.stdout
        pythia_output = result.stdout
        pythia_error = result.stderr  # Capture stderr

        is_index = result.stderr.find("is: ")
        if is_index != -1:
            value_string = result.stderr[is_index + 4:is_index + 8]  # Add 4 to skip "is: "
            extracted_value_new = float(value_string)
        else:
            print("No match found")
            continue

        last_float_after = extracted_value_new

        results.append((msa_name, lower_bound, upper_bound, len(selected_columns),last_float_before, last_float_after, last_float_before - last_float_after))

        if os.path.exists(new_msa_path):
            # Delete the file
            os.remove(new_msa_path)
            print(f"File {new_msa_path} deleted.")
        else:
            print(f"File {new_msa_path} does not exist.")

        df = pd.DataFrame(results, columns=["msa_name", "lower_bound", "upper_bound", "deleted_cols", "diff_before", "diff_after", "diff_diff"])
        if not os.path.isfile(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs_ent.csv")):
            df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs_ent.csv"), index=False)
        else:
            df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs_ent.csv"),
                         index=False,
                         mode='a', header=False)
        results = []
