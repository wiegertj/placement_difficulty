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

    if last_float_before <= 50:
        continue

    for x in range(int(0.1 * alignment_array.shape[1]) - 1):
        print(int(0.1 * alignment_array.shape[1]) - 1)
        random_number = random.randint(0, alignment_array.shape[1])
        print(alignment_array.shape)
        alignment_array_tmp = np.delete(alignment_array, random_number, axis=1)
        print(alignment_array_tmp.shape)
        seq_records = [SeqRecord.SeqRecord(Seq.Seq(''.join(seq)), id=original_ids[i], description="") for i, seq in
                       enumerate(alignment_array_tmp)]
        msa_new = AlignIO.MultipleSeqAlignment(seq_records)

        new_msa_path = os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/raw/msa/tmp_nomodel/",
                                    msa_name + "_pars_tmp_" + str(random_number) + ".fasta")
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

        results.append((msa_name, random_number, last_float_before, last_float_after, last_float_before - last_float_after))

        if os.path.exists(new_msa_path):
            # Delete the file
            os.remove(new_msa_path)
            print(f"File {new_msa_path} deleted.")
        else:
            print(f"File {new_msa_path} does not exist.")

        df = pd.DataFrame(results, columns=["msa_name", "colId", "diff_before", "diff_after", "diff_diff"])
        if not os.path.isfile(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs.csv")):
            df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs.csv"), index=False)
        else:
            df.to_csv(os.path.join("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features",
                                 "site_diffs.csv"),
                         index=False,
                         mode='a', header=False)
        results = []
