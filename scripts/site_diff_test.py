import os
import re
import shutil
import dendropy
import ete3
import subprocess
import glob
import pandas as pd
import numpy as np
import types
import random
from Bio import AlignIO, SeqIO, SeqRecord, Seq
from dendropy.calculate import treecompare

module_path = os.path.join(os.pardir, "configs/feature_config.py")
random.seed(42)
feature_config = types.ModuleType('feature_config')
feature_config.__file__ = module_path

with open(module_path, 'rb') as module_file:
    code = compile(module_file.read(), module_path, 'exec')
    exec(code, feature_config.__dict__)


def remove_gaps(sequence):
    return sequence.replace('-', '')


def calculate_bsd_aligned(tree1, tree2):
    branch_lengths1 = [branch.length for branch in tree1]
    branch_lengths2 = [branch.length for branch in tree2]

    score = np.sum(np.abs(branch_lengths1 - branch_lengths2)) / (np.sum(branch_lengths1) + np.sum(branch_lengths2))
    return score


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()

# print("Searching for already processed datasets ...")
# current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target/loo_result_entropy.csv"))
# dataset_set = set(current_loo_targets['dataset'])
# print("Before filterling" + str(len(filenames)))
# filtered_filenames = [filename for filename in filenames if filename not in dataset_set]
# print("After filterling" + str(len(filtered_filenames)))
loo_reest_samples = pd.read_csv(os.path.join(os.pardir, "data/diff_test/difficulty.csv"))
filenames = loo_reest_samples["dataset"]
filenames = set(filenames)
filtered_filenames = filenames
msa_counter = 0
print(len(filenames))

# existing = set(existing["dataset"])

# for dataset in existing["dataset"]:
#   samples = existing[existing["dataset"] == dataset]["sampleId"]
#  for sampleId in samples:


# filenames = filenames - existing
print(len(filenames))
filtered_filenames = filenames

for msa_name in filtered_filenames:
    results = []
    msa_counter += 1
    print(str(msa_counter) + "/" + str(len(filtered_filenames)))
    print(msa_name)
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    alignment = AlignIO.read(msa_filepath, "fasta")
    original_ids = [record.id for record in alignment]
    sequence_data = [list(record.seq) for record in alignment]
    alignment_array = np.array(sequence_data)

    raxml_path = subprocess.check_output(["which", "raxml-ng"], text=True).strip()
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

    last_float_before = extracted_value

    for x in range(30):
        random_number = random.randint(0, alignment_array.shape[1])
        print(alignment_array.shape)
        alignment_array = alignment_array.delete(alignment_array, random_number, axis=1)
        print(alignment_array.shape)
        seq_records = [SeqRecord.SeqRecord(Seq.Seq(''.join(seq)), id=original_ids[i], description="") for i, seq in
                       enumerate(alignment_array)]
        msa_new = AlignIO.MultipleSeqAlignment(seq_records)

        new_msa_path = os.path.join(os.pardir, "data/raw/msa/tmp_nomodel/",
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

        last_float_after = extracted_value_new

        results.append((random_number, last_float_before, last_float_after, last_float_before - last_float_after))

    df = pd.DataFrame(results, columns=["colId", "diff_before", "diff_after", "diff_diff"])
    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "site_diffs.csv")):
        df.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "site_diffs.csv")), index=False)
    else:
        df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "site_diffs.csv"),
                     index=False,
                     mode='a', header=False)
