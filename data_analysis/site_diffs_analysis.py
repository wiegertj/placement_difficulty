import os
from collections import Counter

import pandas as pd
from Bio import AlignIO
import numpy as np
msa_sel = pd.read_csv(os.path.join(os.path.pardir, "data/site_diff_selection.csv"))
msa_files = msa_sel["msa_name"].values.tolist()
results = []
for msa_file in msa_files:
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", msa_file + "_reference.fasta")

    alignment = AlignIO.read(msa_filepath, 'fasta')
    alignment_array = np.array([list(rec.seq) for rec in alignment])
    col_id = msa_sel[msa_sel["msa_name"] == msa_file]["colId"].values.tolist()

    # Calculate the column entropies
    column_entropies = []
    fractions_std_list = []
    fractions_min_list = []
    fractions_max_list = []
    fractions_mean_list = []
    num_sequences, alignment_length = alignment_array.shape

    for column in alignment_array.T:
        counts = Counter(column)
        print(counts)
        total_count = sum(counts.values())

        # Calculate probabilities and store in a list
        probabilities = [count / total_count for count in counts.values()]
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add a small constant to avoid log(0)
        column_entropies.append(entropy)

        char_counts = Counter(column)
        total_chars = len(column)
        fractions = {char: count / total_chars for char, count in char_counts.items()}

        fractions_std = np.std(fractions)
        fractions_min = np.min(fractions)
        fractions_max = np.max(fractions)
        fractions_mean = np.mean(fractions)
        fractions_std_list.append(fractions_std)
        fractions_max_list.append(fractions_max)
        fractions_mean_list.append(fractions_mean)
        fractions_min_list.append(fractions_min)

    # Normalize the column entropies
    max_entropy = np.log2(4)  # Maximum entropy for a column with 4 nucleotides
    normalized_entropies = np.array(column_entropies) / max_entropy

    from scipy.stats import skew, kurtosis

    # Calculate statistics of normalized entropies
    min_entropy = np.min(normalized_entropies)
    max_entropy = np.max(normalized_entropies)
    skewness_entropy = skew(normalized_entropies)
    kurt_entropy = kurtosis(normalized_entropies)
    std_dev_entropy = np.std(normalized_entropies)
    mean_entropy = np.mean(normalized_entropies)

    for id in col_id:
        col_entropy = column_entropies[id]
        data = {
            "msa_name": msa_file,
            "id": id,
            "col_entropy": col_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "skewness_entropy": skewness_entropy,
            "kurt_entropy": kurt_entropy,
            "std_dev_entropy": std_dev_entropy,
            "mean_entropy": mean_entropy,
            "num_seq": num_sequences,
            "len": alignment_length,
            "fractions_std_list_mean": np.mean(fractions_std_list),
            "fractions_std_list_min": np.min(fractions_std_list),
            "fractions_std_list_max": np.max(fractions_std_list),
            "fractions_std_list_std": np.std(fractions_std_list),
            "fractions_mean_list_mean": np.mean(fractions_mean_list),
            "fractions_mean_list_min": np.min(fractions_mean_list),
            "fractions_mean_list_max": np.max(fractions_mean_list),
            "fractions_mean_list_std": np.std(fractions_mean_list),
            "fractions_max_list_mean": np.mean(fractions_max_list),
            "fractions_max_list_min": np.min(fractions_max_list),
            "fractions_max_list_max": np.max(fractions_max_list),
            "fractions_max_list_std": np.std(fractions_max_list),
            "fractions_min_list_mean": np.mean(fractions_min_list),
            "fractions_min_list_min": np.min(fractions_min_list),
            "fractions_min_list_max": np.max(fractions_min_list),
            "fractions_min_list_std": np.std(fractions_min_list),
        }
        results.append(data)
df_res = pd.DataFrame(results)
df_res.to_csv("site_diff_features.csv")