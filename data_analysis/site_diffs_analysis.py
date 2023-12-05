import os
from collections import Counter

import pandas as pd
from Bio import AlignIO
import numpy as np
msa_sel = pd.read_csv(os.path.join(os.path.pardir, "data/site_diff_selection.csv"))
msa_files = msa_sel["msa_name"].unique().tolist()
results = []
counter = 0
max_frac = []
min_frac = []
mean_frac = []
std_frac = []
for msa_file in msa_files:
    counter += 1
    print(counter)
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
    fraction_gap_list = []
    count_a_list = []
    count_c_list = []
    count_t_list = []
    count_g_list = []

    num_sequences, alignment_length = alignment_array.shape

    for column in alignment_array.T:
        counts = Counter(column)
        keys = len(set(column))
        total_count = sum(counts.values())

        column_upper = np.char.upper(column.astype(str))

        # Create a Counter object
        counts = Counter(column_upper)

        # Extract counts for A, C, T, and G
        count_A = counts['A']
        count_C = counts['C']
        count_T = counts['T']
        count_G = counts['G']

        count_a_list.append(count_A)
        count_c_list.append(count_C)
        count_t_list.append(count_T)
        count_g_list.append(count_G)



        # Calculate probabilities and store in a list
        probabilities = [count / total_count for count in counts.values()]
        probabilities_array = np.array(probabilities) + 1e-10
        max_frac.append(max(probabilities))
        min_frac.append(min(probabilities))
        mean_frac.append(np.mean(probabilities))
        std_frac.append(np.std(probabilities))
        entropy = -np.sum(probabilities * np.log2(probabilities_array ))  # Add a small constant to avoid log(0)
        #max_entropy = np.log2(keys)  # Maximum entropy for a column with 4 nucleotides

        column_entropies.append(entropy)# / max_entropy)

        char_counts = Counter(column)
        total_chars = len(column)
        fractions = probabilities_array
        fractions_std = np.std(fractions)
        fractions_min = np.min(fractions)
        fractions_max = np.max(fractions)
        fractions_mean = np.mean(fractions)
        fractions_std_list.append(fractions_std)
        fractions_max_list.append(fractions_max)
        fractions_mean_list.append(fractions_mean)
        fractions_min_list.append(fractions_min)
        fraction_gap_list.append(len(column) - len(" ".join(column.tolist()).replace("-", "")))

    # Normalize the column entropies

    from scipy.stats import skew, kurtosis

    # Calculate statistics of normalized entropies
    min_entropy = np.min(column_entropies)
    max_entropy = np.max(column_entropies)
    skewness_entropy = skew(column_entropies)
    kurt_entropy = kurtosis(column_entropies)
    std_dev_entropy = np.std(column_entropies)
    mean_entropy = np.mean(column_entropies)

    for id in col_id:
        col_entropy = column_entropies[id]
        diff_diff = msa_sel[(msa_sel["msa_name"] == msa_file) & (msa_sel["colId"] == id)]["diff_diff"].values[0]
        diff_before = msa_sel[(msa_sel["msa_name"] == msa_file) & (msa_sel["colId"] == id)]["diff_before"].values[0]

        data = {
            "msa_name": msa_file,
            "id": id,
            "diff_before": diff_before,
            "diff_diff": diff_diff,
            "col_entropy": col_entropy,
            "col_min_prob": fractions_min_list[id],
            "col_max_prob": fractions_max_list[id],
            "col_mean_prob": fractions_mean_list[id],
            "col_std_prob": fractions_std_list[id],
            "gap_frac": fraction_gap_list[id],
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

msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features",
                                        "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
msa_features = msa_features.drop_duplicates(subset=['dataset'], keep='first')

df_res = df_res.merge(msa_features, left_on=["msa_name"], right_on=["dataset"], how="inner")
print(df_res.shape)
df_res.to_csv("site_diff_features.csv")