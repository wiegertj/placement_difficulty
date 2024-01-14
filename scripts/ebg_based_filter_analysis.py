import os
import re
import shutil
from statistics import mean

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
from scipy.stats import skew, kurtosis

random.seed(200)

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
filenames = filenames
msa_counter = 0
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
results = []
res_list = []
results_filtered = []

for msa_name in filenames[:180]:
    msa_folder_path = os.path.join(base_directory, msa_name)
    try:
        ground_truth = pd.read_csv(
            "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter/" + msa_name + "_10_xxx/" + msa_name + "_10_xxx_features.csv")
        newick_tree_path = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter/" + msa_name + "_10_xxx/" + msa_name + "_10_xxx_median_support_prediction.newick"
        print(newick_tree_path)
        ground_truth_tree = ete3.Tree(newick_tree_path, format=0)

        print("found first")
    except FileNotFoundError:
        try:
            ground_truth = pd.read_csv(
                "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter/" + msa_name + "/" + msa_name + "_features.csv")
            print("found second")

        except FileNotFoundError:
            continue
    ground_truth["prediction_original"] = ground_truth["prediction_median"]

    all_subfolders = [folder for folder in os.listdir(base_directory) if
                      os.path.isdir(os.path.join(base_directory, folder))]

    # Filter subfolders based on msa_name and "taxon"
    filtered_subfolders = [folder for folder in all_subfolders if folder.startswith(msa_name) and "taxon" in folder]

    # print(filtered_subfolders)

    # Iterate through each filtered subfolder
    for subfolder in filtered_subfolders:
        # Construct the path to the subfolder
        subfolder_path = os.path.join(base_directory, subfolder, subfolder)

        # Read the CSV file in the subfolder into a DataFrame
        last_integer = int(re.search(r'\d+', subfolder[::-1]).group()[::-1])
        try:
            newick_tree_tmp = ete3.Tree(os.path.join(subfolder_path, f"{subfolder}_median_support_prediction.newick"), format=0)
        except ete3.parser.newick.NewickError:
            print("failed")
            continue

        try:
            newick_tree_tmp_lower = ete3.Tree(os.path.join(subfolder_path, f"{subfolder}_lower5_support_prediction.newick"),
                                        format=0)
        except ete3.parser.newick.NewickError:
            print("failed")
            continue

        newick_tree_original_copy = ground_truth_tree.copy()
        try:
            leaf_node = newick_tree_original_copy.search_nodes(name="taxon" + str(last_integer))[0]
        except IndexError:
            print("indexerror")
            continue
        leaf_node.delete()

        sum_support_original_copy = 0.0
        sum_support_tmp = 0.0
        uncertainty = 0.0
        max_uncertain = 0.0

        sum_support_filter_list = []
        sum_support_unfilter_list = []

        # Sum up the support values for newick_tree_original_copy
        for node in newick_tree_original_copy.traverse():
            if node.support is not None and not node.is_leaf():
                sum_support_original_copy += node.support
                sum_support_unfilter_list.append(node.support)
                for node_lower in newick_tree_tmp_lower.traverse():
                    if node_lower.support is not None and node_lower.name == node.name:
                        max_uncertain += 100
                        uncertainty += abs(node_lower.support - node.support)


        # Sum up the support values for newick_tree_tmp
        for node in newick_tree_tmp.traverse():
            if node.support is not None and not node.is_leaf():
                sum_support_filter_list.append(node.support)
                sum_support_tmp += node.support

        # Assuming there's only one CSV file in each subfolder
        elementwise_difference = [a - b for a, b in zip(sum_support_filter_list, sum_support_unfilter_list)]

        filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
        filepath = os.path.abspath(filepath)

        # Initialize variables for sequence count and sequence length
        sequence_count = 0
        sequence_length = 0

        # Iterate through the sequences in the FASTA file
        for record in SeqIO.parse(filepath, "fasta"):
            sequence_count += 1
            sequence_length = len(record.seq)

        print(sum_support_tmp)
        print(sum_support_original_copy)
        if (sum_support_tmp / sum_support_original_copy) > 1.08:
            results_filtered.append((sum_support_tmp / sum_support_original_copy, msa_name, "taxon" + str(last_integer), sequence_count, sequence_length, uncertainty, max_uncertain,
                                     min(elementwise_difference), max(elementwise_difference), mean(elementwise_difference), np.std(elementwise_difference), skew(elementwise_difference), kurtosis(elementwise_difference)))
        else:
            results_filtered.append((sum_support_tmp / sum_support_original_copy, msa_name, "taxon" + str(last_integer), sequence_count, sequence_length, uncertainty, max_uncertain,
                                     min(elementwise_difference), max(elementwise_difference), mean(elementwise_difference),np.std(elementwise_difference), skew(elementwise_difference), kurtosis(elementwise_difference)))

        results.append((sum_support_tmp / sum_support_original_copy, msa_name, "taxon" + str(last_integer),
                                 sequence_count, sequence_length, uncertainty, max_uncertain,
                                 min(elementwise_difference), max(elementwise_difference), mean(elementwise_difference),
                                 np.std(elementwise_difference), skew(elementwise_difference),
                                 kurtosis(elementwise_difference)))

import matplotlib.pyplot as plt

print(len(results))
# results = [value for value in results if abs(value[0]) < 0.5]

df_res = pd.DataFrame(results, columns=["result", "msa_name", "sequence_length", "sequence_count","uncertainty", "max_uncertainty", "min_1", "max_1","mean_1","std_1", "skew_1", "kurt_1"])

# Calculate the maximum value per unique "msa_name"
max_values = df_res.groupby('msa_name')['result'].max()

fraction_0_05 = (df_res.groupby('msa_name')['result'].max() >= 1.05).mean()

# Check if there is at least one row with 'result' >= 0.10 for each unique 'msa_name'
fraction_0_10 = (df_res.groupby('msa_name')['result'].max() >= 1.10).mean()
fraction_0_08 = (df_res.groupby('msa_name')['result'].max() >= 1.08).mean()

fraction_0_20 = (df_res.groupby('msa_name')['result'].max() >= 1.20).mean()

print(f"Fraction of msa's with result >= 0.05: {fraction_0_05:.2%}")
print(f"Fraction of msa's with result >= 0.10: {fraction_0_10:.2%}")
print(f"Fraction of msa's with result >= 0.08: {fraction_0_08:.2%}")

print(f"Fraction of msa's with result >= 0.20: {fraction_0_20:.2%}")

# Filter DataFrame for rows with result >= 0.10
filtered_df = df_res[df_res['result'] >= 1.10]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(
        f"Msa_name 10: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")

filtered_df = df_res[df_res['result'] >= 1.05]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(
        f"Msa_name 5: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")

filtered_df = df_res[df_res['result'] >= 1.2]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(
        f"Msa_name 2: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")

# Create a histogram of the maximum values
plt.hist(max_values, bins=20, color='blue', edgecolor='black')

# Customize the plot
plt.title('Histogram of Max Values by msa_name')
plt.xlabel('Max Values')
plt.ylabel('Frequency')

# Save the plot as a PNG file
plt.savefig('MAX_VALUES_HISTOGRAM_BY_MSA_NAME.png')

# Display the plot
plt.show()

filtered_df = df_res[df_res["result"] >= 1.1]

# Count the occurrences of each unique msa_name
msa_counts = filtered_df["msa_name"].value_counts()

# Filter msa_names with 2 or more occurrences
msa_names_2_or_more = msa_counts[msa_counts >= 3]

# Calculate the percentage of unique msa_names with 2 or more rows
percentage_unique_msa_names = (len(msa_names_2_or_more) / len(df_res["msa_name"].unique()))

print(
    f"The percentage of unique msa_name values with 2 or more rows and result >= 1.05 is: {percentage_unique_msa_names:.2f}%")

df_res_filtered = pd.DataFrame(results_filtered, columns=["effect", "msa_name", "taxon", "sequence_count","sequence_length", "uncertainty_pred", "max_uncertainty",  "min_1", "max_1","mean_1","std_1", "skew_1", "kurt_1"])

df_res_filtered.to_csv("filtered_ebg_test.csv")