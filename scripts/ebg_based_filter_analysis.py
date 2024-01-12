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
from Bio import AlignIO, SeqIO
from dendropy.calculate import treecompare

random.seed(200)

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
filenames = filenames
msa_counter = 0
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
results = []

for msa_name in filenames:
    msa_folder_path = os.path.join(base_directory, msa_name)
    try:
        ground_truth = pd.read_csv(
         "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter/" + msa_name + "_10_xxx/" + msa_name + "_10_xxx_features.csv")
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

    #print(filtered_subfolders)

    # Iterate through each filtered subfolder
    for subfolder in filtered_subfolders:

        # Construct the path to the subfolder
        subfolder_path = os.path.join(base_directory, subfolder, subfolder)

        # Read the CSV file in the subfolder into a DataFrame
        csv_files = [file for file in os.listdir(subfolder_path) if file.endswith(".csv")]

        # Assuming there's only one CSV file in each subfolder
        if len(csv_files) == 1:
            csv_file_path = os.path.join(subfolder_path, csv_files[0])

            df = pd.read_csv(csv_file_path)
            df["prediction_taxon"] = df["prediction_median"]
            df_merged = df.merge(ground_truth, on="branchId")
            df_merged["effect"] = df_merged["prediction_original"] - df_merged["prediction_taxon"]


            filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
            filepath = os.path.abspath(filepath)

            # Initialize variables for sequence count and sequence length
            sequence_count = 0
            sequence_length = 0

            # Iterate through the sequences in the FASTA file
            for record in SeqIO.parse(filepath, "fasta"):
                sequence_count += 1
                sequence_length = len(record.seq)

            results.append((1-sum(df_merged["prediction_taxon"])/sum(df_merged["prediction_original"]), msa_name, sequence_length, sequence_count))
import matplotlib.pyplot as plt

print(len(results))
results = [value for value in results if abs(value[0]) < 0.5]

df_res = pd.DataFrame(results, columns=["result", "msa_name", "sequence_length", "sequence_count"])


# Calculate the maximum value per unique "msa_name"
max_values = df_res.groupby('msa_name')['result'].max()

fraction_0_05 = (df_res.groupby('msa_name')['result'].max() >= 0.05).mean()


# Check if there is at least one row with 'result' >= 0.10 for each unique 'msa_name'
fraction_0_10 = (df_res.groupby('msa_name')['result'].max() >= 0.10).mean()

fraction_0_20 = (df_res.groupby('msa_name')['result'].max() >= 0.20).mean()


print(f"Fraction of msa's with result >= 0.05: {fraction_0_05:.2%}")
print(f"Fraction of msa's with result >= 0.10: {fraction_0_10:.2%}")
print(f"Fraction of msa's with result >= 0.20: {fraction_0_20:.2%}")


# Filter DataFrame for rows with result >= 0.10
filtered_df = df_res[df_res['result'] >= 0.10]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(f"Msa_name 10: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")


filtered_df = df_res[df_res['result'] >= 0.05]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(f"Msa_name 5: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")

filtered_df = df_res[df_res['result'] >= 0.2]
print(len(filtered_df["msa_name"].unique()))
print(len(df_res["msa_name"].unique()))

# Print sequence length and count values for filtered msa_names
for index, row in filtered_df.iterrows():
    print(f"Msa_name 2: {row['msa_name']}, Sequence Length: {row['sequence_length']}, Sequence Count: {row['sequence_count']}")


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

filtered_df = df_res[df_res["result"] >= 0.1]

# Count the occurrences of each unique msa_name
msa_counts = filtered_df["msa_name"].value_counts()

# Filter msa_names with 2 or more occurrences
msa_names_2_or_more = msa_counts[msa_counts >= 5]

# Calculate the percentage of unique msa_names with 2 or more rows
percentage_unique_msa_names = (len(msa_names_2_or_more) / len(df_res["msa_name"].unique()))

print(f"The percentage of unique msa_name values with 2 or more rows and result >= 1.05 is: {percentage_unique_msa_names:.2f}%")
