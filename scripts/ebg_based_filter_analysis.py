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
filenames = filenames[:100]
msa_counter = 0
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
results = []

for msa_name in filenames:
    msa_folder_path = os.path.join(base_directory, msa_name)
    try:
        ground_truth = pd.read_csv(
         "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter/" + msa_name + "/" + msa_name + "_features.csv")
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
        print(subfolder_path)

        # Read the CSV file in the subfolder into a DataFrame
        csv_files = [file for file in os.listdir(subfolder_path) if file.endswith(".csv")]

        # Assuming there's only one CSV file in each subfolder
        if len(csv_files) == 1:
            csv_file_path = os.path.join(subfolder_path, csv_files[0])

            df = pd.read_csv(csv_file_path)
            df["prediction_taxon"] = df["prediction_median"]
            df_merged = df.merge(ground_truth, on="branchId")
            df_merged["effect"] = df_merged["prediction_original"] - df_merged["prediction_taxon"]
            print("#"*10)
            print(subfolder)
            results.append((1-sum(df_merged["prediction_taxon"])/sum(df_merged["prediction_original"]), msa_name))
            print(1-sum(df_merged["prediction_taxon"])/sum(df_merged["prediction_original"]))
            print("#"*10)
import matplotlib.pyplot as plt

print(len(results))
results = [value for value in results if abs(value[0]) < 0.5]

df_res = pd.DataFrame(results, columns=["result", "msa_name"])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Set the style for better visualization
sns.set(style="whitegrid")

# Create a single plot with histograms for each unique value in the "msa_name" column
sns.histplot(data=df_res, x="result", hue="msa_name", multiple="stack", bins=20, palette="viridis", edgecolor='black')

# Customize the plot
plt.title('Histograms by msa_name')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Save the plot as a PNG file
plt.savefig('HISTOGRAMS_BY_MSA_NAME.png')

# Display the plot
plt.show()
