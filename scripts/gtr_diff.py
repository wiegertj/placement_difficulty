import os
import subprocess

import pandas as pd

# Load the CSV file
file_path = "/hits/fast/cme/wiegerjs/model_test_res.csv"
df = pd.read_csv(file_path)

# Filter out rows where best_model_bic == 'error'
filtered_df = df[df['best_model_bic'] != 'error']
filtered_df = filtered_df[filtered_df['best_model_bic'] != 'GTR+G']
filtered_df = filtered_df[filtered_df['best_model_bic'] != 'GTR+G+I']
# Select the first 200 rows from the filtered DataFrame
sampled_df = filtered_df.head(200)

# Iterate over the sampled rows and print msa_name and best_model_bic
for index, row in sampled_df.iterrows():
    print(f"msa_name: {row['msa_name']}, best_model_bic: {row['best_model_bic']}")
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", f"{row['msa_name']}_reference.fasta")

    raxml_command = [
        "raxml-ng",
        "--search",
        f"--msa {msa_filepath}",
        f"--model {row['best_model_bic']}",
        # "--model LG4M",
        "--threads auto",
        f"--prefix {row['msa_name'] + '_modelfinder'}"
        # "--data-type AA"

    ]

    # print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)
