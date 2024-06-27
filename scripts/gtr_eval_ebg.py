import os
import subprocess
from ete3 import Tree
import pandas as pd

# Load the CSV file
file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts/nrf_results_filtered.csv"
df = pd.read_csv(file_path)

# Iterate over the sampled rows and print msa_name and best_model_bic
results = []
for index, row in df.iterrows():
    tree_filepath_alt = os.path.join(os.pardir, "scripts/", f"{row['msa_name']}_modelfinder.raxml.bestTree")
    model_filepath_alt = os.path.join(os.pardir, "scripts/", f"{row['msa_name']}_modelfinder.raxml.bestModel")
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", row['msa_name'] + "_reference.fasta")

    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_filepath_alt}",
        f"--bs-trees {1000}",
        f"--msa {msa_filepath}",
        "--redo",
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    #raxml_command = [
     #   "raxml-ng",
      #  "--support",
       # f"--tree {tree_filepath_alt}",
       # f"--bs-trees {boot_path}",
       # "--redo",
       # f"--prefix {row['msa_name']}"]