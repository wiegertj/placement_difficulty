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
    bootstraps_filepath = os.path.join(os.pardir, "data/raw/msa", row['msa_name'] + "_reference.fasta.raxml.bootstraps")

    raxml_command = [
        "ebg",
        f"-model {os.path.abspath(model_filepath_alt)}",
        f"-msa {os.path.abspath(msa_filepath)}",
        f"-tree {os.path.abspath(tree_filepath_alt)}",
        "-t b",
        f"-o {row['msa_name']}_modeltest"
    ]

    #print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)
    #raxml_command = [
     #   "raxml-ng",
      #  "--support",
     #   f"--tree {tree_filepath_alt}",
      #  f"--bs-trees {bootstraps_filepath}",
       # "--redo",
        #f"--prefix {row['msa_name']}_model"]
    #raxml_command = [
     #   "raxml-ng",
    #    "--bootstrap",
    #    f"--model {model_filepath_alt}",
    #    f"--bs-trees {1000}",
    #    f"--msa {msa_filepath}",
    #    "--redo",
    #]

    #subprocess.run(" ".join(raxml_command), shell=True)
    #/hits/fast/cme/wiegerjs/placement_difficulty/scripts/14534_25_model.raxml.support real support

