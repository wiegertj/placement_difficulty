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

    real_support = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_model.raxml.support"
    ebg_support = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_modeltest/{row['msa_name']}_modeltest_median_support_prediction.newick"

    import ete3
    import pandas as pd
    from sklearn.metrics import mean_absolute_error


    # Function to extract support values from a Newick file
    def extract_support_values(newick_file):
        tree = ete3.Tree(newick_file, format=1)  # format=1 assumes internal node support values
        return [node.support for node in tree.traverse() if not node.is_leaf()]


    # Path to your input files (replace with your actual paths)
    msa_name = "example"  # Replace this with actual msa_name as needed
    real_support_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_model.raxml.support"
    ebg_support_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_modeltest/{row['msa_name']}_modeltest_median_support_prediction.newick"
    print(real_support_path)
    print(ebg_support_path)
    # Extract support values from both trees
    real_support_values = extract_support_values(real_support_path)
    ebg_support_values = extract_support_values(ebg_support_path)

    # Compute mean absolute error
    mae = mean_absolute_error(real_support_values, ebg_support_values)

    print(f"Mean Absolute Error between real support and EBG support: {mae}")






