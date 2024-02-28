import os
import subprocess

def list_foldernames(path):
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names


def gather_file_paths(base_path, folder_name):
    msa_path = os.path.abspath(os.path.join(base_path, folder_name, 'gtr_g_sim_msa.fasta'))
    best_tree_path = os.path.abspath(os.path.join(base_path, folder_name, 'gtr_g.raxml.bestTree'))
    best_model_path = os.path.abspath(os.path.join(base_path, folder_name, 'gtr_g.raxml.bestModel'))
    return msa_path, best_tree_path, best_model_path


# Specify the path
base_directory_path = "/hits/fast/cme/wiegerjs/EBG_simulations/data"

# Call the function to get the list of folder names
folders = list_foldernames(base_directory_path)
counter = 0
# Iterate over the folders
results = []
for folder in folders:
    counter += 1
    print(counter)
    # Get file paths
    msa_path, best_tree_path, best_model_path = gather_file_paths(base_directory_path, folder)

    # Check if files exist
    if not (os.path.exists(msa_path) and os.path.exists(best_tree_path) and os.path.exists(best_model_path)):
        print(f"Skipping {folder} - one or more files not found.")
        continue

    # Do something with the file paths, e.g., print them
    print(f"Folder: {folder}")
    print(f"MSA Path: {msa_path}")
    print(f"Best Tree Path: {best_tree_path}")
    print(f"Best Model Path: {best_model_path}")

    results.append((msa_path, best_tree_path, best_model_path))
import pandas as pd
df = pd.DataFrame(results, columns=["msa", "tree", "model"])
df.to_csv("simulation_paths.csv", index=False)