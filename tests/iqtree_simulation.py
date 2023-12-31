import os
import subprocess

def list_foldernames(path):
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names


def gather_file_paths(base_path, folder_name):
    msa_path = os.path.join(base_path, folder_name, 'gtr_g_sim_msa.fasta')
    return msa_path

base_directory_path = "/hits/fast/cme/wiegerjs/EBG_simulations/data"

# Call the function to get the list of folder names
folders = list_foldernames(base_directory_path)
counter = 0
# Iterate over the folders
for folder in folders:
    counter += 1
    print(counter)
    # Get file paths
    msa_path = gather_file_paths(base_directory_path, folder)

    best_tree_path = "/hits/fast/cme/wiegerjs/EBG_simulations/raxml_inf_res/" + folder + "_boot_test_.raxml.bestTree"
    best_model_path = "/hits/fast/cme/wiegerjs/EBG_simulations/raxml_inf_res/" + folder + "_boot_test_.raxml.bestModel"

    print(f"MSA Path: {msa_path}")
    print(f"Best Tree Path: {best_tree_path}")
    print(f"Best Model Path: {best_model_path}")

    # Check if files exist
    if not (os.path.exists(msa_path) and os.path.exists(best_tree_path) and os.path.exists(best_model_path)):
        print(f"Skipping {folder} - one or more files not found.")
        continue

    # Do something with the file paths, e.g., print them
    print(f"Folder: {folder}")
    print(f"MSA Path: {msa_path}")
    print(f"Best Tree Path: {best_tree_path}")
    print(f"Best Model Path: {best_model_path}")

    raxml_command = [
        "/home/wiegerjs/iqtree-2.2.2.6-Linux/bin/iqtree2",
        "-m GTR+G",
        #"-m LG+G",
        "-s " + msa_path,
        "-B 1000",
        "-T 1",
        #"--threads-max 60",
        #"-alrt 1000",
        f"-pre {folder}"
    ]
    #]

    # print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)
