from ete3 import Tree

def read_tree_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def calculate_rf_distance(file_path1, file_path2):
    tree1_str = read_tree_from_file(file_path1)
    tree2_str = read_tree_from_file(file_path2)

    tree1 = Tree(tree1_str, format=1)
    tree2 = Tree(tree2_str, format=1)
    results_distance = tree1.compare(tree2, unrooted=True)

    return results_distance["norm_rf"]

# Example usage:
folder_path = '/hits/fast/cme/wiegerjs/PANDIT/wdir_iq'

import os
def list_foldernames(path):
    folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folder_names


def gather_file_paths(base_path, folder_name):
    msa_path = os.path.join(base_path, folder_name, 'gtr_g_sim_msa.fasta')
    return msa_path
# Get a list of folder names in the specified path
# List all files in the folder
folder_path = '/hits/fast/cme/wiegerjs/EBG_simulations/iqtree_simulation_ufb'

# Get a list of folder names in the specified path
# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files ending with ".treefile"
tree_files = [file for file in all_files if file.endswith('.treefile')]
results = []
for folder_name in tree_files:
    abs_path = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace(".phy.treefile", "")

    abs_path_rx = "/hits/fast/cme/wiegerjs/EBG_simulations/raxml_inf_res/" + dataset + "_boot_test_.raxml.bestTree"
    abs_path_uf = os.path.abspath(os.path.join(folder_path, folder_name))

    rf_distance = calculate_rf_distance(abs_path_uf, abs_path_rx)
    print(rf_distance)
    results.append(rf_distance)

import csv

def write_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['nrf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({'nrf': result})

# Example usage:
output_file = "nrf_ours.csv"

write_results_to_csv(results, output_file)
print("Results have been written to", output_file)





