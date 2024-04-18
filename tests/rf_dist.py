from ete3 import Tree

def read_tree_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def calculate_rf_distance(file_path1, file_path2):
    tree1_str = read_tree_from_file(file_path1)
    tree2_str = read_tree_from_file(file_path2)

    tree1 = Tree(tree1_str)
    tree2 = Tree(tree2_str)

    rf_distance = tree1.robinson_foulds(tree2)[0]
    return rf_distance

# Example usage:
folder_path = '/hits/fast/cme/wiegerjs/PANDIT/wdir_iq'

import os

# Get a list of folder names in the specified path
# List all files in the folder
all_files = os.listdir(folder_path)

# Filter files ending with ".treefile"
tree_files = [file for file in all_files if file.endswith('.treefile')]
results = []
for folder_name in tree_files:
    abs_path_uf = os.path.abspath(os.path.join(folder_path, folder_name))
    dataset = folder_name.replace(".treefile", "")
    abs_path_rx = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{dataset}_pandit_inference_.raxml.bestTree"


    rf_distance = calculate_rf_distance(abs_path_uf, abs_path_rx)
    print(rf_distance)
    results.append(rf_distance)




