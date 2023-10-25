import os
import shutil

import pandas as pd

import os
import subprocess

# Specify the path to the directory containing your folders
list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter=0
# Loop over each subdirectory (folder) within the specified path
for folder_name in list_foldernames:
    folder_path = os.path.abspath(os.path.join(os.pardir, "data/raxml_data",str(folder_name)))
    print(folder_path)
    counter += 1
    print(counter)
    if os.path.isdir(folder_path):
        tree_path = os.path.join(folder_path, str(folder_name) + ".raxml.bestTree")
        if os.path.exists(tree_path):
            counter += 1
            print(counter)
            model_path = os.path.abspath(os.pardir + "/data/raxml_data/" + str(folder_name) + "/" + str(folder_name) + ".raxml.bestModel")
            model_path_to = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/loo/" + str(folder_name) + "_msa_model.txt"
            msa_file_path = os.path.abspath(os.path.join(folder_path, 'msa.fasta'))
            msa_file_to = os.path.join(os.pardir, '/data/raw/msa/' + str(folder_name) + '_reference.fasta')
            tree_path = os.path.join(folder_path, str(folder_name) + ".raxml.bestTree")
            tree_path_to = os.path.join(os.pardir, '/data/raw/reference_tree/' + str(folder_name) + '.newick')

            shutil.copy(model_path, model_path_to)
            shutil.copy(msa_file_path, msa_file_to)
            shutil.copy(tree_path, tree_path_to)