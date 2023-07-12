import shutil
import tarfile
import os
import pandas as pd
import numpy as np

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulty_ranges = np.arange(0.1, 1.1, 0.1)

samples = []
for i in range(len(difficulty_ranges) - 1):
    lower_bound = difficulty_ranges[i]
    upper_bound = difficulty_ranges[i + 1]
    subset = difficulties_df[(difficulties_df['difficult'] >= lower_bound) & (difficulties_df['difficult'] < upper_bound) & (difficulties_df['data_type'] != 'AA')].sample(20)
    samples.append(subset)

result = pd.concat(samples)
result.to_csv(os.path.join(os.pardir, "data/loo_selection.csv"))

# Create reference MSA
for file in result["verbose_name"]:
    file_path = os.path.join(os.pardir, "data/TreeBASEMirror-main/trees/" + file)
    tar_file = os.path.join(file_path, file + ".tar.gz") # path of msa tar file
    print(file)

    # Open the tar.gz file
    with tarfile.open(tar_file, 'r:gz') as tar:
        # Extract the contents of the tar.gz file to the target directory
        tar.extractall(file_path)

    # Rename the extracted directory or file (if needed)
    extracted_path = os.path.join(file_path,
                                  'msa.fasta')  # Replace 'original_extracted_name' with the actual extracted name
    new_name_msa = os.path.join(file_path, file.replace(".phy","_reference.fasta"))  # Replace 'new_name' with the desired new name

    os.rename(extracted_path, new_name_msa)

    copy_to_path = os.path.join(os.pardir, "data/raw/msa")
    shutil.copy(new_name_msa, copy_to_path) # Copy to reference

    copy_to_path = os.path.join(os.pardir, "data/raw/query")
    shutil.copy(new_name_msa, copy_to_path)  # Copy to query
    os.rename(copy_to_path + "/" + file.replace(".phy","_reference.fasta"), copy_to_path + "/" + file.replace(".phy","_query.fasta"))

    #----------------------------- COPY tree-------------------------------------------

    tree_path = os.path.join(file_path, "tree_best.newick")
    new_tree_name = os.path.join(file_path, file.replace(".phy", ".newick"))  # Replace 'new_name' with the desired new name
    os.rename(tree_path, new_tree_name)
    copy_to_path_tree = os.path.join(os.pardir, "data/raw/reference_tree")
    shutil.copy(new_tree_name, copy_to_path_tree)  # Copy to query

    #----------------------------- COPY model -------------------------------------------

    model_path = os.path.join(file_path, "model_0.txt")
    new_model_name = os.path.join(file_path, file.replace(".phy", "_msa_model.txt"))  # Replace 'new_name' with the desired new name
    os.rename(model_path, new_model_name)
    copy_to_path_model = os.path.join(os.pardir, "data/processed/loo")
    shutil.copy(new_model_name, copy_to_path_model)  # Copy to query
