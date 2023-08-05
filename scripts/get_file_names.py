import shutil
import tarfile
import os
import pandas as pd
import numpy as np

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')

# Filter out already used LOO-files
if os.path.exists(os.path.join(os.pardir, "data/loo_selection.csv")):
    df_used = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    names_used = df_used["verbose_name"].unique()
    print(names_used)
    print(len(names_used))
    difficulties_df = difficulties_df[~difficulties_df["verbose_name"].isin(names_used)]

difficulty_ranges = np.arange(0.1, 1.1, 0.1)

samples = []
for i in range(len(difficulty_ranges) - 1):
    lower_bound = difficulty_ranges[i]
    upper_bound = difficulty_ranges[i + 1]
    subset = difficulties_df[
        (difficulties_df['difficult'] >= lower_bound) & (difficulties_df['difficult'] < upper_bound) & (
                    difficulties_df['data_type'] != 'AA')].sample(40)
    samples.append(subset)

result = pd.concat(samples)

if os.path.exists(os.path.join(os.pardir, "data/loo_selection.csv")):
    # Append to the existing file without writing the header again
    result.to_csv(os.path.join(os.pardir, "data/loo_selection.csv"), mode='a', index=False, header=False)
else:
    # Write the data to a new file with the header
    result.to_csv(os.path.join(os.pardir, "data/loo_selection.csv"), index=False, header=True)

# Create reference MSA
for file in result["verbose_name"]:
    file_path = os.path.join(os.pardir, "data/TreeBASEMirror-main/trees/" + file)
    tar_file = os.path.join(file_path, file + ".tar.gz")  # path of msa tar file
    print(file_path)

    if not os.path.exists(os.path.join(file_path, "tree_best.newick")):
        print("Tree file not found for " + file + " skipped")
        continue

    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(file_path)
    except FileNotFoundError:
        print("Not found: " + file + " skipped")
        continue

    extracted_path = os.path.join(file_path,
                                  'msa.fasta')
    new_name_msa = os.path.join(file_path, file.replace(".phy", "_reference.fasta"))

    os.rename(extracted_path, new_name_msa)

    copy_to_path = os.path.join(os.pardir, "data/raw/msa")
    shutil.copy(new_name_msa, copy_to_path)  # Copy to reference

    copy_to_path = os.path.join(os.pardir, "data/raw/query")
    shutil.copy(new_name_msa, copy_to_path)  # Copy to query
    os.rename(copy_to_path + "/" + file.replace(".phy", "_reference.fasta"),
              copy_to_path + "/" + file.replace(".phy", "_query.fasta"))

    # ----------------------------- COPY tree-------------------------------------------

    tree_path = os.path.join(file_path, "tree_best.newick")
    new_tree_name = os.path.join(file_path, file.replace(".phy", ".newick"))
    os.rename(tree_path, new_tree_name)
    copy_to_path_tree = os.path.join(os.pardir, "data/raw/reference_tree")
    shutil.copy(new_tree_name, copy_to_path_tree)

    # ----------------------------- COPY model -------------------------------------------

    model_path = os.path.join(file_path, "model_0.txt")
    new_model_name = os.path.join(file_path, file.replace(".phy", "_msa_model.txt"))
    os.rename(model_path, new_model_name)
    copy_to_path_model = os.path.join(os.pardir, "data/processed/loo")
    shutil.copy(new_model_name, copy_to_path_model)  # Copy to query
