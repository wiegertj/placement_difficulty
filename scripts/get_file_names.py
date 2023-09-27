import shutil
import tarfile
import os
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
from ete3 import Tree

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df.drop_duplicates(subset=["verbose_name"], keep="first", inplace=True)
difficulties_path_new = os.path.join(os.pardir, "data/treebase_difficulty_new.csv")
difficulties_df_new = pd.read_csv(difficulties_path_new, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_merged = difficulties_df_new.merge(difficulties_df, left_on="name", right_on="verbose_name", how="inner")
print(difficulties_merged["difficult"])
difficulties_merged["difficult"] = difficulties_merged["difficulty"]
print(difficulties_merged["difficult"])
difficulties_df = difficulties_merged
print(difficulties_df.shape)
if os.path.exists(os.path.join(os.pardir, "data/loo_selection.csv")):
    df_used = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    names_used = df_used["verbose_name"].unique()
    difficulties_df = difficulties_df[~difficulties_df["verbose_name"].isin(names_used)]

difficulty_ranges = np.arange(0.0, 1.1, 0.1)
samples = []
for i in range(len(difficulty_ranges) - 1):
    lower_bound = difficulty_ranges[i]
    upper_bound = difficulty_ranges[i + 1]
    print("Subset size " + str(lower_bound) + " - " + str(upper_bound))
    print(difficulties_df[
        (difficulties_df['difficult'] >= lower_bound) & (difficulties_df['difficult'] < upper_bound)].shape)
    subset = difficulties_df[
        (difficulties_df['difficult'] >= lower_bound) & (difficulties_df['difficult'] < upper_bound)]
    if len(subset) < 150:
        selected_subset = subset
    else:
        selected_subset = subset.sample(150)
    print("Subset size " + str(lower_bound) + " - " + str(upper_bound))
    print(subset.shape)
    samples.append(selected_subset)

if len(samples) == 1:
    result = samples[0]
else:
    result = pd.concat(samples)



# Create reference MSA and Query file
for file in result["verbose_name"]:
    file_path = os.path.join(os.pardir, "data/TreeBASEMirror-main/trees/" + file)
    tar_file = os.path.join(file_path, file + ".tar.gz")
    print(file_path)
    alternative_path_tree = False

    if not os.path.exists(os.path.join(file_path, "tree_best.newick")):
        if not os.path.exists(os.path.join(file_path, file.replace(".phy","") + ".newick")):
            print("Tree file not found for " + file + " skipped")
            result = result[result['verbose_name'] != file]
            continue
        alternative_path_tree = True


    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(file_path)
    except FileNotFoundError:
        print("Not found: " + file + " skipped")
        result = result[result['verbose_name'] != file]
        continue

    extracted_path = os.path.join(file_path,
                                  'msa.fasta')

    MSA = AlignIO.read(extracted_path, 'fasta')

    unique_sequences = set()
    alignment_dedup = []
    duplicate_counter = 0
    duplicate_names = []
    for record in MSA:
        # Convert the sequence to a string for comparison
        sequence_str = str(record.seq)

        # Check if the sequence is unique
        if sequence_str not in unique_sequences:
            unique_sequences.add(sequence_str)
            alignment_dedup.append(record)
        else:
            duplicate_counter += 1
            duplicate_names.append(record.id)
    print("Duplicate counter: " + str(duplicate_counter))
    print("Duplicates: " + str(duplicate_names))

    new_name_msa = os.path.join(file_path, file.replace(".phy", "_reference.fasta"))
    SeqIO.write(alignment_dedup, new_name_msa, "fasta")

    copy_to_path = os.path.join(os.pardir, "data/raw/msa")
    shutil.copy(new_name_msa, copy_to_path)

    copy_to_path = os.path.join(os.pardir, "data/raw/query")
    shutil.copy(new_name_msa, copy_to_path)
    os.rename(copy_to_path + "/" + file.replace(".phy", "_reference.fasta"),
              copy_to_path + "/" + file.replace(".phy", "_query.fasta"))

    # ----------------------------- COPY tree-------------------------------------------
    if alternative_path_tree == False:
        tree_path = os.path.join(file_path, "tree_best.newick")
    else:
        tree_path = os.path.join(file_path, file.replace(".phy","") + ".newick")


    # Load the tree from the file
    tree = Tree(tree_path)
    print("--------------------------------")
    print("To delete: " + str(len(duplicate_names)))
    print("Leaves before: " + str(len(tree.get_leaves())))
    deleted = 0
    for taxon in duplicate_names:
        node = tree.search_nodes(name=taxon)
        if node:
            node[0].delete()
            deleted += 1
    if deleted < len(duplicate_names):
        print("Did not find every node, skipped")
        result = result[result['verbose_name'] != file]
        continue

    print("Leaves after: " + str(len(tree.get_leaves())))
    print("--------------------------------")


    new_tree_name = os.path.join(file_path, file.replace(".phy", ".newick"))
    tree.write(outfile=new_tree_name)
    copy_to_path_tree = os.path.join(os.pardir, "data/raw/reference_tree")
    shutil.copy(new_tree_name, copy_to_path_tree)

    # ----------------------------- COPY model -------------------------------------------
    alternative_path_model = False
    if not os.path.exists(os.path.join(file_path, "model_0.txt")):
        if not os.path.exists(os.path.join(file_path, file.replace(".phy", "") + "_msa_model.txt")):
            continue
        alternative_path_model = True

    try:
        if alternative_path_model:
            model_path = os.path.join(file_path, file.replace(".phy", "") + "_msa_model.txt")
        else:
            model_path = os.path.join(file_path, "model_0.txt")
        new_model_name = os.path.join(file_path, file.replace(".phy", "_msa_model.txt"))
        os.rename(model_path, new_model_name)
        copy_to_path_model = os.path.join(os.pardir, "data/processed/loo")
        shutil.copy(new_model_name, copy_to_path_model)
    except FileNotFoundError:
        print("Not found: " + file + " skipped")
        result = result[result['verbose_name'] != file]
        continue
if os.path.exists(os.path.join(os.pardir, "data/loo_selection.csv")):
    result.to_csv(os.path.join(os.pardir, "data/loo_selection.csv"), mode='a', header=False, index=False)
else:
    result.to_csv(os.path.join(os.pardir, "data/loo_selection.csv"), header=True, index=False)