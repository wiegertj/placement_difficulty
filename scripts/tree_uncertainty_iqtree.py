

import subprocess
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/processed/target/loo_result_entropy.csv"))
loo_selection["dataset"] = loo_selection["dataset"] + ".newick"
filenames = loo_selection['dataset'].tolist()
filenames = set(filenames)
filenames = list(filenames)
filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()


filenames_filtered = []
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
for file in filenames:
    try:
        dupl = duplicate_data.loc[duplicate_data["name"] == file.replace(".newick", ".phy")].iloc[0]
        dupl = dupl["no_duplicates"]
    except IndexError:
        print("File not found in placements")
        continue
    print(dupl)
    print(file)

    if dupl == 0:
        print("removed")
        accepted.append(file)
    else:
        filenames_filtered.append(file)
    print("--------")
accepted_df = pd.DataFrame(accepted, columns=["dataset"])
accepted_df.to_csv(os.path.join(os.pardir, "data/bs_support_pred_selection.csv"), index=False)
for file in filenames_filtered:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames_filtered.remove(file)
print(len(filenames_filtered))
counter = 0
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    print(tree_path)

    if os.path.exists(os.path.join(os.pardir, "scripts/") + tree_filename.replace(".newick", "") + "_1000.raxml.support"):
        print("Found already, move on")
        #continue

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    if num_leaves >= 800:
        print("Too large, skipped")
        continue



    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    if sequence_length >= 8000:
        print("Too large, skipped")
        continue

    existing_csv_file = os.path.join(os.pardir, "data/bs_support_pred_selection.csv")
    filtered_filenames_df_tmp = pd.DataFrame([tree_filename], columns=["dataset"])
    filtered_filenames_df_tmp.to_csv(existing_csv_file, mode='a', header=False, index=False)

    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    output_prefix = tree_filename.split(".")[0] + "_1000"  # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                      output_prefix+".raxml.bootstraps")

    raxml_command = [
        "iqtree-ng",
        f"--m {model_path}",
        f"--s {msa_filepath}",
        f"--t {tree_path}",
        f"--bb {1000}"
    ]

    #print("Boot")

    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    #raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
