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
filenames_filtered = []
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
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
    else:
        filenames_filtered.append(file)
    print("--------")
for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)
print(len(filenames))
counter = 0
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    if os.path.exists(os.path.join(os.pardir, "scripts/") + tree_filename.replace(".newick", "") + "_1000.raxml.support"):
        print("Found already, move on")
        #continue

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    if num_leaves >= 1000:
        print("Too large, skipped")
        continue


    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    if sequence_length >= 10000:
        print("Too large, skipped")
        continue


    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    output_prefix = tree_filename.split(".")[0] + "_100"  # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                      output_prefix+".raxml.bootstraps")

    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_path}",
        f"--bs-trees {100}",
        f"--msa {msa_filepath}",
        "--redo",
        f"--prefix {output_prefix}"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {bootstrap_filepath}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)
