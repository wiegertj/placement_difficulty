import subprocess
import pandas as pd
import os

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)

counter = 0
for tree_filename in filenames:
    counter += 1
    print(counter)
    if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
                                      tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
        print("Skipped, already found: " + tree_filename)
        continue
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))
    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    bootstrap_filepath = os.path.join(os.pardir, "data/raw/msa",
                                      tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    output_prefix = tree_filename.split(".")[0]  # Using the filename as the prefix

    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_path}",
        f"--bs-trees {100}",
        f"--msa {msa_filepath}",
        "--redo"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {bootstrap_filepath}",
                     "--redo"]

    subprocess.run(" ".join(raxml_command), shell=True)
