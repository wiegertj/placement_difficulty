import subprocess
import pandas as pd
import os

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()


for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)




# Loop through each tree file
for tree_filename in filenames:
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))
    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    bootstrap_filepath =  os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")
    # Output prefix for each tree
    output_prefix = tree_filename.split(".")[0]  # Using the filename as the prefix

    # Command to run RAxML for bootstrap analysis
    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_path}",
        f"--bs-trees {100}",
        f"--msa {msa_filepath}",
        "--redo"
    ]

    # Run RAxML using subprocess
    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")


    raxml_command = ["raxml-ng",
        "--support",
        f"--tree {tree_filename}",
        f"--bs-trees {bootstrap_filepath}",
        "--redo"   ]


    subprocess.run(" ".join(raxml_command), shell=True)

# Read the bootstrap support values from the output file
    with open(f"{output_prefix}.raxml.support", "r") as support_file:
        for line in support_file:
            bootstrap_supports.append(float(line.strip()))

# Calculate the average bootstrap support for the original tree
average_support = sum(bootstrap_supports) / len(bootstrap_supports)

print(f"Average support of branches in the original tree: {average_support:.2f}")
