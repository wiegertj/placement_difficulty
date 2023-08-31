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
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))
    bootstrap_supports = []

    # Output prefix for each tree
    output_prefix = tree_filename.split(".")[0]  # Using the filename as the prefix

    # Command to run RAxML for bootstrap analysis
    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--bs-trees {100}",
        f"--msa {msa_filepath}",
        "--redo"
    ]

    # Run RAxML using subprocess
    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    # Read the bootstrap support values from the output file
    with open(f"{output_prefix}.raxml.support", "r") as support_file:
        for line in support_file:
            bootstrap_supports.append(float(line.strip()))

# Calculate the average bootstrap support for the original tree
average_support = sum(bootstrap_supports) / len(bootstrap_supports)

print(f"Average support of branches in the original tree: {average_support:.2f}")
