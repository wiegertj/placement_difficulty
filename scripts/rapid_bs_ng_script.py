
# Base directory to iterate
base_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/raw"

import os
import time
import subprocess
import pandas as pd
# Base directory to iterate
base_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/raw"

# Dictionary to store tuples of (.newick, .model.txt) paths
file_paths = {}

# Iterate through subfolders
for root, dirs, files in os.walk(base_dir):
    newick_path = None
    model_path = None
    fasta_path = None
    msa_type = None  # Will store whether it's AA or DNA

    # Find .newick, .model.txt, and .fasta files
    for file in files:
        if file.endswith(".newick"):
            newick_path = os.path.join(root, file)
        elif file.endswith("_model.txt"):
            model_path = os.path.join(root, file)
        elif file.endswith(".fasta"):
            fasta_path = os.path.join(root, file)

    # If all three files are found in the subfolder, determine msa_type
    if newick_path and model_path and fasta_path:
        # Read the first line of the .model.txt file to check the type
        with open(model_path, 'r') as model_file:
            first_line = model_file.readline().strip()
            msa_type = "DNA" if first_line.startswith("GTR") else "AA"

        # Store paths and msa_type as a tuple
        subfolder = os.path.basename(root)
        file_paths[subfolder] = (newick_path, model_path, fasta_path, msa_type, subfolder)
results = []
# Output results
print("File Paths (Newick, Model, Fasta, MSA Type):")
for subfolder, paths in file_paths.items():
    print(f"{subfolder}: {paths}")
    tree_path = paths[0]
    model_path = paths[1]
    msa_path = paths[2]
    type = paths[3]

    output_prefix = subfolder + "_rb_ng"


    raxml_command = [
        "/hits/fast/cme/wiegerjs/rapid_boot_ng_dev/raxml-ng-dev/build/bin/raxml-ng-adaptive",
        "--bootstrap",
        "--bs-metric", "rbs",
        "--model", model_path,
        f"--bs-trees", "1000",
        "--msa", msa_path,
        "--redo",
        "--prefix", output_prefix
    ]
    try:
        start_time = time.time()  # Record start time
        subprocess.run(raxml_command, check=True)  # Execute the command
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
    except subprocess.CalledProcessError as e:
        # Format the command for copy-paste if it fails
        failed_command = " ".join(raxml_command)
        print(f"Error occurred while running the command:\n{failed_command}\n")
        print(f"Error details: {e}")
        results.append({"subprocess": subfolder, "elapsed_time": None})

    results.append({"subprocess": subfolder, "elapsed_time": elapsed_time})


# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save or display the DataFrame
print(df)
df.to_csv("/hits/fast/cme/wiegerjs/rapid_ng_times.csv", index=False)