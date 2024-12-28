
# Base directory to iterate
base_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/raw"

import os

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
        elif file.endswith(".model.txt"):
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
        file_paths[subfolder] = (newick_path, model_path, fasta_path, msa_type)

# Output results
print("File Paths (Newick, Model, Fasta, MSA Type):")
for subfolder, paths in file_paths.items():
    print(f"{subfolder}: {paths}")


    #raxml_command = [
    #    "raxml-ng",
     #   "--bootstrap",
      #  "--model", model_path,
       # f"--bs-trees {5000}",
      #  "--msa", msa_filepath,
       # "--redo",
       # "--prefix", "output_prefix"
    #]

