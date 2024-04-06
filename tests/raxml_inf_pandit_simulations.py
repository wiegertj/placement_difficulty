import os
import subprocess
import pandas as pd

df = pd.read_csv("/hits/fast/cme/wiegerjs/PANDIT/data_pandit/folder_names.csv")

counter = 0
# Iterate over the folders
for folder in df["folder"]:
    msa_path = f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{folder}/data.{folder}"
    counter += 1
    print(counter)
    # Get file paths

    # Check if files exist
    if not (os.path.exists(msa_path)):
        print(f"Skipping {folder} - one or more files not found.")
        continue

    # Do something with the file paths, e.g., print them
    print(f"Folder: {folder}")
    print(f"MSA Path: {msa_path}")

    #raxml_command = [
     #   "ebg",
      #  f"-model {best_model_path}",
       # f"-msa {msa_path}",
        #f"-tree {best_tree_path}",
        #"-t b",
        #f"-o {folder}"
    #]
    prefix = folder + "_pandit_inference_"
    raxml_command = [
        "raxml-ng",
        "--search",
        f"--msa {msa_path}",
        "--model GTR+G",
        # "--model LG4M",
        "--threads auto",
        f"--prefix {prefix}"
        # "--data-type AA"

    ]

    # print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)
