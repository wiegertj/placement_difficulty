import os
import pandas as pd

import os
import subprocess
# Specify the path to the directory containing your folders
list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter=0
# Loop over each subdirectory (folder) within the specified path
for folder_name in list_foldernames:
    folder_path = os.path.abspath(os.path.join(os.pardir, "data/raxml_data",str(folder_name)))
    print(folder_path)
    counter += 1
    print(counter)
    if os.path.isdir(folder_path):
        msa_file_path = os.path.join(folder_path, 'msa.fasta')
        abs_msa_file_path = os.path.abspath(msa_file_path)
        print(abs_msa_file_path)
        if os.path.exists(abs_msa_file_path):
            with open(msa_file_path, 'r') as msa_file:
                os.chdir(folder_path)

                print("found " + abs_msa_file_path)
                raxml_command = [
                "raxml-ng",
                "--search",
                f"--model GTR+G",
                f"--msa {abs_msa_file_path}",
                "--redo",
                "--tree pars{50},rand{50}",
                f"--prefix {folder_name}"
                ]

                subprocess.run(" ".join(raxml_command), shell=True)

                model_path =  os.path.abspath(os.pardir + "/" + str(folder_name) + "/" + str(folder_name) + ".raxml.bestModel")
                tree_path = os.path.abspath(os.pardir + "/" + str(folder_name) + "/" + str(folder_name) + ".raxml.bestTree")

                raxml_command = [
                    "raxml-ng",
                    "--bootstrap",
                    f"--model {model_path}",
                    f"--bs-trees {1000}",
                    f"--msa {abs_msa_file_path}",
                    "--redo",
                    f"--prefix {folder_name}"
                ]

                subprocess.run(" ".join(raxml_command), shell=True)

                print(f"Bootstrap analysis for {folder_name} completed.")
                boot_path = os.path.abspath(os.pardir + "/" + str(folder_name)+ "/" + str(folder_name) + ".raxml.bootstraps")
                raxml_command = [
                    "raxml-ng",
                "--support",
                f"--tree {tree_path}",
                f"--bs-trees {boot_path}",
                "--redo",
                f"--prefix {folder_name}"]

                subprocess.run(" ".join(raxml_command), shell=True)
                os.chdir(os.pardir)
                os.chdir(os.pardir)