import os
import pandas as pd

import os
import subprocess
# Specify the path to the directory containing your folders
path = os.pardir + "/data/raxml_data"

# Loop over each subdirectory (folder) within the specified path
for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)

    if os.path.isdir(folder_path):
        msa_file_path = os.path.join(folder_path, 'msa.fasta')

        if os.path.exists(msa_file_path):
            with open(msa_file_path, 'r') as msa_file:
                abs_msa_file_path = os.path.abspath(msa_file_path)
                os.chdir(folder_path)

                print("found " + abs_msa_file_path)
                raxml_command = [
                "raxml-ng",
                "--search",
                f"--model GTR+G",
                f"--msa {abs_msa_file_path}",
                "--redo",
                "--tree pars{50}, rand{50}",
                f"--prefix {folder_name}"
                ]

                subprocess.run(" ".join(raxml_command), shell=True)
