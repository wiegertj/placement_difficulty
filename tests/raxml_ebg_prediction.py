import time
import pandas as pd
import subprocess
import os
from ete3 import Tree
from Bio import SeqIO, AlignIO

list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter = 0
results = []
# Loop over each subdirectory (folder) within the specified path
for folder_name in list_foldernames:
    folder_path = os.path.abspath(os.path.join(os.pardir, "data/raxml_data", str(folder_name)))
    # print(folder_path)
    counter += 1
    if (counter < 50 and counter != 1):
        continue
    print(counter)
    tree_path = os.path.abspath(os.pardir + "/data/raxml_data/" + str(folder_name) + "/" + str(folder_name) + ".raxml.bestTree")

    if not os.path.exists(tree_path):
        print("not found tree")
        continue
    if os.path.isdir(folder_path):
        model_path = os.path.abspath(os.pardir + "/data/raxml_data/" + str(folder_name) + "/" + str(folder_name) + ".raxml.bestModel")
        msa_file_path = os.path.join(folder_path, 'msa.fasta')
        msa_file_path = os.path.abspath(msa_file_path)

        start_time = time.time()

        raxml_command = [
            "ebg",
            f"-model {model_path}",
            f"-msa {msa_file_path}",
            f"-tree {tree_path}",
            "-pi 75",
            "-t b",
            f"-o {folder_name}"
        ]

        # print("Boot")
        print("Started")
        s = " ".join(raxml_command)
        print(s)
        subprocess.run(" ".join(raxml_command), shell=True)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print("Elapsed time (seconds):", elapsed_time)

