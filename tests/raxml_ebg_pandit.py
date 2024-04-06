import time
import pandas as pd
import subprocess
import os
from ete3 import Tree
from Bio import SeqIO, AlignIO

list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter = 0
results = []
import pandas as pd

df = pd.read_csv("/hits/fast/cme/wiegerjs/PANDIT/data_pandit/folder_names.csv")

counter = 0
# Iterate over the folders
for folder in df["folder"]:

    model_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{folder}_pandit_inference_.raxml.bestModel"
    tree_path = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{folder}_pandit_inference_.raxml.bestTree"
    msa_path =  f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{folder}/data.{folder}"

    start_time = time.time()
    folder_name = str(folder) + "_ebg_pandit"

    raxml_command = [
        "ebg",
        f"-model {model_path}",
        f"-msa {msa_path}",
        f"-tree {tree_path}",
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

