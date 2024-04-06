import time
import pandas as pd
import subprocess
import os
from Bio import Nexus
from Bio import SeqIO
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
    msa_path_ebg = f"/hits/fast/cme/wiegerjs/PANDIT/data_pandit/{folder}/data.{folder}"

    # Define the path for the new FASTA file
    nexus = Nexus.Nexus.Nexus()

    with open(msa_path, "r") as msa_file:
        nexus.read(msa_file)

    # Extract sequences from the Nexus file
    msa_sequences = []


    for taxon, sequence in nexus.charsets.items():
        msa_sequences.append(SeqIO.SeqRecord(SeqIO.Seq("".join(sequence)), id=taxon))

    # Define the path for the new FASTA file
    fasta_path = msa_path + "_converted.fasta"

    # Write the sequences to the new FASTA file
    SeqIO.write(msa_sequences, fasta_path, "fasta")
    print(fasta_path)

    if not os.path.exists(fasta_path):
        print("Conversion failed")
        continue

    print("Conversion completed successfully.")


    start_time = time.time()
    folder_name = str(folder) + "_ebg_pandit"

    raxml_command = [
        "ebg",
        f"-model {model_path}",
        f"-msa {fasta_path}",
        f"-tree {tree_path}",
        "-t b",
        f"-o {folder_name}",
        "-redo"
    ]

    # print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time (seconds):", elapsed_time)

