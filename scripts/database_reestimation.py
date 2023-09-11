import os
import shutil
import sys
import dendropy
import ete3
import subprocess
import glob
import pandas as pd
import numpy as np
import types
import random
import tqdist
from Bio import AlignIO, SeqIO
from dendropy.calculate import treecompare


def remove_gaps(sequence):
    return sequence.replace('-', '')


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()

msa_counter = 0
for msa_name in filenames:
    msa_counter += 1
    print(str(msa_counter) + "/" + str(len(filenames)))
    print(msa_name)

    rf_distances = []
    bsd_distances = []

    sequence_ids = []
    try:
        for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"), "fasta"):
            sequence_ids.append(record.id)
    except FileNotFoundError:
        print("Reference MSA not found: " + msa_name)
        continue

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    MSA = AlignIO.read(filepath, 'fasta')
    counter = 0

    output_file_disaligned = output_file.replace(".fasta", "_disaligned.fasta")
    with open(output_file, "r") as input_handle, open(output_file_disaligned, "w") as output_handle:
        for line in input_handle:
            if line.startswith('>'):
                output_handle.write(line)
            else:
                sequence = line.strip()
                disaligned_sequence = remove_gaps(sequence)
                output_handle.write(disaligned_sequence + '\n')

            # Use MAFFT to realign MSA without query sequence then realign query to new MSA
            command = ["mafft", "--preservecase", output_file_disaligned]

            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                mafft_output = result.stdout
                aligned_output_file = output_file_disaligned.replace("_disaligned.fasta", "_aligned_mafft_bias.fasta")

                with open(aligned_output_file, "w") as output_file:
                    output_file.write(mafft_output)

                with open(output_file_disaligned.replace("_disaligned.fasta", "_added_query.fasta"),
                          "w") as output_file:
                    output_file.write(mafft_output)

            except subprocess.CalledProcessError as e:
                print("Error running MAFFT:")
                print(e.stderr)

            # ------------------------------------------ run RAxML-ng with LOO MSA ------------------------------------------

            command = ["raxml-ng", "--search", "--msa", aligned_output_file, "--model",
                       "GTR+G", "tree", "pars{50}, rand{50}", "--redo"]
            print(command)

            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    print("RAxML-ng process completed successfully.")
                    print("Output:")
                    print(stdout)
                else:
                    print("RAxML-ng process failed with an error.")
                    print("Error Output:")
                    print(stderr)
                    continue
            except:
                print("problem occured RAXML")
