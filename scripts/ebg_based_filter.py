import os
import re
import shutil
import dendropy
import ete3
import subprocess
import glob
import pandas as pd
import numpy as np
import types
import random
from Bio import AlignIO, SeqIO
from dendropy.calculate import treecompare

random.seed(200)

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
filenames = filenames[30:100]
msa_counter = 0
for msa_name in filenames:
    if msa_name == "17080_0":
        continue
    msa_counter += 1
    print(str(msa_counter) + "/" + str(len(filenames)))
    print(msa_name)

    rf_distances = []
    bsd_distances = []
    pythia_non_reest = []
    sequence_ids = []
    try:
        for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"), "fasta"):
            sequence_ids.append(record.id)

    except FileNotFoundError:
        print("Reference MSA not found: " + msa_name)
        continue

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    filepath = os.path.abspath(filepath)

    original_tree_path_tmp = os.path.abspath(os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick"))
    model_path_tmp = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"))

    MSA = AlignIO.read(filepath, 'fasta')

    counter = 0

    # Create random sample
    if 20 >= len(sequence_ids):
        sequence_ids_sample = sequence_ids
    else:
        sequence_ids_sample = random.sample(sequence_ids, 20)

    for to_query in sequence_ids_sample:

        counter += 1
        print(to_query)
        print(str(counter) + "/" + str(len(sequence_ids_sample)))

        new_alignment = []

        # Delete one out of MSA
        for record in MSA:
            if record.id != to_query:
                seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")

                new_alignment.append(seq_record)

        output_file = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_ebg_filter_" + to_query + ".fasta")
        output_file = os.path.abspath(output_file)

        with open(output_file, "w") as new_alignment_output:
            SeqIO.write(new_alignment, new_alignment_output, "fasta")

        # Delete one from tree
        original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick")

        tree_path = original_tree_path  # use original tree without reestimation
        print("-------------------------------------------")

        print("Getting original from " + tree_path)
        print("Start without reestimation")

        with open(tree_path, 'r') as file:
            print(tree_path)
            newick_tree = file.read()
            tree = ete3.Tree(newick_tree)

            leaf_names = tree.get_leaf_names()
            leaf_count = len(leaf_names)
            leaf_node = tree.search_nodes(name=to_query)[0]
            leaf_node.delete()
            leaf_names = tree.get_leaf_names()
            leaf_count = len(leaf_names)
            original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree_tmp",
                                              msa_name + "_" + to_query + "_ebg_filter.newick")
            print("Start creating loo tree")
            original_tree_path = os.path.abspath(original_tree_path)
            print("Storing to: " + original_tree_path)

            newick_string = tree.write()
            try:
                with open(original_tree_path, 'w') as file:
                    file.write(newick_string)
                print(f"Newick tree has been saved to {original_tree_path}")
            except Exception as e:
                print(f"An error occurred while saving the Newick tree: {str(e)}")
            print("-------------------------------------------")
        model_path = os.path.abspath(os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"))
        # make result dir
        if os.path.exists(       os.path.join(os.pardir, "data/processed/ebg_filter",
                         msa_name + "_" + to_query)):
            # If it exists, delete the directory and its contents
            shutil.rmtree(       os.path.join(os.pardir, "data/processed/ebg_filter",
                         msa_name + "_" + to_query))
        os.mkdir(
            os.path.join(os.pardir, "data/processed/ebg_filter",
                         msa_name + "_" + to_query))
        curdir_tmp = os.curdir
        os.chdir(os.path.join(os.pardir, "data/processed/ebg_filter",
                              msa_name + "_" + to_query))

        command = ["ebg",
                   f"-model {os.path.abspath(model_path)}",
                   f"-msa {os.path.abspath(output_file)}",
                   f"-tree {os.path.abspath(original_tree_path)}",
                   "-t b",
                   f"-o {msa_name + '_' + to_query}",
                   "-redo"]
        print(" ".join(command))

        try:
            subprocess.run(" ".join(command), shell=True)

        except:
            print("failed")
        os.chdir(curdir_tmp)
        os.chdir(os.path.abspath(os.path.join(os.pardir,
                              msa_name + "_" + to_query)))


        current_file_path = os.path.abspath(__file__)

        # Get the directory containing the currently executed file
        current_directory = os.path.dirname(current_file_path)

        os.chdir(current_directory)

    os.chdir(os.path.join(os.pardir, "data/processed/ebg_filter"
                          ))

    command = ["ebg",
               f"-model {model_path_tmp}",
               f"-msa {filepath}",
               f"-tree {original_tree_path_tmp}",
               "-t b",
               f"-o {msa_name}",
               "-redo"]
    print(" ".join(command))

    try:
        subprocess.run(" ".join(command), shell=True)
    except:
        print("failed")

    current_file_path = os.path.abspath(__file__)

    # Get the directory containing the currently executed file
    current_directory = os.path.dirname(current_file_path)

    os.chdir(current_directory)

