import os
import shutil
import dendropy
import ete3
import subprocess
import glob
import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
from dendropy.calculate import treecompare
import types
import sys
import importlib.util

module_path = os.path.join(os.pardir, "configs/feature_config.py")

feature_config = types.ModuleType('feature_config')
feature_config.__file__ = module_path

with open(module_path, 'rb') as module_file:
    code = compile(module_file.read(), module_path, 'exec')
    exec(code, feature_config.__dict__)


def calculate_bsd_aligned(tree1, tree2):
    # Get the branch lengths of the aligned branches
    branch_lengths1 = [branch.length for branch in tree1]
    branch_lengths2 = [branch.length for branch in tree2]

    # Calculate the BSD
    score = np.sum(np.abs(branch_lengths1 - branch_lengths2)) / (np.sum(branch_lengths1) + np.sum(branch_lengths2))

    return score


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()

print(filenames)
msa_counter = 0
for msa_name in filenames:
    msa_counter += 1
    print(str(msa_counter) + "/" + str(len(filenames)))
    print(msa_name)

    rf_distances = []
    bsd_distances = []

    # Get all sequence Ids
    sequence_ids = []
    for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"), "fasta"):
        sequence_ids.append(record.id)

    if len(sequence_ids) >= feature_config.SEQUENCE_COUNT_THRESHOLD:  # if too large, skip
        continue

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    MSA = AlignIO.read(filepath, 'fasta')

    if len(MSA[0].seq) >= feature_config.SEQUENCE_LEN_THRESHOLD:  # if too large, skip
        continue

    counter = 0

    # Perform LOO for each sequence
    for to_query in sequence_ids:

        if os.path.exists(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)):
            if not os.listdir(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)):
                print("Empty folder found for " + msa_name + " " + to_query + " filling it")
                os.rmdir(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)) # delete empty folder
            else:
                print("Skipping " + msa_name + " " + to_query + " result already exists")
                continue

        counter += 1
        print(to_query)
        print(str(counter) + "/" + str(len(sequence_ids)))

        new_alignment = []
        query_alignment = []

        # Split MSA into query and rest
        for record in MSA:
            if record.id != to_query:
                seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")
                new_alignment.append(seq_record)
            else:
                seq_record = SeqIO.SeqRecord(seq=record.seq, id=record.id, description="")
                query_alignment.append(seq_record)

        # Write temporary query and MSA to files
        output_file = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_" + to_query + ".fasta")
        output_file = os.path.abspath(output_file)

        output_file_query = os.path.join(os.pardir, "data/processed/loo", msa_name + "_query_" + to_query + ".fasta")
        output_file_query = os.path.abspath(output_file_query)

        SeqIO.write(new_alignment, output_file, "fasta")
        SeqIO.write(query_alignment, output_file_query, "fasta")

        # Get output tree path for result stroing
        output_file_tree = output_file.replace(".fasta", ".newick")

        if feature_config.REESTIMATE_TREE == True and len(
                sequence_ids) <= feature_config.REESTIMATE_TREE_SEQ_THRESHOLD:  # Reestimate smaller trees

            # ------------------------------------------ run RAxML-ng with LOO MSA ------------------------------------------

            command = ["raxml-ng", "--search", "--msa", output_file, "--model",
                       os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"), ]
            try:
                # Start the subprocess
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Wait for the process to complete and capture the output
                stdout, stderr = process.communicate()

                # Check the return code
                if process.returncode == 0:
                    # Success
                    print("RAxML-ng process completed successfully.")
                    print("Output:")
                    print(stdout)
                else:
                    # Error
                    print("RAxML-ng process failed with an error.")
                    print("Error Output:")
                    print(stderr)

            except FileNotFoundError:
                print("RAxML-ng executable not found. Please make sure RAxML-ng is installed and in the system")

            rel_tree_path = os.path.join(os.pardir, "data/processed/loo",
                                         msa_name + "_msa_" + to_query + ".fasta.raxml.bestTree")
            tree_path = os.path.abspath(rel_tree_path)

            # Calculate RF-statistics
            with open(tree_path, 'r') as file:
                newick_tree = file.read()
                tree = ete3.Tree(newick_tree)

                original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick")
                with open(original_tree_path, 'r') as original_file:

                    original_newick_tree = original_file.read()
                    original_tree = ete3.Tree(original_newick_tree)

                    # Get the leaf names
                    leaf_names = original_tree.get_leaf_names()

                    # Get the leaf count
                    leaf_count = len(leaf_names)
                    leaf_node = original_tree.search_nodes(name=to_query)[0]

                    # Delete the leaf node
                    leaf_node.delete()

                    results_distance = original_tree.robinson_foulds(tree, unrooted_trees=True)

                    # BSD distance

                    tree_list = dendropy.TreeList()
                    tree_list.read(data=original_tree.write(format=1), schema="newick")
                    tree_list.read(data=tree.write(format=1), schema="newick")

                    bsd_aligned = treecompare.euclidean_distance(tree_list[0], tree_list[1])

                    print("Branch Score Distance (Aligned Trees):", bsd_aligned)

                    print("MSA " + str(msa_name + " query " + str(to_query)))
                    print("RF distance is %s over a total of %s" % (results_distance[0], results_distance[1]))
                    rf_distances.append(
                        (msa_name + "_" + to_query, results_distance[0] / results_distance[1], bsd_aligned))
                    df_rf = pd.DataFrame(rf_distances, columns=["dataset_sampleId", "norm_rf_dist", "bsd"])

                    if not os.path.isfile(os.path.join(os.pardir, "data/processed/final", "norm_rf_loo.csv")):
                        # Create the file if it doesn't exist
                        df_rf.to_csv(os.path.join(os.pardir, "data/processed/final", "norm_rf_loo.csv"), index=False,
                                     columns=["dataset_sampleId", "norm_rf_dist", "bsd"])
                    else:
                        # Append to the file if it exists
                        df_rf.to_csv(os.path.join(os.pardir, "data/processed/final", "norm_rf_loo.csv"), index=False,
                                     mode='a', header=False, columns=["dataset_sampleId", "norm_rf_dist", "bsd"])
                    rf_distances = []
        else:
            original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree", msa_name + ".newick")
            tree_path = original_tree_path  # use original tree without reestimation

            with open(tree_path, 'r') as file:
                print(tree_path)
                newick_tree = file.read()
                tree = ete3.Tree(newick_tree)

                # Get the leaf names
                leaf_names = tree.get_leaf_names()

                # Get the leaf count
                leaf_count = len(leaf_names)
                print(leaf_count)

                leaf_node = tree.search_nodes(name=to_query)[0]

                # Delete the leaf node
                leaf_node.delete()

                leaf_names = tree.get_leaf_names()

                # Get the leaf count
                leaf_count = len(leaf_names)
                print(leaf_count)

                original_tree_path = os.path.join(os.pardir, "data/raw/reference_tree_tmp",
                                                  msa_name + "_" + to_query + ".newick")
                tree.write(outfile=original_tree_path, format=1)

                tree_path = original_tree_path

        # ------------------------------------ run epa-ng with new RAxML-ng tree ---------------------------------------

        # Create new directory for placement result
        if os.path.exists(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)):
            shutil.rmtree(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query))

        os.mkdir(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query))
        command = ["epa-ng", "--model", os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt"),
                   "--ref-msa", output_file, "--tree", tree_path, "--query", output_file_query, "--redo", "--outdir",
                   os.path.join(os.pardir, "data/processed/loo_results/" + msa_name + "_" + to_query), "--filter-max",
                   "10000", "--filter-acc-lwr", "0.99"]

        try:
            # Start the subprocess
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait for the process to complete and capture the output
            stdout, stderr = process.communicate()

            # Check the return code
            if process.returncode == 0:
                # Success
                print("EPA-ng process completed successfully.")
                print("Output:")
                print(stdout)
            else:
                # Error
                print("EPA-ng process failed with an error.")
                print("Error Output:")
                print(stderr)

        except FileNotFoundError:
            print("EPA-ng executable not found. Please make sure EPA-ng is installed and in the system PATH.")
        if feature_config.REESTIMATE_TREE == False:  # Delete tmp tree
            os.remove(os.path.join(os.pardir, "data/raw/reference_tree_tmp", msa_name + "_" + to_query + ".newick"))

        # ------------------------------------ Cleanup ---------------------------------------

        files = glob.glob(os.path.join(os.path.join(os.pardir, "data/processed/loo", f"*{to_query}*")))

        # Iterate over the temporary files and remove them
        for file_path in files:
            os.remove(file_path)
