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

module_path = os.path.join(os.pardir, "configs/feature_config.py")
random.seed(42)
feature_config = types.ModuleType('feature_config')
feature_config.__file__ = module_path

with open(module_path, 'rb') as module_file:
    code = compile(module_file.read(), module_path, 'exec')
    exec(code, feature_config.__dict__)


def remove_gaps(sequence):
    return sequence.replace('-', '')


def calculate_bsd_aligned(tree1, tree2):
    branch_lengths1 = [branch.length for branch in tree1]
    branch_lengths2 = [branch.length for branch in tree2]

    score = np.sum(np.abs(branch_lengths1 - branch_lengths2)) / (np.sum(branch_lengths1) + np.sum(branch_lengths2))
    return score


loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()

#current_loo_targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target/loo_result_entropy.csv"))
#dataset_set = set(current_loo_targets['dataset'])
#print(len(filenames))
#filtered_filenames = [filename for filename in filenames if filename not in dataset_set]
#print(len(filtered_filenames))
filtered_filenames = filenames
msa_counter = 0
for msa_name in filtered_filenames:
    msa_counter += 1
    print(str(msa_counter) + "/" + str(len(filtered_filenames)))
    print(msa_name)

    rf_distances = []
    bsd_distances = []

    sequence_ids = []
    try:
        for record in SeqIO.parse(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"), "fasta"):
            sequence_ids.append(record.id)

        if len(sequence_ids) >= feature_config.SEQUENCE_COUNT_THRESHOLD:  # if too large, skip
            continue
    except FileNotFoundError:
        print("Reference MSA not found: " + msa_name)
        continue

    filepath = os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta")
    MSA = AlignIO.read(filepath, 'fasta')

    if len(MSA[0].seq) >= feature_config.SEQUENCE_LEN_THRESHOLD:  # if too large, skip
        continue
    counter = 0

    # Create random sample
    if feature_config.LOO_SAMPLE_SIZE >= len(sequence_ids):
        sequence_ids_sample = sequence_ids
    else:
        sequence_ids_sample = random.sample(sequence_ids, feature_config.LOO_SAMPLE_SIZE)

    for to_query in sequence_ids_sample:

        #if os.path.exists(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)):
         #   if not os.listdir(os.path.join(os.pardir, "data/processed/loo_results",
          #                                 msa_name + "_" + to_query)):  # if folder empty
           #     print("Empty folder found for " + msa_name + " " + to_query + " filling it")
            #    os.rmdir(os.path.join(os.pardir, "data/processed/loo_results",
             #                         msa_name + "_" + to_query))  # delete empty folder
            #else:
             #   if feature_config.SKIP_EXISTING_PLACEMENTS_LOO:
              #      print("Skipping " + msa_name + " " + to_query + " result already exists")
               #     continue

        counter += 1
        print(to_query)
        print(str(counter) + "/" + str(len(sequence_ids_sample)))

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

        output_file = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_" + to_query + ".fasta")
        output_file = os.path.abspath(output_file)

        output_file_query = os.path.join(os.pardir, "data/processed/loo", msa_name + "_query_" + to_query + ".fasta")
        output_file_query = os.path.abspath(output_file_query)

        SeqIO.write(new_alignment, output_file, "fasta")
        SeqIO.write(query_alignment, output_file_query, "fasta")

        output_file_tree = output_file.replace(".fasta", ".newick")

        if feature_config.REESTIMATE_TREE == True:

            # Disalign msa
            output_file_disaligned = output_file.replace(".fasta", "_disaligned.fasta")
            output_file_query_disaligned = output_file_query.replace(".fasta", "_disaligned.fasta")
            with open(output_file, "r") as input_handle, open(output_file_disaligned, "w") as output_handle:
                for line in input_handle:
                    if line.startswith('>'):
                        output_handle.write(line)
                    else:
                        sequence = line.strip()
                        disaligned_sequence = remove_gaps(sequence)
                        output_handle.write(disaligned_sequence + '\n')
            # Disalign query
            with open(output_file_query, "r") as input_handle, open(output_file_query_disaligned, "w") as output_handle:
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
                aligned_output_file = output_file_disaligned.replace("_disaligned.fasta", "_aligned.fasta")

                with open(aligned_output_file, "w") as output_file:
                    output_file.write(mafft_output)

                command = ["mafft", "--preservecase", "--keeplength", "--add", output_file_query_disaligned,
                           "--reorder",
                           output_file_disaligned.replace("_disaligned.fasta", "_aligned.fasta")]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

                mafft_output = result.stdout

                with open(output_file_disaligned.replace("_disaligned.fasta", "_added_query.fasta"),
                          "w") as output_file:
                    output_file.write(mafft_output)

                extracted_query = None
                with open(output_file_disaligned.replace("_disaligned.fasta", "_added_query.fasta"), 'r') as input_file:
                    for record in SeqIO.parse(input_file, 'fasta'):
                        if record.id == to_query:
                            extracted_query = record
                with open(output_file_disaligned.replace("_disaligned.fasta", "_query_realigned.fasta"),
                          'w') as output_file:
                    SeqIO.write(extracted_query, output_file, 'fasta')
                query_path_epa = output_file_disaligned.replace("_disaligned.fasta", "_query_realigned.fasta")
                print("MAFFT Reestimation of " + msa_name + to_query + " was successful!")

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

                    model_path_epa = os.path.join(os.pardir, "data/processed/loo",
                                                  msa_name + "_msa_" + str(to_query) + "_aligned.fasta.raxml.bestModel")

                else:
                    print("RAxML-ng process failed with an error.")
                    print("Error Output:")
                    print(stderr)
                    continue

            except FileNotFoundError:
                print("RAxML-ng executable not found. Please make sure RAxML-ng is installed and in the system")

            rel_tree_path = os.path.join(os.pardir, "data/processed/loo",
                                         msa_name + "_msa_" + to_query + "_aligned.fasta.raxml.bestTree")
            tree_path = os.path.abspath(rel_tree_path)
            tree_path_epa = tree_path

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

                    results_distance = original_tree.compare(tree, unrooted=True)
                    print("Started quartet computation")
                    print(original_tree_path)
                    print(os.path.abspath(tree_path))
                    command = ["./home/wiegerjs/tqDist-1.0.2/bin/quartet_distance", original_tree_path, os.path.abspath(tree_path)]
                    try:
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        stdout, stderr = process.communicate()

                        if process.returncode == 0:
                            print("Quartet process completed successfully.")
                            print("Output:")
                            print(stdout)

                            model_path_epa = os.path.join(os.pardir, "data/processed/loo",
                                                          msa_name + "_msa_" + str(
                                                              to_query) + "_aligned.fasta.raxml.bestModel")

                        else:
                            print("Quartet Error")
                            print("Error Output:")
                            print(stderr)
                            continue

                    except FileNotFoundError:
                        print("RAxML-ng executable not found. Please make sure RAxML-ng is installed and in the system")
                    #quartet_distance = tqdist.quartet_distance(original_tree_path, tree_path)


                    # BSD distance

                    tree_list = dendropy.TreeList()
                    tree_list.read(data=original_tree.write(format=1), schema="newick")
                    tree_list.read(data=tree.write(format=1), schema="newick")

                    # Normalize Branch Lengths to be between 0 and 1
                    tree_len_1 = tree_list[0].length()
                    for edge in tree_list[0].postorder_edge_iter():
                        if edge.length is None:
                            edge.length = 0
                        else:
                            edge.length = float(edge.length) / tree_len_1

                    tree_len_2 = tree_list[1].length()
                    for edge in tree_list[1].postorder_edge_iter():
                        if edge.length is None:
                            edge.length = 0
                        else:
                            edge.length = float(edge.length) / tree_len_2

                    bsd_aligned = treecompare.euclidean_distance(tree_list[0], tree_list[1])

                    print("Branch Score Distance (Aligned Trees):", bsd_aligned)

                    print("MSA " + str(msa_name + " query " + str(to_query)))
                    print("RF distance is %s over a total of" % (results_distance["norm_rf"]))
                    print("Quartet Distance: " + quartet_distance)
                    rf_distances.append(
                        (msa_name + "_" + to_query, results_distance["norm_rf"], bsd_aligned, quartet_distance))
                    df_rf = pd.DataFrame(rf_distances, columns=["dataset_sampleId", "norm_rf_dist", "norm_bsd_dist", "norm_quartet_dist"])

                    if not os.path.isfile(os.path.join(os.pardir, "data/processed/final", "norm_rf_loo.csv")):
                        df_rf.to_csv(os.path.join(os.pardir, "data/processed/final", "dist_loo_reestimate.csv"), index=False,
                                     columns=["dataset_sampleId", "norm_rf_dist", "norm_bsd_dist", "norm_quartet_dist"])
                    else:
                        df_rf.to_csv(os.path.join(os.pardir, "data/processed/final", "dist_loo_reestimate.csv"), index=False,
                                     mode='a', header=False, columns=["dataset_sampleId", "norm_rf_dist", "norm_bsd_dist", "norm_quartet_dist"])
                    rf_distances = []
                    msa_path_epa = aligned_output_file
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
                tree_path_epa = tree_path
                msa_path_epa = output_file
                model_path_epa = os.path.join(os.pardir, "data/processed/loo", msa_name + "_msa_model.txt")
                query_path_epa = output_file_query

        # ------------------------------------ run epa-ng with new RAxML-ng tree ---------------------------------------

        if os.path.exists(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query)):
            shutil.rmtree(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query))

        os.mkdir(os.path.join(os.pardir, "data/processed/loo_results", msa_name + "_" + to_query))
        print(model_path_epa)
        command = ["epa-ng", "--model", model_path_epa,
                   "--ref-msa", msa_path_epa, "--tree", tree_path_epa, "--query", query_path_epa, "--redo", "--outdir",
                   os.path.join(os.pardir, "data/processed/loo_results/" + msa_name + "_" + to_query), "--filter-max",
                   "10000", "--filter-acc-lwr", "0.99"]

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print("EPA-ng process completed successfully.")
                print("Output:")
                print(stdout)
            else:
                print("EPA-ng process failed with an error.")
                print("Error Output:")
                print(command)

        except FileNotFoundError:
            print("EPA-ng executable not found. Please make sure EPA-ng is installed and in the system PATH.")
        if feature_config.REESTIMATE_TREE == False:  # Delete tmp tree
            os.remove(os.path.join(os.pardir, "data/raw/reference_tree_tmp", msa_name + "_" + to_query + ".newick"))

        # ------------------------------------ Cleanup ---------------------------------------

        files = glob.glob(os.path.join(os.path.join(os.pardir, "data/processed/loo", f"*{to_query}*")))

        for file_path in files:
            os.remove(file_path)
