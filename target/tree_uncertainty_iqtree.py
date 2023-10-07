import time

print("Started")

import pandas as pd

print("Started1")

import subprocess


print("Started2")

import os

print("Started3")

from ete3 import Tree

print("Started4")

from Bio import SeqIO, AlignIO

print("Started5")


filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()


filenames_filtered = filenames
duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames_filtered = filenames_filtered[:300]
for tree_filename in filenames_filtered:
    counter += 1
    print(str(counter) + "/" + str(len(filenames_filtered)))
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue 13808_7
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick" or tree_filename == "20632_1.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)
    print(tree_path)

    if os.path.exists(os.path.join(os.pardir, "scripts/") + tree_filename.replace(".newick", "") + "_parsimony_supp_99.raxml.support"):
        print("Found already, move on")
        #continue

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    if num_leaves >= 800:
        print("Too large, skipped")
        continue



    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    if sequence_length >= 8000:
        print("Too large, skipped")
        continue
    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")

    with open(model_path, 'r') as file:
        # Read the first line from the file
        line = file.readline().strip()

        # Find the position of the second closing bracket }
        second_end_brace = line.find('}', line.find('}') + 1)

        if second_end_brace != -1:
            # Extract the substring up to the second closing bracket }
            result_string = line[:second_end_brace + 1]

            # Find the last float within the first set of brackets
            #last_float_start = result_string.rfind('/')
            #if last_float_start != -1:
             #   last_float_end = result_string.rfind(' ', 0, last_float_start)
              #  if last_float_end != -1:
               #     # Delete the last float within the first set of brackets
                #    result_string = result_string[:last_float_end] + result_string[last_float_start:]


    output_prefix = tree_filename.split(".")[0] + "_1000"  # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                      output_prefix+".raxml.bootstraps")





    found_model = result_string.replace("/1.000000}", "}").replace("U","").replace("/",",")

    # Specify the file name where you want to save the string
    file_name_iqtreemodel = model_path.replace("model","model_iqtree")

    # Open the file in write mode and write the string to it
    with open(file_name_iqtreemodel, "w") as file:
        file.write(found_model)

    start_time = time.time()

    raxml_command = [
        "iqtree",
        f"-m GTR+G",
        f"-s {msa_filepath}",
        f"-t {tree_path}",
        f"-B {1000}",
        "--redo"
    ]

    #print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)

    end_time = time.time()

    folder_path = os.path.join(os.pardir, "data/raw/msa")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files that contain "_parsimony_100temp_" in their names
    files_to_delete = [file for file in file_list if
                       ((".nex" in file) or (".log" in file))]


    # Delete the filtered files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)


    elapsed_time = end_time - start_time
    print("Elapsed time (seconds):", elapsed_time)
    alignment = AlignIO.read(msa_filepath, "fasta")

    num_sequences = len(alignment)

    # Get the length of the alignment
    alignment_length = alignment.get_alignment_length()

    # Print the number of sequences and length of the alignment
    print("Number of Sequences:", num_sequences)
    print("Length of Alignment:", alignment_length)

    data = {
        'elapsed_time': [int(elapsed_time)],
        'num_seq': [num_sequences],
        'len': [alignment_length]
    }

    time_dat = pd.DataFrame(data)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "pars_boot_times_iqrtree.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "pars_boot_times_iqrtree.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "pars_boot_times_iqrtree.csv"),
                        index=False,
                        mode='a', header=False)

#raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
