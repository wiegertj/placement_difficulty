import subprocess
import time

import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO, AlignIO
import re

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)

results = []
counter = 0
for tree_filename in filenames:
    counter += 1
    print(counter)
    #if os.path.exists(os.path.join(os.pardir, "data/raw/msa",
    #                                  tree_filename.replace(".newick", "_reference.fasta") + ".raxml.bootstraps")):
    #    print("Skipped, already found: " + tree_filename)
    #    continue
    if tree_filename == "11762_1.newick" or tree_filename == "11762_0.newick":
        print("skipped too large!")
        continue

    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_filename)

    t = Tree(tree_path)
    num_leaves = len(t.get_leaves())

    #if num_leaves >= 500:
     #   print("Too large, skipped")
      #  continue


    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    #if sequence_length >= 5000:
     #   print("Too large, skipped")


    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    output_prefix = tree_filename.split(".")[0] + "_parsimony_1000"  # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                      output_prefix+".raxml.startTree")
    start_time = time.time()

    raxml_command = [
        "raxml-ng",
        "--start",
        f"--model {model_path}",
        "--tree pars{1000}",
        f"--msa {msa_filepath}",
        "--redo",
        f"--prefix {output_prefix}"
    ]
    print(raxml_command)
    subprocess.run(" ".join(raxml_command), shell=True)


    print(f"Parsimony analysis for {tree_filename} completed.")

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {tree_path}",
                     f"--bs-trees {bootstrap_filepath}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)
    #bootstrap_filepath = os.path.join(os.pardir, "scripts/") + file.replace(".newick", "") + "_parsimony_supp_199.raxml.support"

    folder_path = os.path.join(os.pardir, "scripts")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files that contain "_parsimony_100temp_" in their names
    files_to_delete = [file for file in file_list if
                       ((".log" in file) or (".rba" in file) or ("reduced" in file))]

    # Delete the filtered files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)

    end_time = time.time()

    elapsed_time = end_time - start_time
    alignment = AlignIO.read(msa_filepath, "fasta")

    print("Elapsed time (seconds):", elapsed_time)
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
                                       "pars_boot_times_noboot1000.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "pars_boot_times_noboot1000.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "pars_boot_times_noboot1000.csv"),
                        index=False,
                        mode='a', header=False)

    #raxml_command = ["raxml-ng",
     #                "--rfdist",
      #               f"--tree {bootstrap_filepath}",
       #              "--redo",
        #             f"--prefix {output_prefix}"]
    #result =  subprocess.run(" ".join(raxml_command), shell=True)
    #result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    #print("result")
    #print(result.stdout)
    # Check if the command was successful
    #if result.returncode == 0:
     #   # Extract numbers following "set: " using regular expressions
      #  numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)

      #  # Convert the extracted strings to integers or floats
       # numbers = [int(num) if num.isdigit() else float(num) for num in numbers]

     #   # Print the extracted numbers
        #print("Extracted numbers:", numbers)
    #else:
     #   # Print an error message if the command failed
      #  print("Command failed with the following error message:")
       # print(result.stderr)
    #try:
     #  results.append((tree_filename.replace(".newick", ""), numbers[0], numbers[1], numbers[2]))
    #except IndexError:
     #   print("number extraction failed ....")

#res_df = pd.DataFrame(results, columns=["dataset", "avg_rf", "avg_rel_rf", "no_top"])
#res_df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features.csv"), index=False)