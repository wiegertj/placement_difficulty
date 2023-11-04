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

    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break


    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")




    output_prefix = tree_filename.split(".")[0] + "_1000"  # Using the filename as the prefix

    bootstrap_filepath = os.path.join(os.pardir, "scripts",
                                      output_prefix+".raxml.bootstraps")



    alignment = AlignIO.read(msa_filepath, "fasta")



    # Specify the file name where you want to save the string
    file_name_iqtreemodel = model_path.replace("model","model_iqtree")



    raxml_command = [
        "iqtree",
        f"-m {file_name_iqtreemodel}",
        f"-s {msa_filepath}",
        f"-t {tree_path}",
        f"-B {1000}"
    ]

    #print("Boot")
    print("Started")
    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Bootstrap analysis for {tree_filename} completed.")

    folder_path = os.path.join(os.pardir, "data/raw/msa")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files that contain "_parsimony_100temp_" in their names
    files_to_delete = [file for file in file_list if
                       ((".nex" in file) or (".log" in file) or (".iqtree" in file) or (".treefile" in file))]

    # Delete the filtered files
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)




#raxml_command = ["raxml-ng",
     #                "--support",
      #               f"--tree {tree_path}",
       #              f"--bs-trees {bootstrap_filepath}",
        #             "--redo",
         #            f"--prefix {output_prefix}"]

    #subprocess.run(" ".join(raxml_command), shell=True)
