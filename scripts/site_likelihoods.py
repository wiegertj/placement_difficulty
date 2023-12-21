import os
import pandas as pd

import subprocess
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO
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
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", tree_filename.replace(".newick", "_reference.fasta"))

    for record in SeqIO.parse(msa_filepath, "fasta"):
        sequence_length = len(record.seq)
        break

    model_path = os.path.join(os.pardir, "data/processed/loo", tree_filename.replace(".newick", "") + "_msa_model.txt")
    output_prefix = tree_filename.split(".")[0] + "_siteliks_"  # Using the filename as the prefix


    raxml_command = [
        "raxml-ng",
        "--sitelh",
        f"--model {model_path}",
        f"--tree {tree_path}",
        f"--msa {msa_filepath}",
        "--pat-comp off",
        "--redo",
        "--site-repeats off",
        f"--prefix {output_prefix}"
    ]
    print(raxml_command)
    subprocess.run(" ".join(raxml_command), shell=True)

    print(f"Likelihood analysis for {tree_filename} completed.")


    #result =  subprocess.run(" ".join(raxml_command), shell=True)
    #result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    #print("result")
    #print(result.stdout)
    # Check if the command was successful
    #if result.returncode == 0:
        # Extract numbers following "set: " using regular expressions
      #  numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)

      #  # Convert the extracted strings to integers or floats
     #   numbers = [int(num) if num.isdigit() else float(num) for num in numbers]

     #   # Print the extracted numbers
     #   print("Extracted numbers:", numbers)
    #else:
      #  # Print an error message if the command failed
      #  print("Command failed with the following error message:")
      #  print(result.stderr)
    #try:
       # results.append((tree_filename.replace(".newick", ""), numbers[0], numbers[1], numbers[2]))
    #except IndexError:
        #print("number extraction failed ....")

#res_df = pd.DataFrame(results, columns=["dataset", "avg_rf", "avg_rel_rf", "no_top"])
#res_df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features.csv"), index=False)