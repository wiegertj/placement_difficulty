import os
import re
import statistics
import subprocess

import pandas as pd
from ete3 import Tree
from scipy.stats import skew

grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)
loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
results = []
counter = 0
import numpy as np

for file in filenames:
    counter += 1
    print(counter)
    print(file)

    support_path_no_boot = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_parsimony_1000.raxml.startTree"
    support_path_boot = os.path.join(grandir, "data/raw/reference_tree/tmp_old") + file.replace(".newick", "") + "_pars_boot.txt"
    #support_path_low = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_parsimony_100_low.raxml.support"
    #support_path_low1000 = os.path.join(grandir, "scripts/") + file.replace(".newick",
     #                                                                   "") + "_parsimony_1000_low.raxml.s
    output_prefix_no_boot = file.split(".")[0] + "_parsimony_1000"  # Using the filename as the prefix
    output_prefix_boot = file.split(".")[0] + "_parsimony_199"  # Using the filename as the prefix


    raxml_command = ["raxml-ng",
                    "--rfdist",
                   f"--tree {support_path_no_boot}",
                  "--redo",
                 f"--prefix {output_prefix_no_boot}"]
    #result =  subprocess.run(" ".join(raxml_command), shell=True)
    result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    print("result")
    print(result.stdout)
    #Check if the command was successful
    if result.returncode == 0:
      # Extract numbers following "set: " using regular expressions
      numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)

    #  # Convert the extracted strings to integers or floats
      numbers = [int(num) if num.isdigit() else float(num) for num in numbers]

    #   # Print the extracted numbers
      print("Extracted numbers:", numbers)
    else:
    #   # Print an error message if the command failed
      print("Command failed with the following error message:")
      print(result.stderr)
    try:
        avg_rf_no_boot = numbers[0]
        avg_rel_rf_no_boot = numbers[1]
        no_top_no_boot = numbers[2]
    except (IndexError, NameError) as e:
        print("number extraction failed ....")
        continue
    ###########

    raxml_command = ["raxml-ng",
                     "--rfdist",
                     f"--tree {support_path_no_boot}",
                     "--redo",
                     f"--prefix {output_prefix_no_boot}"]
    #result = subprocess.run(" ".join(raxml_command), shell=True)
    result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            shell=True)

    print("result")
    print(result.stdout)
    # Check if the command was successful
    if result.returncode == 0:
        # Extract numbers following "set: " using regular expressions
        numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)

        #  # Convert the extracted strings to integers or floats
        numbers = [int(num) if num.isdigit() else float(num) for num in numbers]

        #   # Print the extracted numbers
        print("Extracted numbers:", numbers)
    else:
        #   # Print an error message if the command failed
        print("Command failed with the following error message:")
        print(result.stderr)
    try:
        avg_rf_boot = numbers[0]
        avg_rel_rf_boot = numbers[1]
        no_top_boot = numbers[2]
    except (IndexError, NameError) as e:
        print("number extraction failed ....")
        continue

    results.append((file.replace(".newick", ""), avg_rf_no_boot, avg_rf_boot, avg_rel_rf_no_boot, avg_rel_rf_boot, no_top_boot, no_top_no_boot))

res_df = pd.DataFrame(results, columns=["dataset", "avg_rf_no_boot", "avg_rf_boot", "avg_rel_rf_no_boot", "avg_rel_rf_boot", "no_top_boot", "no_top_no_boot"])
res_df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features.csv"), index=False)