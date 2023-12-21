# Open the file in read mode
import os
from statistics import mean
import numpy as np
import pandas as pd

import subprocess
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO
import re

from scipy.stats import skew, kurtosis

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
for file in filenames:
    if not os.path.exists(os.path.join(os.pardir, "data/raw/reference_tree", file)):
        #print("Not found " + file)
        filenames.remove(file)

results = []
counter = 0
for tree_filename in filenames:
    dataset = tree_filename.replace(".newick", "")
    counter += 1
    print(counter)
    likpath = os.path.join(os.pardir, "scripts/", tree_filename.replace(".newick", "") + "_siteliks_.raxml.siteLH")
    try:
        with open(likpath, 'r') as file:
            # Read the lines from the file
            lines = file.readlines()

        # Check if there are at least two lines in the file
        if len(lines) >= 2:
            # Extract the second line (index 1) and remove leading/trailing whitespace
            second_line = lines[1].strip()
            #print(second_line)
            # Use regular expression to extract numbers from the second line
            import re

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", second_line)
            numbers = numbers[1:]

            numbers = [float(number) for number in numbers]
            min_value = min(numbers)
            max_value = max(numbers)

            # Perform min-max scaling
            scaled_numbers = [(x / sum(numbers)) for x in numbers]

            mean_loglik = mean(scaled_numbers)
            min_loglik = min(scaled_numbers)
            max_loglik = max(scaled_numbers)
            std_loglik = np.std(scaled_numbers)
            skw_loglik = skew(scaled_numbers)
            kurt_loglik = kurtosis(scaled_numbers, fisher=True)

            results.append((dataset, mean_loglik, min_loglik, max_loglik, std_loglik, skw_loglik, kurt_loglik))

        else:
            print("File does not contain at least two lines.")
    except FileNotFoundError:
        print("File not found")
        continue


df = pd.DataFrame(results, columns=["dataset","mean_loglik", "min_loglik", "max_loglik", "std_loglik", "skw_loglik", "kurt_loglik"])
df.to_csv(os.path.join(os.pardir, "data/processed/features/loglik.csv"))