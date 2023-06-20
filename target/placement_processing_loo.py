import math
import os
import json
import pandas as pd
from Bio import Phylo
from scipy.stats import entropy


def extract_jplace_info(directory):

    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jplace'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    tree_name = '_'.join(folder_name.split('_')[:2])

                    # Extract sample name
                    # Get branch count for normalization
                    tree = Phylo.read(os.path.join(os.pardir, "data/raw/reference_tree", tree_name + ".newick"), "newick")
                    num_branches = tree.count_terminals() - 1

                    # Extract placement likelihood weight ratios
                    sample_name = ""
                    for placement in data['placements']:
                        sample_name = placement['n'][0]
                        probabilities = placement['p']
                        like_weight_ratios = [tup[2] for tup in probabilities]
                        entropy_val = entropy(like_weight_ratios) / math.log(num_branches)
                        results.append((tree_name, sample_name, entropy_val))

    df = pd.DataFrame(results, columns=["dataset", "sampleId", "entropy"])
    df.to_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.newick"))

# Usage example
loo_results_path = os.path.join(os.pardir, "data/processed/loo_results")
extract_jplace_info(loo_results_path)
