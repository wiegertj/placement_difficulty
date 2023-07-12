import json
import os
import pandas as pd
import math
from scipy.stats import entropy
from Bio import Phylo


def extract_entropy(jplace_file, tree_file) -> pd.DataFrame:

    # Get branch count for normalization
    tree = Phylo.read(os.path.join(os.pardir, "data/raw/reference_tree", tree_file), "newick")
    num_branches = tree.count_terminals() - 1

    entropies = []
    with open(jplace_file, 'r') as f:
        jplace_data = json.load(f)
        for placement in jplace_data['placements']:
            sample_name = placement['n'][0]
            probabilities = placement['p']
            like_weight_ratios = [tup[2] for tup in probabilities]
            entropy_val = entropy(like_weight_ratios, base=2) / math.log2(num_branches)
            entropies.append((sample_name, entropy_val))

    df = pd.DataFrame(entropies, columns=['sampleId', 'entropy'])
    return df


if __name__ == '__main__':

    for file, tree_file in [("neotrop_10k_epa_result.jplace", "neotrop.newick"), ("bv_epa_result.jplace", "bv.newick"),
                            ("tara_epa_result.jplace", "tara.newick")]:
        jplace_file_path = os.path.join(os.pardir, "data/raw/placements", file)
        df = extract_entropy(jplace_file_path, tree_file)
        df.to_csv(os.path.join(os.pardir, "data/processed/target", file.replace(".jplace", "") + "_entropy.csv"))
