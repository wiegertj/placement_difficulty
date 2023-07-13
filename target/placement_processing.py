import json
import os
import pandas as pd
import math
import io
from scipy.stats import entropy
from Bio import Phylo


def extract_entropy(jplace_file, tree_file) -> pd.DataFrame:
    # Get branch count for normalization
    tree = Phylo.read(os.path.join(os.pardir, "data/raw/reference_tree", tree_file), "newick")
    num_branches = tree.count_terminals() - 1
    print(jplace_file)

    entropies = []
    with open(jplace_file, 'r') as f:
        jplace_data = json.load(f)

        # Calculate max and min branch length for min-max-normalization
        max_distance = 0
        min_distance = float("inf")
        for clade1 in tree.find_clades():
            for clade2 in tree.find_clades():
                distance = tree.distance(clade1, clade2)
                if distance > max_distance:
                    max_distance = distance
                if (distance < min_distance) and (clade1.name != clade2.name):
                    min_distance = distance

        for placement in jplace_data['placements']:
            sample_name = placement['n'][0]
            probabilities = placement['p']
            like_weight_ratios = [tup[2] for tup in probabilities]
            entropy_val = entropy(like_weight_ratios, base=2) / math.log2(num_branches)

            # calculate drop in lwr between best two branches
            drop = 0
            if len(like_weight_ratios) > 1:
                sorted_lwr = sorted(like_weight_ratios, reverse=True)
                largest_lwr1, largest_lwr2 = sorted_lwr[:2]
                drop = abs(largest_lwr1 - largest_lwr2)
            else:
                drop = 1

            # branch distance between two best
            branch_distance = 0
            if len(like_weight_ratios) > 1:
                tree = Phylo.read(io.StringIO(jplace_data["tree"]), "newick")
                best_edge = probabilities[0][0]
                second_best_edge = probabilities[1][0]

                clade_distance = 0
                clade_distance = tree.distance(best_edge, second_best_edge)
                if clade_distance != 0:
                    branch_distance = (clade_distance - min_distance) / (max_distance - min_distance)

            entropies.append((sample_name, entropy_val, drop, branch_distance))

    df = pd.DataFrame(entropies, columns=['sampleId', 'entropy', "lwr_drop", "branch_dist_best_two_placements"])
    return df


if __name__ == '__main__':

    for file, tree_file in [("neotrop_10k_epa_result.jplace", "neotrop.newick"), ("bv_epa_result.jplace", "bv.newick"),
                            ("tara_epa_result.jplace", "tara.newick")]:
        jplace_file_path = os.path.join(os.pardir, "data/raw/placements", file)
        df = extract_entropy(jplace_file_path, tree_file)
        df.to_csv(os.path.join(os.pardir, "data/processed/target", file.replace(".jplace", "") + "_entropy.csv"))
