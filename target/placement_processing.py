import json
import multiprocessing
import os
import pandas as pd
import math
import io
from scipy.stats import entropy
from Bio import Phylo


def calculate_distance(tree, clade) -> list:
    """
    Function for calculating the distances between a clade and all other ones
    :param tree: Biopython tree
    :param clade: Clade to compute the distances for in the tree
    :return:
    """
    clades = tree.find_clades()
    distances = []
    for clade_tmp in clades:
        if clade_tmp.name != clade.name:
            distance = tree.distance(clade, clade_tmp)
            distances.append(distance)
    return distances


def process_placements(*args) -> (str, float, float, float):
    """
    Function for processing a single placement
    :param args: (, (jplace placement, tree in newick format string, max-dist int, min-dist int, no. of branches))
    :return: (sample_name, normalized entropy of placements, drop of lwr between best two placements, branch distance between best two placement)
    """
    placement, tree, max_distance, min_distance, num_branches = args[0]
    sample_name = placement['n'][0]
    probabilities = placement['p']
    like_weight_ratios = [tup[2] for tup in probabilities]
    entropy_val = entropy(like_weight_ratios, base=2) / math.log2(num_branches)  # normalized entropy

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
        tree = Phylo.read(io.StringIO(tree), "newick")
        best_edge = probabilities[0][0]
        second_best_edge = probabilities[1][0]

        for clade in tree.find_clades():
            if clade.name == "{" + str(best_edge) + "}":
                best_edge = clade
            elif clade.name == "{" + str(second_best_edge) + "}":
                second_best_edge = clade
        clade_distance = tree.distance(best_edge, second_best_edge)
        if clade_distance != 0:
            branch_distance = (clade_distance - min_distance) / (max_distance - min_distance)

    return sample_name, entropy_val, drop, branch_distance


def extract_targets(jplace_file, tree_file) -> pd.DataFrame:
    """
    Function to process Neotrop, BV and Tara placements
    :param jplace_file: Jplace-file name for either neotrop, BV or Tara in data/raw/placements
    :param tree_file: Name of the tree-file in data/raw/reference_tree
    :return: Dataframe with the targets
    """
    tree_file_path = os.path.join(os.pardir, "data/raw/reference_tree", tree_file)

    with open(tree_file_path, 'r') as file:
        newick_string = file.read()

    num_branches = newick_string.count(":")
    entropies = []
    with open(jplace_file, 'r') as f:
        jplace_data = json.load(f)

        print("Calculated list of distances ... start finding min/max")

        if tree_file == "neotrop.newick":
            max_distance = 4.668786892515589  # precalculated values
            min_distance = 0.014107303220626066
        elif tree_file == "bv.newick":
            max_distance = 3.5175978763653024
            min_distance = 1.00000050002909e-06
        elif tree_file == "tara.newick":
            max_distance = 6.4444184
            min_distance = 1.0e-06

        pool = multiprocessing.Pool()
        results = pool.imap_unordered(process_placements,
                                      [(placement, jplace_data["tree"], max_distance, min_distance, num_branches) for
                                       placement in jplace_data['placements']])

        counter = 0
        for result in results:
            entropies.append(result)
            counter += 1
            if counter % 50 == 0:
                print("Processed: " + str(counter) + " placements of " + jplace_file)

        pool.close()
        pool.join()

    df = pd.DataFrame(entropies, columns=['sampleId', 'entropy', "lwr_drop", "branch_dist_best_two_placements"])
    return df


if __name__ == '__main__':

    for file, tree_file in [("neotrop_10k_epa_result.jplace", "neotrop.newick"), ("bv_epa_result.jplace", "bv.newick"),
                            ("tara_epa_result.jplace", "tara.newick")]:
        jplace_file_path = os.path.join(os.pardir, "data/raw/placements", file)
        df = extract_targets(jplace_file_path, tree_file)
        df.to_csv(os.path.join(os.pardir, "data/processed/target", file.replace(".jplace", "") + "_entropy.csv"))
