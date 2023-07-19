import io
import math
import multiprocessing
import os
import json
import pandas as pd
from Bio import Phylo
from scipy.stats import entropy


def extract_targets(*args):
    root, file = args[0]
    file_path = os.path.join(root, file)

    with open(file_path, 'r') as f:
        print(file_path)
        data = json.load(f)

        file_path = os.path.join(root, file)
        folder_name = os.path.basename(os.path.dirname(file_path))
        tree_name = '_'.join(folder_name.split('_')[:2])

        # Get branch count for normalization
        tree = Phylo.read(os.path.join(os.pardir, "data/raw/reference_tree", tree_name + ".newick"),
                          "newick")
        num_branches = tree.count_terminals() - 1

        # Extract placement likelihood weight ratios
        sample_name = ""
        for placement in data['placements']:
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
                tree = Phylo.read(io.StringIO(data["tree"]), "newick")
                best_edge = probabilities[0][0]
                second_best_edge = probabilities[1][0]

                for clade in tree.find_clades():
                    if clade.name == "{" + str(best_edge) + "}":
                        best_edge = clade
                    elif clade.name == "{" + str(second_best_edge) + "}":
                        second_best_edge = clade

                max_distance = 0
                min_distance = float("inf")
                for clade1 in tree.find_clades():
                    for clade2 in tree.find_clades():
                        distance = tree.distance(clade1, clade2)
                        if distance > max_distance:
                            max_distance = distance
                        if (distance < min_distance) and (clade1.name != clade2.name):
                            min_distance = distance

                clade_distance = 0
                clade_distance = tree.distance(best_edge, second_best_edge)
                if clade_distance != 0:
                    branch_distance = (clade_distance - min_distance) / (max_distance - min_distance)

            return tree_name, sample_name, entropy_val, drop, branch_distance


def extract_jplace_info(directory):
    targets = []
    file_list = [(root, file) for root, dirs, files in os.walk(directory) for file in files if
                 file.endswith('.jplace')]

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(extract_targets, file_list)

    for result in results:
        targets.append(result)

    pool.close()
    pool.join()

    df = pd.DataFrame(results,
                      columns=["dataset", "sampleId", "entropy", "lwr_drop", "branch_dist_best_two_placements"])
    df.to_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"), index=False)


if __name__ == '__main__':
    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    loo_results_path = os.path.join(os.pardir, "data/processed/loo_results")
    extract_jplace_info(loo_results_path)
