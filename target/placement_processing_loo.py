import io
import math
import multiprocessing
import os
import json
import pandas as pd
from Bio import Phylo
from scipy.stats import entropy
from joblib import Parallel, delayed
from multiprocessing import Pool
import re


def get_min_max(list):
    return min(list), max(list)


def calculate_distance(tree, clade):
    clades = tree.find_clades()
    distances = []
    for clade_tmp in clades:
        if clade_tmp.name != clade.name:
            distance = tree.distance(clade, clade_tmp)
            distances.append(distance)
    return distances


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
        newick_string = data["tree"]
        num_branches = newick_string.count(":")

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

                # compute min/max distance between clades in parallel
                all_clades = [(tree, clade1) for clade1 in tree.find_clades()]

                distances = Parallel(n_jobs=-1)(delayed(calculate_distance)(*args) for args in all_clades)

                print("Calculates list of distances ... start finding min/max")

                results = Parallel(n_jobs=-1)(delayed(get_min_max)(distance) for distance in distances)

                # Extract the minimum and maximum values from the results
                min_distance = min(result[0] for result in results)
                max_distance = max(result[1] for result in results)

                clade_distance = 0
                clade_distance = tree.distance(best_edge, second_best_edge)
                if clade_distance != 0:
                    branch_distance = (clade_distance - min_distance) / (max_distance - min_distance)

            return tree_name, sample_name, entropy_val, drop, branch_distance


def get_files_with_extension(directory):
    file_list = [(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.jplace')]
    return file_list


def extract_jplace_info(directory):
    counter = 0

    print("Start creating filelist ... ")

    with Pool(processes=80) as pool:
        # Split the directories for each process
        directories = [os.path.join(directory, subdir) for subdir in os.listdir(directory)]
        results = pool.map(get_files_with_extension, directories)

        # Merge the results from all processes into a single file list
    file_list = [item for sublist in results for item in sublist]
    # Now you have the complete file list
    print("Finished creating filelist ... ")

    if os.path.exists(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv")):
        current_df = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
        filtered_file_list = []

        for file_entry in file_list:
            dataset = file_entry[0].split('/')[4].split('_taxon')[0]
            dataset_match = current_df['dataset'].str.contains(dataset).any()
            if dataset_match:
                print("Found in df")
                print(file_entry)
            if not dataset_match:
                filtered_file_list.append(file_entry)
                print("Not found in df")
                print(file_entry)

        print("Finished filtering filelist ... ")
        file_list = filtered_file_list
    targets = []

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(extract_targets, file_list)

    for result in results:
        counter += 1
        print(str(counter) + "/" + str(len(file_list)))
        targets.append(result)

    pool.close()
    pool.join()

    df = pd.DataFrame(targets,
                      columns=["dataset", "sampleId", "entropy", "lwr_drop", "branch_dist_best_two_placements"])
    df.to_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"), index=False, header=False,
              mode='a')


if __name__ == '__main__':
    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    loo_results_path = os.path.join(os.pardir, "data/processed/loo_results")
    extract_jplace_info(loo_results_path)
