import io
import math
import multiprocessing
import os
import json
import re

import pandas as pd
from Bio import Phylo
from scipy.stats import entropy
from joblib import Parallel, delayed
from multiprocessing import Pool


def get_min_max(list) -> (float, float):
    """
    Function to return min/max of a list
    :param list: list to return the min/max of
    :return: min and max of values
    """
    return min(list), max(list)


def calculate_distance(tree, clade) -> list:
    """
    Function to calculate the distance of a clade and every other clade in a tree
    :param tree: tree in Biopython format
    :param clade: clade-identifier
    :return: list of distance values
    """
    clades = tree.find_clades()
    distances = []
    for clade_tmp in clades:
        if clade_tmp.name != clade.name:
            distance = tree.distance(clade, clade_tmp)
            distances.append(distance)
    return distances


def extract_targets(*args):
    """
    Function to process one single jplace-file and return targets
    :param args: (, (root directory of the file, filename))
    :return: name of the tree, name of the sample, normalized placement entropy, drop of lwr between two best placements, branch length distance between two best placements
    """
    root, file = args[0]
    file_path = os.path.join(root, file)

    with open(file_path, 'r') as f:
        print(file_path)
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print("JSON Decoder Error: Skipped ")
            return 0

        file_path = os.path.join(root, file)
        folder_name = os.path.basename(os.path.dirname(file_path))
        tree_name = '_'.join(folder_name.split('_')[:2])

        newick_string = data["tree"]
        num_branches = newick_string.count(":")

        # Extract placement likelihood weight ratios
        sample_name = ""
        for placement in data['placements']:
            sample_name = placement['n'][0]
            probabilities = placement['p']
            like_weight_ratios = [tup[2] for tup in probabilities]
            entropy_val = entropy(like_weight_ratios, base=2) / math.log2(num_branches - 1)

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

                print("Calculates list of distances ... start finding min/max of " + tree_name)

                results = Parallel(n_jobs=-1)(delayed(get_min_max)(distance) for distance in distances)

                # Extract the minimum and maximum values from the results
                min_distance = min(result[0] for result in results)
                max_distance = max(result[1] for result in results)

                clade_distance = 0
                clade_distance = tree.distance(best_edge, second_best_edge)
                if clade_distance != 0:
                    branch_distance = (clade_distance - min_distance) / (max_distance - min_distance)

            return tree_name, sample_name, entropy_val, drop, branch_distance


def get_files_with_extension(directory) -> list:
    """
    Function to get all jplace-files in a directory
    :param directory: directory to search in
    :return: file-list with results
    """
    file_list = [(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.jplace')]
    return file_list


def extract_jplace_info(directory):
    """
    1. Create file-list of all jplace-files in directory in parallel
    2. Filter for already processed files by looking at results in "data/processed/target/loo_result_entropy.csv"
    3. Calculate targets for not found elements in parallel
    4. Store results in "data/processed/target/loo_result_entropy.csv" or append if exists
    :param directory: directory with EPA-ng results
    :return:
    """
    counter = 0

    print("Start creating filelist ... ")

    with Pool(processes=80) as pool:
        directories = [os.path.join(directory, subdir) for subdir in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subdir)) and re.match(r'.*_200_r1_\d{3}$', subdir)]
        results = pool.map(get_files_with_extension, directories)

    file_list = [item for sublist in results for item in sublist]
    print("Finished creating filelist ... ")

    #selection = pd.read_csv(os.path.join(os.pardir, "data/", "reest_selection.csv"))
    #selectionList = selection["reest_files"].str.replace(".newick", "").values.tolist()

    if not os.path.exists(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy_200.csv")):
        #current_df = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
        filtered_file_list = []

        for file_entry in file_list:
            dataset = file_entry[0].split('/')[4].split('_taxon')[0]
            #if dataset in selectionList:
            filtered_file_list.append(file_entry)


        #for file_entry in file_list:
         #   dataset = file_entry[0].split('/')[4].split('_taxon')[0]
          #  if dataset == "15861_1" or dataset == "15861_0" or dataset == "14688_29":
           #     continue
         #   dataset_match = current_df['dataset'].str.contains(dataset).any()
          #  if dataset_match:
            #    print("Found in df")
           #     print(file_entry)
            #if not dataset_match:
             #   filtered_file_list.append(file_entry)
              #  print("Not found in df")
               # print(file_entry)

        print("Finished filtering filelist ... ")

        file_list = filtered_file_list
    targets = []

    pool = multiprocessing.Pool()
    results = pool.imap_unordered(extract_targets, file_list)

    for result in results:
        counter += 1
        print(str(counter) + "/" + str(len(file_list)))
        if result != 0:
            print(result)
            targets.append(result)

    pool.close()
    pool.join()

    df = pd.DataFrame(targets,
                      columns=["dataset", "sampleId", "entropy", "lwr_drop", "branch_dist_best_two_placements"])
    if os.path.exists(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy_200_r1.csv")):
        df.to_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy_200_r1.csv"), header=False,
                  mode='a', index=False)
    else:
        df.to_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy_200_r1.csv"), header=True)


if __name__ == '__main__':
    if multiprocessing.current_process().name == 'MainProcess':
        multiprocessing.freeze_support()

    loo_results_path = os.path.join(os.pardir, "data/processed/loo_results")
    extract_jplace_info(loo_results_path)
