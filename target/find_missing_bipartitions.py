import subprocess
import sys
from math import inf

import ete3
from dendropy import Tree, TaxonNamespace
import itertools
from ete3 import Tree
from dendropy import Bipartition
import pandas as pd
import os


def get_bipartition(node):
    if not node.is_leaf():
        try:
            left_children = sorted([leaf.name for leaf in node.children[0].iter_leaves()])
            right_children = sorted([leaf.name for leaf in node.children[1].iter_leaves()])
            bipartition = (left_children, right_children)
            return bipartition
        except IndexError:
            return None
    return None


# Read the true and false bipartitions


df = pd.read_csv(os.path.join(os.pardir, "data/processed/features/split_features/all_data.csv"))

all_dataset = df["dataset"].unique().tolist()

for test_set in all_dataset:
    print(test_set)

    trees_pars = os.path.join(os.pardir, "scripts",
                              test_set + "_parsimony_10000_nomodel.raxml.startTree")

    df_test = df[df["dataset"] == test_set]
    if df_test.shape[0] == 0:
        print("Error, skipped")
        continue

    print(df_test.shape)

    consensus_path = os.path.join(os.pardir, "features/split_features",
                                  test_set + "_consensus10000nomodel_.raxml.consensusTreeMRE")

    true_bipartitions = []
    false_bipartitions = []

    with open(consensus_path, "r") as cons:
        tree_str = cons.read()
        phylo_tree = Tree(tree_str)

        branch_id_counter_ref = 0
        for node in phylo_tree.traverse():
            branch_id_counter_ref += 1
            if not node.is_leaf():
                node.__setattr__("name", branch_id_counter_ref)
                matching_row = df_test[df_test['parsBranchId'] == node.name]
                inML = int(matching_row["inML"].values[0])
                if inML == 1:
                    true_bipartitions.append(get_bipartition(node))
                else:
                    false_bipartitions.append(get_bipartition(node))

    with open(trees_pars, "r") as tree_file:
        score_max = -float(inf)
        best_tree = ""
        for line in tree_file:

            tree = Tree(line)
            score = 0
            bipartitions = []
            for node in tree.traverse():
                if not node.is_leaf():
                    bipar_tmp = get_bipartition(node)
                    if bipar_tmp is not None:
                        bipartitions.append(get_bipartition(node))

            for bipar in bipartitions:
                if bipar in true_bipartitions:
                    score += 1
                elif bipar in false_bipartitions:
                    score -= 1
            if score > score_max:
                score_max = score
                best_tree = tree
        print("Best Score: " + str(score_max))
        print("Best Tree: " + str(best_tree))

        best_tree.write(outfile=test_set + "best_pars_tree.newick", format=1)

        original_path = os.path.join(os.pardir, "data/raw/reference_tree",
                                     test_set + ".newick")

        command = ["/home/wiegerjs/tqDist-1.0.2/bin/quartet_dist", "-v",
                   os.path.abspath(test_set + "best_pars_tree.newick"),
                   os.path.abspath(original_path)]
        try:
            command_string = " ".join(command)
            print(command_string)
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
            lines = output.strip().split('\n')
            values = lines[0].split()
            quartet_distance = float(values[3])
        except:
            print("quartet went wrong")

        with open(original_path, 'r') as original_file:

            original_newick_tree = original_file.read()
            original_tree = ete3.Tree(original_newick_tree)

            results_distance = original_tree.compare(best_tree, unrooted=True)

            nrf_distance = results_distance["norm_rf"]

            results_distance_cons = original_tree.compare(phylo_tree, unrooted=True)

            nrf_distance_cons = results_distance_cons["norm_rf"]

            command = ["/home/wiegerjs/tqDist-1.0.2/bin/quartet_dist", "-v",
                       os.path.abspath(consensus_path),
                       os.path.abspath(original_path)]
            try:
                command_string = " ".join(command)
                print(command_string)
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                lines = output.strip().split('\n')
                values = lines[0].split()
                quartet_distance_consensus = float(values[3])
            except:
                print("quartet went wrong")

            print(nrf_distance)
            print(quartet_distance)

            score_max = score_max / (len(true_bipartitions) + len(false_bipartitions))
            print(score_max)

        results = [(test_set, nrf_distance, nrf_distance_cons,quartet_distance_consensus , quartet_distance, score_max)]
        df_tmp = pd.DataFrame(results, columns=["dataset", "nrf","nrf_cons" ,"quartet_cons","quartet", "relative_score"])
        if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                           "cons_comp_target_best_pars_10000.csv")):
            df_tmp.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                "cons_comp_target_best_pars_10000.csv")), index=False)
        else:
            df_tmp.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                   "cons_comp_target_best_pars_10000.csv"),
                      index=False,
                      mode='a', header=False)
