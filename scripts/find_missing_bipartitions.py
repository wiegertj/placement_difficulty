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
df = pd.read_csv(os.path.join(os.pardir, "data/processed/final/split_prediction.csv"))
test_set = df["dataset"].unique().tolist()[0]

trees_pars = os.path.join(os.pardir, "scripts",
                          test_set + "_parsimony_1000_nomodel.raxml.startTree")


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
df_test = df[df["dataset"] == test_set]

consensus_path = os.path.join(os.pardir, "features/split_features",
                                  test_set + "_consensus1000nomodel_.raxml.consensusTreeMRE")


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
            inML = matching_row["inML"].values[0]
            if inML == 1:
                true_bipartitions.append(get_bipartition(node))
            else:
                false_bipartitions.append(get_bipartition(node))

taxon_namespace = phylo_tree.get_leaf_names()

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
                                 test_set)

    command = ["/home/wiegerjs/tqDist-1.0.2/bin/quartet_dist", "-v", os.path.abspath(test_set + "best_pars_tree.newick"),
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

        with open(consensus_path, 'r') as consensus_file:
            consensus_newick_tree = consensus_file.read()
            consensus_tree = ete3.Tree(consensus_newick_tree)

        results_distance = original_tree.compare(consensus_tree, unrooted=True)

        nrf_distance = results_distance["norm_rf"]

        print(nrf_distance)
        print(quartet_distance)

    results = [(test_set, nrf_distance, quartet_distance)]
    df = pd.DataFrame(results, columns=["dataset", "nrf", "quartet"])
    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "cons_comp_target_best_pars.csv")):
        df.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                            "cons_comp_target_best_pars.csv")), index=False)
    else:
        df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                               "cons_comp_target_best_pars.csv"),
                  index=False,
                  mode='a', header=False)



