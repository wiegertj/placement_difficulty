# parsimony 1000er laden, nomodel
# consenus tree der 1000er erstellen
# support auf consensus tree => parsimony support auf parsimony konsensus
# for each branch => check if in ML tree
#   if yes => 1
#   if not => 0
# ---------------------------------------------
# parsimony_branchId | branch_support_parsimony_consensus_tree_noboot | branch_support_parsimony_consensus_tree_boot |... MSA Features/Pars_topo_features ... | inML
# => classifier with probability
import statistics
import subprocess

import ete3
import pandas as pd
import os
from ete3 import Tree
import numpy as np
from Bio import SeqIO
from scipy.stats import skew


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

grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)

loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"), )
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

for file in filenames:
    if not os.path.exists(os.path.join(grandir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)
results = []
counter = 0
not_counter = 0
for file in filenames:
    counter += 1
    print(counter)
    trees_pars = os.path.join(grandir, "scripts",
                 file.replace(".newick","") + "_parsimony_1000_nomodel.raxml.startTree")

    if not os.path.exists(trees_pars):
        continue

    output_prefix = file.replace(".newick", "") + "_consensus1000nomodel_"
    dataset = file.replace(".newick", "")

    raxml_command = [
        "raxml-ng",
        "--consense MRE",
        f"--tree {trees_pars}",
        "--redo",
        f"--prefix {output_prefix}",
        "--log ERROR"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)


    consensus_path = os.path.join(grandir, "features/split_features",
                 file.replace(".newick","") + "_consensus1000nomodel_.raxml.consensusTreeMRE")

    original_path = os.path.join(grandir, "data/raw/reference_tree",
                 file)

    output_prefix = file.replace(".newick", "") + "_consensus1000nomodelsupport_"


    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {consensus_path}",
                     f"--bs-trees {trees_pars}",
                     "--redo",
                     f"--prefix {output_prefix}"
                     "--log ERROR"]

    subprocess.run(" ".join(raxml_command), shell=True)


    support_path = consensus_path.replace("consensusTreeMRE", "support").replace("nomodel","nomodelsupport")

    with open(original_path, "r") as original_file:
        original_str = original_file.read()
        phylo_tree_original = Tree(original_str)

    if os.path.exists(support_path):
        print("Found support file")
        with open(support_path, "r") as support_file:
            tree_str = support_file.read()
            phylo_tree = Tree(tree_str)

        branch_id_counter = 0
        for node in phylo_tree.traverse():
            branch_id_counter += 1
            if not node.is_leaf():
                node.__setattr__("name", branch_id_counter)
                node_in_ml_tree = 0

            # Check if bipartition exists in ML tree as label
                bipartition = get_bipartition(node)
                level = node.get_distance(phylo_tree, topology_only=True)
                if bipartition is not None:
                    for node_ml in phylo_tree_original.traverse():
                        bipartition_ml = get_bipartition(node_ml)
                        if bipartition_ml is not None:
                            first_match = False
                            second_match = False
                            if (bipartition[0] == bipartition_ml[0]) or (bipartition[0] == bipartition_ml[1]):
                                first_match = True
                            if (bipartition[1] == bipartition_ml[0]) or (bipartition[1] == bipartition_ml[1]):
                                second_match = True
                            if second_match and first_match:
                                    node_in_ml_tree = 1

                childs_inner = [node_child for node_child in node.traverse() if not node_child.is_leaf()]
                parents_inner = node.get_ancestors()
                supports_childs = []
                weighted_supports_childs = []
                for child in childs_inner:
                    supports_childs.append(child.support)
                    weighted_supports_childs.append(child.support * child.get_distance(phylo_tree, topology_only=True))

                supports_parents = []
                weighted_supports_parents = []
                for parent in parents_inner:
                    supports_parents.append(parent.support)
                    weighted_supports_parents.append(parent.support * parent.dist)

                if len(supports_childs) >= 1:
                    min_pars_supp_child = min(supports_childs)
                    max_pars_supp_child = max(supports_childs)
                    mean_pars_supp_child = statistics.mean(supports_childs)
                else:
                    min_pars_supp_child = -1
                    max_pars_supp_child = -1
                    mean_pars_supp_child = -1
                    std_pars_supp_child = -1
                    skw_pars_supp_child = -1

                if len(supports_childs) > 1:
                    std_pars_supp_child = np.std(supports_childs)
                    skw_pars_supp_child = skew(supports_childs)

                ###
                if len(weighted_supports_childs) >= 1:
                    min_pars_supp_child_w = min(weighted_supports_childs)
                    max_pars_supp_child_w = max(weighted_supports_childs)
                    mean_pars_supp_child_w = statistics.mean(weighted_supports_childs)
                else:
                    min_pars_supp_child_w = -1
                    max_pars_supp_child_w = -1
                    mean_pars_supp_child_w = -1
                    std_pars_supp_child_w = -1
                    skw_pars_supp_child_w = -1

                if len(weighted_supports_childs) > 1:
                    std_pars_supp_child_w = np.std(weighted_supports_childs)
                    skw_pars_supp_child_w = skew(weighted_supports_childs)

                ############

                if len(supports_parents) >= 1:
                    min_pars_supp_parents = min(supports_parents)
                    max_pars_supp_parents = max(supports_parents)
                    mean_pars_supp_parents = statistics.mean(supports_parents)
                else:
                    min_pars_supp_parents = -1
                    max_pars_supp_parents = -1
                    mean_pars_supp_parents = -1
                    std_pars_supp_parents = -1
                    skw_pars_supp_parents = -1

                if len(supports_parents) > 1:
                    std_pars_supp_parents = np.std(supports_parents)
                    skw_pars_supp_parents = skew(supports_parents)

                ###

                if len(weighted_supports_parents) >= 1:
                    min_pars_supp_parents_w = min(weighted_supports_parents)
                    max_pars_supp_parents_w = max(weighted_supports_parents)
                    mean_pars_supp_parents_w = statistics.mean(weighted_supports_parents)
                else:
                    min_pars_supp_parents_w = -1
                    max_pars_supp_parents_w = -1
                    mean_pars_supp_parents_w = -1
                    std_pars_supp_parents_w = -1
                    skw_pars_supp_parents_w = -1

                if len(weighted_supports_parents) > 1:
                    std_pars_supp_parents_w = np.std(weighted_supports_parents)
                    skw_pars_supp_parents_w = skew(weighted_supports_parents)

                results.append((dataset, node.name, node.support, node_in_ml_tree, level,
                                min_pars_supp_parents_w, max_pars_supp_parents_w, mean_pars_supp_parents_w,
                                std_pars_supp_parents_w, skw_pars_supp_parents_w,
                                min_pars_supp_parents, max_pars_supp_parents, mean_pars_supp_parents,
                                std_pars_supp_parents, skw_pars_supp_parents,
                                min_pars_supp_child_w, max_pars_supp_child_w, mean_pars_supp_child_w,
                                std_pars_supp_child_w, skw_pars_supp_child_w,
                                min_pars_supp_child, max_pars_supp_child, mean_pars_supp_child,
                                std_pars_supp_child, skw_pars_supp_child
                                ))
    else:
        print("Not found support")
        not_counter += 1
        print(not_counter)


result_df = pd.DataFrame(results, columns=["dataset", "parsBranchId", "pars_support_cons", "inML", "level"])
result_df.to_csv(os.path.join(grandir, "data/processed/final/split_prediction.csv"))


