import os
import statistics

import pandas as pd
from ete3 import Tree
from scipy.stats import skew

grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)
loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
results = []
counter = 0
import numpy as np

for file in filenames:
    counter += 1
    print(counter)
    print(file)

    support_path = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_parsimony_100.raxml.support"
    support_path_low = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_parsimony_100_low.raxml.support"
    support_path_low1000 = os.path.join(grandir, "scripts/") + file.replace(".newick",
                                                                        "") + "_parsimony_1000_low.raxml.support"
    support_path_low500 = os.path.join(grandir, "scripts/") + file.replace(".newick",
                                                                            "") + "_parsimony_500_low.raxml.support"
    if not os.path.exists(support_path):
        print("Couldnt find support: " + support_path)
        continue

    if not os.path.exists(support_path_low):
        print("Couldnt find support low: " + support_path_low)
        continue

    with open(support_path_low, "r") as support_file_low:
        tree_str_low = support_file_low.read()
        tree_low = Tree(tree_str_low)

        branch_id_counter_low = 0

        for node_low in tree_low.traverse():
            branch_id_counter_low += 1
            node_low.__setattr__("name", branch_id_counter_low)

    with open(support_path_low1000, "r") as support_file_low1000:
        tree_str_low1000 = support_file_low1000.read()
        tree_low1000 = Tree(tree_str_low1000)

        branch_id_counter_low1000 = 0

        for node_low1000 in tree_low1000.traverse():
            branch_id_counter_low1000 += 1
            node_low1000.__setattr__("name", branch_id_counter_low1000)

    with open(support_path_low500, "r") as support_file_low500:
        tree_str_low500 = support_file_low500.read()
        tree_low500 = Tree(tree_str_low500)

        branch_id_counter_low500 = 0

        for node_low500 in tree_low500.traverse():
            branch_id_counter_low500 += 1
            node_low500.__setattr__("name", branch_id_counter_low500)


    with open(support_path, "r") as support_file:
        tree_str = support_file.read()
        tree = Tree(tree_str)

        count = 0
        branch_id_counter = 0
        all_supports = []
        for node in tree.traverse():
            if not node.is_leaf():
                all_supports.append(node.support)

        min_pars_supp_tree = min(all_supports)
        max_pars_supp_tree = max(all_supports)
        mean_pars_supp_tree = statistics.mean(all_supports)
        std_pars_supp_tree = np.std(all_supports)
        skw_pars_supp_tree = skew(all_supports)

        for node in tree.traverse():
            all_supps = []
            all_diff_supps = []
            branch_id_counter += 1
            node.__setattr__("name", branch_id_counter)
            if node.support is not None and not node.is_leaf():
                node_low = tree_low.search_nodes(name=branch_id_counter)[0]
                node_low_support = node_low.support


                diff_support_100 = node.support - node_low_support

                node_low1000 = tree_low1000.search_nodes(name=branch_id_counter)[0]
                node_low_support1000 = node_low1000.support

                node_low500 = tree_low500.search_nodes(name=branch_id_counter)[0]
                node_low_support500 = node_low500.support

                diff_support_500 = node.support - node_low_support500

                all_supps.append(node.support)
                all_supps.append(node_low_support)
                all_supps.append(node_low_support1000)
                all_supps.append(node_low_support500)









                diff_support_1000 = node.support - node_low_support1000

                diff_support_500 = node.support - node_low_support500

                all_diff_supps.append(diff_support_100)
                all_diff_supps.append(diff_support_500)
                all_diff_supps.append(diff_support_1000)

                mean_all_supps = statistics.mean(all_supps)
                std_all_supps = np.std(all_supps)
                mean_all_diff_supps = statistics.mean(all_diff_supps)
                std_all_diff_supps = np.std(all_diff_supps)


                childs_inner = [node_child for node_child in node.traverse() if not node_child.is_leaf()]
                parents_inner = node.get_ancestors()
                supports_childs = []
                weighted_supports_childs = []
                for child in childs_inner:
                    supports_childs.append(child.support)
                    weighted_supports_childs.append(child.support * child.dist)

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



                results.append((file.replace(".newick", ""), node.name, node.support / 100, min_pars_supp_child,
                                max_pars_supp_child,
                                mean_pars_supp_child, std_pars_supp_child, skw_pars_supp_child, min_pars_supp_child_w,
                                max_pars_supp_child_w,
                                mean_pars_supp_child_w, std_pars_supp_child_w, skw_pars_supp_child_w,
                                min_pars_supp_parents, max_pars_supp_parents,
                                mean_pars_supp_parents, std_pars_supp_parents, skw_pars_supp_parents,
                                min_pars_supp_parents_w,
                                max_pars_supp_parents_w,
                                mean_pars_supp_parents_w, std_pars_supp_parents_w, skw_pars_supp_parents_w, min_pars_supp_tree, max_pars_supp_tree, std_pars_supp_tree, skw_pars_supp_tree, mean_pars_supp_tree,
                                diff_support_100 /100, node_low_support/100, diff_support_1000/100, node_low_support1000 / 100, diff_support_500/100, node_low_support500 / 100,
                                mean_all_supps, std_all_supps, mean_all_diff_supps, std_all_diff_supps
                                ))


df_res = pd.DataFrame(results, columns=["dataset", "branchId", "parsimony_support",
                                        "min_pars_supp_child", "max_pars_supp_child",
                                        "mean_pars_supp_child", "std_pars_supp_child", "skw_pars_supp_child",
                                        "min_pars_supp_child_w", "max_pars_supp_child_w",
                                        "mean_pars_supp_child_w", "std_pars_supp_child_w", "skw_pars_supp_child_w",
                                        "min_pars_supp_parents", "max_pars_supp_parents",
                                        "mean_pars_supp_parents", "std_pars_supp_parents", "skw_pars_supp_parents",
                                        "min_pars_supp_parents_w",
                                        "max_pars_supp_parents_w",
                                        "mean_pars_supp_parents_w", "std_pars_supp_parents_w",
                                        "skw_pars_supp_parents_w",
                                        "min_pars_supp_tree", "max_pars_supp_tree", "std_pars_supp_tree", "skw_pars_supp_tree",
                                        "mean_pars_supp_tree", "diff_support_100", "node_100_support" ,"diff_support_1000", "node_1000_support",

"diff_support_500", "node_500_support",
                                        "mean_all_supps", "std_all_supps", "mean_all_diff_supps", "std_all_diff_supps"

                                        ])
df_res.to_csv(os.path.join(grandir, "data/processed/features/bs_features/parsimony.csv"))
