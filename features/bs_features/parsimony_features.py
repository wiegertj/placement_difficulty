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

    if not os.path.exists(support_path):
        print("Couldnt find support: " + support_path)
        continue

    with open(support_path, "r") as support_file:
        tree_str = support_file.read()
        tree = Tree(tree_str)

        count = 0
        branch_id_counter = 0
        for node in tree.traverse():
            branch_id_counter += 1
            node.__setattr__("name", branch_id_counter)
            if node.support is not None and not node.is_leaf():

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
                                mean_pars_supp_parents_w, std_pars_supp_parents_w, skw_pars_supp_parents_w,
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

                                        ])
df_res.to_csv(os.path.join(grandir, "data/processed/features/bs_features/parsimony.csv"))
