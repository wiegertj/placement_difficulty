import os
import glob
import re
import shutil
import statistics
import subprocess
import sys

import numpy as np
from ete3 import Tree
from scipy.stats import skew
import pandas as pd

def height(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return max(height(node.children[0]), height(node.children[1])) + 1


def imbalance_ratio(node):
    if node is None:
        return 0
    try:
        left_height = height(node.children[0])
        right_height = height(node.children[1])
    except IndexError:
        return 0
    return max(left_height, right_height) / min(left_height, right_height)


def compute_tree_imbalance(tree):
    imbalance_values = []

    def collect_imbalance(node):
        if node is None:
            return
        if not node.is_leaf():
            ir = imbalance_ratio(node)
            imbalance_values.append(ir)
            for child in node.children:
                collect_imbalance(child)

    collect_imbalance(tree)
    return imbalance_values


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

search_directory = "/hits/fast/cme/wiegerjs/cons_pred/Tree/out"
file_name = "t1.raxml.mlTrees"
counter = 0
for root, dirs, files in os.walk(search_directory):
    if file_name in files:
        counter += 1
        if counter <= 889:
            continue
        print(counter)
        #print(root)
        os.chdir(root)  # Change the working directory to the directory where the file is found
        folder_name = os.path.basename(root)

        if folder_name == "13572_1.phy" or folder_name == "15260_0.phy" or folder_name == "13572_6.phy" or folder_name == "26784_1.phy" or folder_name == "11548_0.phy" or folder_name == "14093_0.phy":
            continue
        file_path = os.path.abspath(file_name)
        print(f"Folder: {folder_name}, File: {file_path}")
        msa_path = "/hits/basement/cme/hoehledi/example_workflow/TreeBase/" + folder_name
        if not os.path.exists(msa_path):
            print("MSA not found")
            print(msa_path)
            continue
        shutil.copy(msa_path, root)
        msa_path = root + "/" + folder_name

        raxml_command = [
            "raxml-ng",
            "--consense MRE",
            f"--tree {file_path}",
            "--redo",
            f"--prefix {folder_name + 'ground_truth_cons'}",
            "--log ERROR"
        ]

        subprocess.run(" ".join(raxml_command), shell=True)
        os.chdir(root)  # Change the working directory to the directory where the file is found

        raxml_command = [
            "raxml-ng",
            "--start",
            f"--model GTR+G",
            "--tree pars{1000}",
            f"--msa {msa_path}",
            "--redo",
            "--log ERROR"

        ]

        subprocess.run(" ".join(raxml_command), shell=True)

        parsimony_trees_path = msa_path + ".raxml.startTree"

        ###### GET RF FEATURES
        os.chdir(root)  # Change the working directory to the directory where the file is found

        raxml_command = ["raxml-ng",
                         "--rfdist",
                         f"--tree {parsimony_trees_path}",
                         "--redo"
                         ]
        result = subprocess.run(" ".join(raxml_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                shell=True)

        if result.returncode == 0:
            numbers = re.findall(r'set:\s+(-?[\d.]+)', result.stdout)
            numbers = [int(num) if num.isdigit() else float(num) for num in numbers]
            print("RF Distances:", numbers)
        else:
            print("RF computation failed:")
            print(result.stderr)
        try:
            avg_rf_no_boot = numbers[0]
        except (IndexError, NameError) as e:
            print("number extraction failed ....")

        ######## GET PARSIMONY SUPPORT FEATURES

        raxml_command = [
            "raxml-ng",
            "--consense MRE",
            f"--tree {parsimony_trees_path}",
            "--redo",
            f"--prefix {folder_name + 'pars_cons'}",
            "--log ERROR"

        ]

        subprocess.run(" ".join(raxml_command), shell=True)

        consensus_path = root + "/" + folder_name + 'pars_cons' +".raxml.consensusTreeMRE"

        raxml_command = ["raxml-ng",
                         "--support",
                         f"--tree {consensus_path}",
                         f"--bs-trees {parsimony_trees_path}",
                         "--redo",
                         "--log ERROR"
                         ]
        subprocess.run(" ".join(raxml_command), shell=True)

        support_path = consensus_path + ".raxml.support"
        ###### EXTRACT FEATURES
        if os.path.exists(support_path):
            results = []

            with open(support_path, "r") as support_file:
                tree_str = support_file.read()
                phylo_tree = Tree(tree_str)

            branch_id_counter = 0
            branch_id_counter_ref = 0
            phylo_tree_reference = phylo_tree.copy()
            for node in phylo_tree_reference.traverse():
                branch_id_counter_ref += 1
                if not node.is_leaf():
                    node.__setattr__("name", branch_id_counter_ref)

            ml_trees_consensus_path =  root + "/" + folder_name + "ground_truth_cons.raxml.consensusTreeMRE"
            print("ml cons path " + ml_trees_consensus_path)
            with open(ml_trees_consensus_path, "r") as phylo_tree_original_file:
                tree_str_ori = phylo_tree_original_file.read()
                phylo_tree_original = Tree(tree_str_ori)

            in_ML_counter = 0
            not_in_ML_counter = 0

            for node in phylo_tree.traverse():

                ###### TARGET

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
                                if (bipartition[0] == bipartition_ml[0]) or (
                                        bipartition[0] == bipartition_ml[1]):
                                    first_match = True
                                if (bipartition[1] == bipartition_ml[0]) or (
                                        bipartition[1] == bipartition_ml[1]):
                                    second_match = True
                                if second_match and first_match:
                                    node_in_ml_tree = 1
                                if node_in_ml_tree != 1:
                                    not_in_ML_counter += 1
                                else:
                                    in_ML_counter += 1

                ########

                if not node.is_leaf():
                    node.__setattr__("name", branch_id_counter)
                    childs_inner = [node_child for node_child in node.traverse() if not node_child.is_leaf()]
                    parents_inner = node.get_ancestors()
                    supports_childs = []
                    weighted_supports_childs = []
                    for child in childs_inner:
                        supports_childs.append(child.support)
                        weighted_supports_childs.append(
                            child.support * child.get_distance(phylo_tree, topology_only=True))

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

                    if len(supports_childs) > 1:
                        std_pars_supp_child = np.std(supports_childs)

                    if len(weighted_supports_childs) >= 1:
                        min_pars_supp_child_w = min(weighted_supports_childs)
                        max_pars_supp_child_w = max(weighted_supports_childs)
                        mean_pars_supp_child_w = statistics.mean(weighted_supports_childs)
                    else:
                        min_pars_supp_child_w = -1
                        max_pars_supp_child_w = -1
                        mean_pars_supp_child_w = -1
                        std_pars_supp_child_w = -1

                    if len(weighted_supports_childs) > 1:
                        std_pars_supp_child_w = np.std(weighted_supports_childs)

                    if len(supports_parents) >= 1:
                        min_pars_supp_parents = min(supports_parents)
                        max_pars_supp_parents = max(supports_parents)
                        mean_pars_supp_parents = statistics.mean(supports_parents)
                    else:
                        min_pars_supp_parents = -1
                        max_pars_supp_parents = -1
                        mean_pars_supp_parents = -1
                        std_pars_supp_parents = -1

                    if len(supports_parents) > 1:
                        std_pars_supp_parents = np.std(supports_parents)

                    phylo_tree_tmp = phylo_tree_reference.copy()  # copy reference
                    found_nodes = phylo_tree_tmp.search_nodes(name=node.name)

                    left_subtree = found_nodes[0].detach()
                    right_subtree = phylo_tree_tmp

                    irs_left = compute_tree_imbalance(left_subtree)
                    irs_right = compute_tree_imbalance(right_subtree)
                    irs_std_right = np.std(irs_right)
                    irs_skw_right = skew(irs_right)

                    results.append((node.name, node.support,
                                    std_pars_supp_parents, min_pars_supp_child_w, std_pars_supp_child_w,
                                    min_pars_supp_child, mean_pars_supp_child, std_pars_supp_child, irs_std_right,
                                    irs_skw_right, avg_rf_no_boot, node_in_ml_tree))
            data = pd.DataFrame(results,
                                    columns=['branchId', 'pars_support_cons', 'std_pars_supp_parents', 'min_pars_supp_child_w',
                                             'std_pars_supp_child_w', 'min_pars_supp_child', 'mean_pars_supp_child',
                                             'std_pars_supp_child', 'irs_std_right', 'irs_skw_right',
                                             'avg_rf_no_boot', "inCons"])
            data.to_csv("/hits/fast/cme/wiegerjs/cons_pred/work/" + folder_name + ".csv")
            print("Saved file")
