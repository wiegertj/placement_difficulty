import re
import statistics
import subprocess
import pandas as pd
import pickle
import numpy as np
import os
from ete3 import Tree
from scipy.stats import skew


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


def get_features(filename):
    msa_path = os.path.join(os.curdir, filename)

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

    raxml_command = ["raxml-ng",
                     "--rfdist",
                     f"--tree {parsimony_trees_path}",
                     "--redo"]
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
        return None

    raxml_command = [
        "raxml-ng",
        "--consense MRE",
        f"--tree {parsimony_trees_path}",
        "--redo",
        "--log ERROR"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    consensus_path = parsimony_trees_path + ".raxml.consensusTreeMRE"

    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {consensus_path}",
                     f"--bs-trees {parsimony_trees_path}",
                     "--redo",
                     "--log ERROR"]

    subprocess.run(" ".join(raxml_command), shell=True)

    support_path = consensus_path + ".raxml.support"

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

        for node in phylo_tree.traverse():
            branch_id_counter += 1
            if not node.is_leaf():
                node.__setattr__("name", branch_id_counter)
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
                                irs_skw_right, avg_rf_no_boot))
        return results


features_list = get_features("test_reference.fasta")  # filename of the MSA in the current directory

features = pd.DataFrame(features_list,
                        columns=['branchId', 'pars_support_cons', 'std_pars_supp_parents', 'min_pars_supp_child_w',
                                 'std_pars_supp_child_w', 'min_pars_supp_child', 'mean_pars_supp_child',
                                 'std_pars_supp_child', 'irs_std_right', 'irs_skw_right',
                                 'avg_rf_no_boot'])

with open(os.curdir + '/branch_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

print(model.predict(features.drop(columns=["branchId"])))
