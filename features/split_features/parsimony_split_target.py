# parsimony 1000er laden, nomodel
# consenus tree der 1000er erstellen
# support auf consensus tree => parsimony support auf parsimony konsensus
# for each branch => check if in ML tree
#   if yes => 1
#   if not => 0
# ---------------------------------------------
# parsimony_branchId | branch_support_parsimony_consensus_tree_noboot | branch_support_parsimony_consensus_tree_boot |... MSA Features/Pars_topo_features ... | inML
# => classifier with probability

import subprocess

import ete3
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO

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

loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()

for file in filenames:
    if not os.path.exists(os.path.join(grandir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)
results = []
counter = 0
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
        f"--prefix {output_prefix}"
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
                     f"--prefix {output_prefix}"]

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
                results.append((dataset, node.name, node.support, node_in_ml_tree))

result_df = pd.DataFrame(results, columns=["dataset", "parsBranchId", "pars_support_cons", "inML"])
result_df.to_csv(os.path.join(grandir, "data/processed/final/split_prediction.csv"))


