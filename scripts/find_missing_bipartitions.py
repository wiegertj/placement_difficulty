from dendropy import Tree, TaxonNamespace
import itertools
from ete3 import Tree

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


# Define your threshold for compatibility (less than 50%)
compatibility_threshold = 0.5
def are_bipartitions_compatible(bipartition1, bipartition2):
    # Check if bipartition1 is nested within bipartition2
    if set(bipartition1).issubset(bipartition2):
        return True
    # Check if bipartition2 is nested within bipartition1
    if set(bipartition2).issubset(bipartition1):
        return True
    # Check if the intersection of bipartition1 and bipartition2 is empty
    if set(bipartition1) & set(bipartition2) == set():
        return True
    # If none of the conditions are met, bipartitions are not compatible
    return False

# Create a TaxonNamespace for your taxa
# Read newick trees from the file and process them
with open(trees_pars, "r") as tree_file:
    for line in tree_file:
        # Parse the newick tree
        tree = Tree(line)

        # Extract bipartitions from the tree

        bipartitions = []
        for node in tree.traverse():
            bipartitions.append(get_bipartition(node))
        print(bipartitions)

        for bipar in bipartitions:
            for bipar_true in true_bipartitions:
                is_comp_true = are_bipartitions_compatible(bipar, bipar_true)
                if is_comp_true:
                    if bipar not in false_bipartitions:
                        true_bipartitions.append(bipar)

        if len(true_bipartitions) == len(df_test) - 1:
            break

# Now your true_bipartitions list contains bipartitions for your complete tree
