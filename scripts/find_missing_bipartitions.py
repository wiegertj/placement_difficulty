import sys

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

def create_bitmask(bipartition, taxon_namespace):
    # Create an empty bitmask with 0s
    bitmask = [0] * len(taxon_namespace)
    bipartition = bipartition[0]

    # Set bits to 1 for taxa on one side of the bipartition
    for taxon in bipartition:
        index = taxon_namespace.index(taxon)
        bitmask[index] = 1

    return bitmask

# Your taxon_namespace (list of taxa)

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

# Define your threshold for compatibility (less than 50%)
compatibility_threshold = 0.5
def are_bipartitions_compatible(bipartition1, bipartition2):
    # Check if bipartition1 is nested within bipartition2

    bitmask1 = create_bitmask(bipartition1, taxon_namespace)
    bitmask2 = create_bitmask(bipartition2, taxon_namespace)
    bitmask_str1 = "".join(map(str, bitmask1))
    bitmask_str2 = "".join(map(str, bitmask2))
    filler = []
    for i in range(0, len(bitmask_str1)-1):
        if bitmask_str2[i] == str(0) and bitmask_str1[i] == str(0):
            filler.append(str(1))
        else:
            filler.append(str(0))
    filler_str = "".join(map(str, filler))

    for bit1, bit2, fill_bit in zip(bitmask1, bitmask2, filler):
        # If both bipartitions have the same taxon, they are incompatible
        if bit1 == bit2 == 1:
            return False
        # If a taxon is in the fill bitmask but not in either bipartition, they are incompatible
        if fill_bit == 1 and bit1 == bit2 == 0:
            return False
        # If no incompatibilities are found, the bipartitions are compatible
    return True


# Create a TaxonNamespace for your taxa
# Read newick trees from the file and process them
with open(trees_pars, "r") as tree_file:
    for line in tree_file:
        # Parse the newick tree
        tree = Tree(line)

        # Extract bipartitions from the tree

        bipartitions = []
        for node in tree.traverse():
            bipar_tmp = get_bipartition(node)
            if bipar_tmp is not None:
                bipartitions.append(get_bipartition(node))
        print(bipartitions)

        for bipar in bipartitions:
            for bipar_true in true_bipartitions:
                is_comp_true = are_bipartitions_compatible(bipar, bipar_true)
                if is_comp_true:
                    if bipar not in false_bipartitions:
                        print("added")
                        print(len(true_bipartitions))
                        true_bipartitions.append(bipar)
                        if len(true_bipartitions) == len(taxon_namespace) - 1:
                            # Add all trivial bipartitions
                            for taxon in taxon_namespace:
                                true_bipartitions.append({taxon})


                            def bipartition_to_newick(bipartition):
                                return "(".join(",".join(taxon for taxon in cluster) for cluster in bipartition) + ");"


                            # Join the bipartitions in Newick format
                            newick_tree = "(" + ",".join(bipartition_to_newick(bp) for bp in bipartitions) + ");"

                            # Store the Newick tree in a file named "test.txt"
                            with open("test.txt", "w") as tree_file:
                                tree_file.write(newick_tree)

                            # Exit the program
                            sys.exit()



# Now your true_bipartitions list contains bipartitions for your complete tree
