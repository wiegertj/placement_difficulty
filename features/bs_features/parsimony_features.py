import os
import pandas as pd
from ete3 import Tree

grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)
loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
results = []
counter = 0

for file in filenames:
    counter += 1
    print(counter)
    print(file)

    support_path = os.path.join(grandir, "scripts/") + file.replace(".newick", "") + "_parsimony_100.raxml.support"

    if not os.path.exists(support_path):
        print("Couldnt find support: " + support_path)

    with open(support_path, "r") as support_file:
        tree_str = support_file.read()
        tree = Tree(tree_str)

        count = 0
        branch_id_counter = 0
        for node in tree.traverse():
            branch_id_counter += 1
            node.__setattr__("name", branch_id_counter)
            if node.support is not None and not node.is_leaf():
                results.append((file.replace(".newick", ""), node.name, node.support / 100))

df_res = pd.DataFrame(results, columns=["dataset", "branchId", "parsimony_support"])
df_res.to_csv(os.path.join(grandir, "data/processed/features/bs_features/parsimony.csv"))
