
import os

print("Started3")

from ete3 import Tree
import pandas as pd
import os

# Directories
raw_data_dir = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/raw"
support_dir = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts"
results = []
# Iterate through each subfolder in the raw data directory
for root, dirs, files in os.walk(raw_data_dir):
    for subfolder in dirs:
        subfolder_name = subfolder
        # Construct the expected support file path
        support_file_path = os.path.join(support_dir, f"{subfolder_name}_rapid_ng_support.raxml.support")

        # Check if the support file exists
        if os.path.exists(support_file_path):
            try:
                with open(support_file_path, "r") as support_file:
                    tree_str = support_file.read()
                    tree = Tree(tree_str)  #

                    branch_id_counter = 0

                    for node in tree.traverse():
                        all_supps = []
                        all_diff_supps = []
                        branch_id_counter += 1
                        node.__setattr__("name", branch_id_counter)
                        if node.support is not None and not node.is_leaf():
                            results.append((subfolder_name, branch_id_counter, node.support))
            except FileNotFoundError:
                continue

        else:
            print(f"no: {support_file_path}")


results_df = pd.DataFrame(results, columns=["dataset", "branchId", "support_raxml_classic"])
print(len(results_df["dataset"].unique()))
results_df.to_csv("/hits/fast/cme/wiegerjs/rapid_bs_ng_support.csv")
