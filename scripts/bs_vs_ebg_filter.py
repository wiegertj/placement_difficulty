import os
import subprocess
import sys

import ete3
import pandas as pd
from Bio import SeqIO, AlignIO

link = "/hits/fast/cme/wiegerjs/placement_difficulty/scripts/filtered_ebg_test.csv"
df = pd.read_csv(link)

idx = df.groupby('msa_name')['effect'].idxmax()

# Extract the corresponding rows
result_df = df.loc[idx]

print(result_df)
print(result_df.shape)

for index, row in result_df.iterrows():
    taxon = row["taxon"]
    msa_name = row["msa_name"]
    sbs_path_filtered = f"/hits/fast/cme/wiegerjs/placement_difficulty/scripts/{row['msa_name']}_{taxon}.raxml.support"
    try:
        sbs_tree_filtered = ete3.Tree(sbs_path_filtered, format=0)
    except Exception as e:
        continue


    sbs_tree_unfiltered = os.path.join(os.pardir, "data/raw/reference_tree/") + msa_name + ".raxml.support"
    sbs_tree_unfiltered = ete3.Tree(sbs_tree_unfiltered, format=0)

    leaf_node = sbs_tree_unfiltered.search_nodes(name=taxon)[0]
    leaf_node.delete()
    leaf_names = sbs_tree_unfiltered.get_leaf_names()

    sum_support_original_copy = 0.0
    sum_support_tmp = 0.0

    # Sum up the support values for newick_tree_original_copy
    for node in sbs_tree_unfiltered.traverse():
        if node.support is not None:
            sum_support_original_copy += node.support

    # Sum up the support values for newick_tree_tmp
    for node in sbs_tree_filtered.traverse():
        if node.support is not None:
            sum_support_tmp += node.support

    print(f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_tmp / sum_support_original_copy}")

