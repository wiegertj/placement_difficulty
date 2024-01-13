import os
import statistics
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
result_new = []

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


    sbs_tree_unfiltered = os.path.join(os.pardir, "data/raw/reference_tree/") + msa_name + ".newick.raxml.support"
    print(sbs_tree_unfiltered)
    sbs_tree_unfiltered = ete3.Tree(sbs_tree_unfiltered, format=0)

    leaf_node = sbs_tree_unfiltered.search_nodes(name=taxon)[0]
    leaf_node.delete()
    leaf_names = sbs_tree_unfiltered.get_leaf_names()

    sum_support_filter = 0.0
    sum_support_unfilter = 0.0


    # Sum up the support values for newick_tree_original_copy
    for node in sbs_tree_unfiltered.traverse():
        if node.support is not None and not node.is_leaf():
            sum_support_unfilter += node.support

    # Sum up the support values for newick_tree_tmp
    for node in sbs_tree_filtered.traverse():
        if node.support is not None and not node.is_leaf():
            sum_support_filter += node.support

    print(sum_support_filter)
    print(sum_support_unfilter)

    result_new.append((sum_support_filter, sum_support_unfilter, sum_support_filter / sum_support_unfilter,taxon, msa_name, row['effect']))


    print(f"msa: {msa_name} taxon: {taxon} effect: {row['effect']}  new_effect {sum_support_filter / sum_support_unfilter}")

df_final = pd.DataFrame(result_new, columns=["new", "old","ratio" ,"taxon", "msa_name", "effect"])
print(df_final[["ratio", "effect"]])
print(df_final["ratio"].mean())
print(df_final["ratio"].median())
print(statistics.mean(df_final["ratio"] - df_final["effect"]))