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
grandir = os.path.join(os.getcwd(), os.pardir, os.pardir)

loo_selection = pd.read_csv(os.path.join(grandir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", ".newick").tolist()
for file in filenames:
    if not os.path.exists(os.path.join(grandir, "data/raw/reference_tree", file)):
        print("Not found " + file)
        filenames.remove(file)

for file in filenames:
    trees_pars = os.path.join(grandir, "scripts",
                 file.replace(".newick","") + "_parsimony_1000_nomodel.raxml.startTree")

    if not os.path.exists(trees_pars):
        continue

    output_prefix = file.replace(".newick", "") + "_consensus1000nomodel_"

    raxml_command = [
        "raxml-ng",
        "--consense",
        f"--tree {trees_pars}",
        "--redo",
        f"--prefix {output_prefix}"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)


    consensus_path = os.path.join(grandir, "features/split_features",
                 file.replace(".newick","") + "_consensus1000nomodel_.raxml.consensusTreeMR")

    original_path = os.path.join(os.pardir, "data/raw/reference_tree",
                 file)

    output_prefix = file.replace(".newick", "") + "_consensus1000nomodelsupport_"


    raxml_command = ["raxml-ng",
                     "--support",
                     f"--tree {consensus_path}",
                     f"--bs-trees {trees_pars}",
                     "--redo",
                     f"--prefix {output_prefix}"]

    subprocess.run(" ".join(raxml_command), shell=True)


    support_path = ""
