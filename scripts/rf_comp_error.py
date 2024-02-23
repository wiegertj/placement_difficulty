import os
import pandas as pd
import subprocess

filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", ".newick")
filenames_aa = loo_selection["dataset"].values.tolist()

duplicate_data = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
accepted = []
counter = 0
filenames = filenames[:180]
filenames = filenames + filenames_aa
# Loop over each subdirectory (folder) within the specified path
counter = 0

for file in filenames:
    file = file.replace(".newick", "")
    file_path_parsimonies = f"/hits/fast/cme/wiegerjs/placement_difficulty/tests/{file}ebg_test/ebg_tmp/parsimony_tmp_1000.raxml.startTree"

    raxml_command = [
        "raxml-ng",
        "--consense MRE",
        f"--tree {file_path_parsimonies}",
        "--redo",
        "--log ERROR"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)

    consensus_path = file_path_parsimonies + ".raxml.consensusTreeMRE"
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)

    consensus_path = file_path_parsimonies + ".raxml.consensusTreeMRE"
    tree_path = os.path.join(os.pardir, "data/raw/reference_tree", file)

    with open(consensus_path, 'r') as consensus_file, open(tree_path, 'r') as tree_file, open(file + "_combined.txt",
                                                                                              'w') as output_file:
        output_file.write(consensus_file.read())
        output_file.write(tree_file.read())


