import subprocess

import ete3
import pandas as pd
import os
from ete3 import Tree
from Bio import SeqIO

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/processed/target/loo_result_entropy.csv"))
loo_selection["dataset"] = loo_selection["dataset"] + ".newick"
filenames = loo_selection['dataset'].tolist()
filenames = set(filenames)
filenames = list(filenames)
filenames = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))["verbose_name"].str.replace(".phy", ".newick").values.tolist()

for file in filenames:
    trees_pars = os.path.join(os.pardir, "scripts",
                 file.replace(".newick","") + "_parsimony_20000.raxml.startTree")

    if not os.path.exists(trees_pars):
        continue

    output_prefix = file.replace(".newick", "") + "_consensus_"

    raxml_command = [
        "raxml-ng",
        "--consense",
        f"--tree {trees_pars}",
        "--redo",
        f"--prefix {output_prefix}"
    ]

    subprocess.run(" ".join(raxml_command), shell=True)


    consensus_path = os.path.join(os.pardir, "target",
                 file.replace(".newick","") + "_consensus_.raxml.consensusTreeMR")
    original_path = os.path.join(os.pardir, "data/raw/reference_tree",
                 file)

    command = ["/home/wiegerjs/tqDist-1.0.2/bin/quartet_dist", "-v",  os.path.abspath(consensus_path),
               os.path.abspath(original_path)]
    try:
        command_string = " ".join(command)
        print(command_string)
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        lines = output.strip().split('\n')
        values = lines[0].split()
        quartet_distance = float(values[3])
    except:
        print("quartet went wrong")

    with open(original_path, 'r') as original_file:

        original_newick_tree = original_file.read()
        original_tree = ete3.Tree(original_newick_tree)

        with open(consensus_path, 'r') as consensus_file:
            consensus_newick_tree = consensus_file.read()
            consensus_tree = ete3.Tree(consensus_newick_tree)

        results_distance = original_tree.compare(consensus_tree, unrooted=True)

        nrf_distance = results_distance["norm_rf"]

        print(nrf_distance)
        print(quartet_distance)

    results = [(file,nrf_distance, quartet_distance)]
    df = pd.DataFrame(results, columns=["dataset", "nrf", "quartet"])
    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "cons_comp.csv")):
        df.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "cons_comp.csv")), index=False)
    else:
        df.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                             "cons_comp.csv"),
                     index=False,
                     mode='a', header=False)

