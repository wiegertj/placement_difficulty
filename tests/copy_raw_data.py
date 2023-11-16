import os
import shutil

import pandas as pd
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/processed/final/bs_support.csv"))
loo_selection = loo_selection["dataset"].values.tolist()
loo_selection_aa = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/loo_selection_aa_test.csv")
loo_selection_aa = loo_selection_aa["name"].str.replace(".phy", "").values.tolist()
all_datasets = loo_selection + loo_selection_aa

raw_path = "/hits/fast/cme/wiegerjs/EBG-train/EBG-train/data/raw"

for folder_name in all_datasets:
    msa_file_base = os.path.join(os.pardir, "data/raw/msa", folder_name + "_reference.fasta")
    model_file_base = os.path.join(os.pardir, "data/processed/loo", folder_name + "_msa_model.txt")
    tree_path_base = os.path.join(os.pardir, "data/raw/reference_tree", folder_name + ".newick")
    support_path_base = os.path.join(os.pardir, "scripts/") + folder_name + "_1000.raxml.support"
    support_path_iq_base = os.path.join(os.pardir, "data/raw/msa/") + folder_name + "_reference.fasta.treefile"
    classic_raxml_boots_base = os.path.join(os.pardir, "data/processed/raxml_rapid_bs_deimos_test/RAxML_bootstrap." + folder_name + "_1000_bs_raxml_classic")

    folder_path = os.path.join(raw_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    shutil.copy(msa_file_base, os.path.join(folder_path, f"{folder_name}_msa.fasta"))
    shutil.copy(model_file_base, os.path.join(folder_path, f"{folder_name}_model.txt"))
    shutil.copy(tree_path_base, folder_path)
    shutil.copy(support_path_base, folder_path)

    if os.path.exists(support_path_iq_base):
        shutil.copy(support_path_iq_base, os.path.join(folder_path, f"{folder_name}_iqtree.treefile"))

    if os.path.exists(classic_raxml_boots_base):
        shutil.copy(classic_raxml_boots_base, os.path.join(folder_path, f"{folder_name}_rapid_bs.support"))

    break