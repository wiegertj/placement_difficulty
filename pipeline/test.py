import subprocess

# Dataset names
datasets = [
    "14954_0",
    "15021_3",
]

# Base paths
msa_base_path = "/hits/fast/cme/wiegerjs/placement_difficulty/data/raw/msa"
tree_base_path = "/hits/fast/cme/wiegerjs/placement_difficulty/data/raw/reference_tree"

for dataset in datasets:
    msa_file_path = f"{msa_base_path}/{dataset}_reference.fasta"
    model_path = f"{msa_base_path}/{dataset}.bestModel"
    tree_path = f"{tree_base_path}/{dataset}.bestTree"

    raxml_command = [
        "ebg",
        f"-model {model_path}",
        f"-msa {msa_file_path}",
        f"-tree {tree_path}",
        f"-o {dataset}",
        "-light",
        "-redo"
    ]

    print(f"Started processing {dataset}")
    command_str = " ".join(raxml_command)
    print(command_str)
    subprocess.run(command_str, shell=True)
