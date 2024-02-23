import pandas as pd
import os
import subprocess
ebg_times = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/benchmark_ebg.csv")

for msa_name in ebg_times["dataset"].tolist():
    msa_filepath = os.path.abspath(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"))

    prefix = msa_name + "_parsing"
    raxml_command = [
        "raxml-ng",
        "--parse",
        f"--msa {msa_filepath}",
       # "--model LG+G",
        f"--prefix {prefix}"

    ]

    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)