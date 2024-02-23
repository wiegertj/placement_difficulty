import pandas as pd
import os
import subprocess
ebg_times = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/benchmark_ebg.csv")
diffs = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/treebase_difficulty.csv")
diffs["dataset"] = diffs["dataset"].str.replace(".phy", "")
ebg_times = ebg_times.merge(diffs, on=["dataset"], how="inner")
for index, row in ebg_times.iterrows():
    msa_name = row["dataset"]
    datatype = row["data_type"]
    msa_filepath = os.path.abspath(os.path.join(os.pardir, "data/raw/msa", msa_name + "_reference.fasta"))

    prefix = msa_name + "_parsing"
    if datatype == 'AA' or datatype == "DataType.AA":
        filler = "LG+G"
    else:
        filler = "GTR+G"
    raxml_command = [
        "raxml-ng",
        "--parse",
        f"--msa {msa_filepath}",
        f"--model " + filler,
        f"--prefix {prefix}",
        "--log ERROR"

    ]

    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)