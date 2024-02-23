import pandas as pd
import os
import subprocess
ebg_times = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/benchmark_ebg.csv")
diffs = pd.read_csv("/hits/fast/cme/wiegerjs/placement_difficulty/data/treebase_difficulty.csv")
diffs["dataset"] = diffs["verbose_name"].str.replace(".phy", "")
ebg_times = ebg_times.merge(diffs, on=["dataset"], how="inner")
results = []
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
        #"--log ERROR",
        "--redo"

    ]

    s = " ".join(raxml_command)
    print(s)
    subprocess.run(" ".join(raxml_command), shell=True)

    import re


    def extract_numbers(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                partitions_match = re.search(r'(\d+) partitions', line)
                patterns_match = re.search(r'(\d+) patterns', line)
                if partitions_match:
                    partitions_number = partitions_match.group(1)
                    print("Number of partitions:", partitions_number)
                if patterns_match:
                    patterns_number = patterns_match.group(1)
                    print("Number of patterns:", patterns_number)
                    return patterns_number


    # Usage
    file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/tests/" +msa_name +"_parsing.raxml.log"  # Specify your log file path here
    patterns_number = extract_numbers(file_path)

    results.append((msa_name, patterns_number))

print(results)