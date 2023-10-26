import os
import shutil
import time

import pandas as pd

import os
import subprocess

# Specify the path to the directory containing your folders
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
loo_selection = loo_selection[:180]
filenames = loo_selection["dataset"].values.tolist()
# Loop over each subdirectory (folder) within the specified path
counter = 0
for file in filenames:
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", file + "_reference.fasta")
    model_path = os.path.join(os.pardir, "data/processed/loo", file + "_msa_model.txt")

    counter += 1
    if counter <= 122:
        continue
    if file == "20675_0":
        continue
    start = time.time()
    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model {model_path}",
        f"--bs-trees {1000}",
        f"--msa {msa_filepath}",
        "--redo",
        "--threads auto{60}"
    ]
    subprocess.run(" ".join(raxml_command), shell=True)
    elpased_time = time.time() - start
    data = {
        'dataset': file.replace(".newick", ""),
        'elapsed_time': int(elpased_time)
    }
    data_lost = [data]
    time_dat = pd.DataFrame(data_lost)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "bootstrap_times_standard.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "bootstrap_times_standard.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "bootstrap_times_standard.csv"),
                        index=False,
                        mode='a', header=False)

    print(elpased_time)

