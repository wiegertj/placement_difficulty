import os
import shutil
import time

import pandas as pd

import os
import subprocess

# Specify the path to the directory containing your folders
# Specify the path to the directory containing your folders
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
filenames = loo_selection["dataset"].values.tolist()
# Loop over each subdirectory (folder) within the specified path
counter = 0

for file in filenames:
    if file == "20675_0":
        continue
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", file + "_reference.fasta")
    print(msa_filepath)
    model_path = os.path.join(os.pardir, "data/processed/loo", file + "_msa_model.txt")

    counter += 1
    print(counter)
    start = time.time()

    raxml_command = [
        "raxml-ng",
        "--search",
        f"--msa {msa_filepath}",
        "--model PROTGTR+G",
        "--redo",
        "--threads auto{60}",
        "--data-type AA"

    ]




    subprocess.run(" ".join(raxml_command), shell=True)
    elpased_time = time.time() - start
    data = {
        'dataset': file.replace(".newick", ""),
        'elapsed_time_inference': int(elpased_time)
    }
    data_lost = [data]
    time_dat = pd.DataFrame(data_lost)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "inference_times_standard_20.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "inference_times_standard_20.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "inference_times_standard_20.csv"),
                        index=False,
                        mode='a', header=False)

    print(elpased_time)

