import os
import shutil
import time

import pandas as pd

import os
import subprocess

# Specify the path to the directory containing your folders
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection_aa_test.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
filenames = loo_selection["dataset"].values.tolist()
# Loop over each subdirectory (folder) within the specified path
counter = 0
filenames = ["10169_0", "16009_1", "9987_1", "22999_3", "13815_2", "16629_0", "25256_20", "16453_0", "18850_2", "2250_0"]

for file in filenames:
    msa_filepath = os.path.join(os.pardir, "data/raw/msa", file + "_reference.fasta")
    model_path = os.path.join(os.pardir, "data/processed/loo", file + "_msa_model.txt")

    counter += 1
    print(counter)
    print(len(filenames))
    start = time.time()
    #raxml_command = [
     #   "raxml-ng",
      #  "--bootstrap",
    #   # f"--model {model_path}",
     #   f"--bs-trees {1000}",
      #  f"--msa {msa_filepath}",
       # "--redo",
        #"--threads auto{60}"
    #]

    raxml_command = [
        "raxml-ng",
        "--bootstrap",
        f"--model GTR+G",
        f"--bs-trees {100}",
        f"--msa {msa_filepath}",
        "--redo",
        "--threads 1"
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
                                       "bootstrap_times_standard_SMALL.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  "bootstrap_times_standard_SMALL.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     "bootstrap_times_standard_SMALL.csv"),
                        index=False,
                        mode='a', header=False)

    print(elpased_time)

