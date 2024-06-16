import os
import pandas as pd
import subprocess


# Define the function to run ModelTest-NG and get the best model
def run_modeltest_ng(filepath, data_type):
    output_prefix = "modeltest_output"

    # Normalize the data type for comparison
    data_type_normalized = data_type.lower()

    if data_type_normalized in ["dna", "datatype.dna"]:
        data_type_flag = "-d nt"
    else:
        data_type_flag = "-d aa"

    command = f"modeltest-ng -i {filepath} {data_type_flag} -o {output_prefix}"
    subprocess.run(command, shell=True, check=True)

    # Read the output file to get the best model
    best_model_path = f"{output_prefix}.best_model"
    with open(best_model_path, "r") as f:
        best_model = f.readline().strip()

    return best_model


# Path to the loo_selection CSV
loo_selection_path = os.path.join(os.pardir, "data/loo_selection.csv")

# Read the loo_selection CSV
loo_selection = pd.read_csv(loo_selection_path)
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()

# Path to the results CSV
results_csv_path = "/hits/fast/cme/wiegerjs/model_test_res.csv"

# Create the results CSV if it doesn't exist
if not os.path.exists(results_csv_path):
    results_df = pd.DataFrame(columns=["msa_name", "best_model"])
else:
    results_df = pd.read_csv(results_csv_path)

msa_counter = 0
for index, row in loo_selection.iterrows():
    msa_name = row['verbose_name'].replace(".phy", "")
    data_type = row['data_type']

    msa_counter += 1
    print(f"{msa_counter}/{len(filenames)}")
    print(msa_name)

    filepath = os.path.join(os.pardir, "data/raw/msa", f"{msa_name}_reference.fasta")

    # Run ModelTest-NG and get the best model
    best_model = run_modeltest_ng(filepath, data_type)

    # Append the result to the DataFrame
    results_df = results_df.append({"msa_name": msa_name, "best_model": best_model}, ignore_index=True)

# Save the results to CSV
results_df.to_csv(results_csv_path, index=False)
