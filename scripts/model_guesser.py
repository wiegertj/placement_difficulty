import os
import pandas as pd
import subprocess


# Define the function to run ModelTest-NG and get the best model
def run_modeltest_ng(filepath, data_type):
    data_type_flag = "-d nt" if data_type.lower() in ["dna", "datatype.dna"] else "-d aa"

    command = f"modeltest-ng -i {filepath} {data_type_flag}"
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

    # Extract the best model from the output
    output_lines = result.stdout.splitlines()
    for line in output_lines:
        if line.startswith("Model:"):
            best_model = line.split(":")[1].strip()

            break
    print(f'best model {best_model}')
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

    # Append the result to the DataFrame using concat
    new_row = pd.DataFrame({"msa_name": [msa_name], "best_model": [best_model]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Save the results to CSV
results_df.to_csv(results_csv_path, index=False)
