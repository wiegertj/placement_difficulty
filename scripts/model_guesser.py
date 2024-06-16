import os
import pandas as pd
import subprocess


# Define the function to run ModelTest-NG and get the best model
def run_modeltest_ng(filepath, data_type):
    data_type_flag = "-d nt" if data_type.lower() in ["dna", "datatype.dna"] else "-d aa"

    command = f"modeltest-ng -i {filepath} {data_type_flag} --force"
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    # Extract the best model from the output
    best_model_bic = None
    best_model_aic = None
    model_count = 0

    output_lines = result.stdout.splitlines()
    for line in output_lines:
        if ".fasta --model" in line:
            model_count += 1
            parts = line.split()
            model_index = parts.index('--model') + 1
            model = parts[model_index].strip()
            if model_count == 1:
                best_model_bic = model
            elif model_count == 2:
                best_model_aic = model
                break

    if best_model_bic is not None and best_model_aic is not None:
        print(f'best model bic {best_model_bic}')
        print(f'best model bic {best_model_aic}')
        print('#'*20)


        return best_model_bic, best_model_aic


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
results_csv_path = "/hits/fast/cme/wiegerjs/model_test_res.csv"

# Create the results CSV if it doesn't exist
if not os.path.exists(results_csv_path):
    results_df = pd.DataFrame(columns=["msa_name", "best_model_bic", "best_model_aic"])
    results_df.to_csv(results_csv_path, index=False)

msa_counter = 0
for index, row in loo_selection.iterrows():
    msa_name = row['verbose_name'].replace(".phy", "")
    data_type = row['data_type']

    msa_counter += 1
    print(f"{msa_counter}/{len(filenames)}")
    print(msa_name)

    filepath = os.path.join(os.pardir, "data/raw/msa", f"{msa_name}_reference.fasta")

    # Run ModelTest-NG and get the best models
    try:
        best_model_bic, best_model_aic = run_modeltest_ng(filepath, data_type)
    except:
        best_model_aic = 'error'
        best_model_bic = 'error'
        # Append the result directly to the CSV file
        with open(results_csv_path, 'a') as f:
            pd.DataFrame(
                {"msa_name": [msa_name], "best_model_bic": [best_model_bic],
                 "best_model_aic": [best_model_aic]}).to_csv(f,
                                                             header=False,
                                                             index=False)



    if best_model_aic is None or best_model_bic is None:
        best_model_aic = 'error'
        best_model_bic = 'error'

    # Append the result directly to the CSV file
    with open(results_csv_path, 'a') as f:
        pd.DataFrame(
            {"msa_name": [msa_name], "best_model_bic": [best_model_bic], "best_model_aic": [best_model_aic]}).to_csv(f,
                                                                                                                     header=False,
                                                                                                                     index=False)


