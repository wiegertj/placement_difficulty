import os
import pandas as pd

# Specify the base directory
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
filenames = filenames[:100]
msa_counter = 0
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
results = []

for msa_name in filenames:
    # Initialize a list to store DataFrames
    df_list = []

    # Iterate through folders in the base directory
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        # Check if it's a directory, ends with "xxx", and starts with msa_name
        if (
            os.path.isdir(folder_path)
            and folder_name.endswith("xxx")
            and folder_name.startswith(msa_name)
        ):
            # List CSV files in the folder
            csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

            # Assuming there is only one CSV file in each folder
            if len(csv_files) == 1:
                csv_file_path = os.path.join(folder_path, csv_files[0])

                # Read the CSV file into a DataFrame
                df = pd.read_csv(csv_file_path)
                df["id"] = folder_name

                # Append the DataFrame to the list
                df_list.append(df)

    # Concatenate the list of DataFrames into a single DataFrame
    concatenated_df = pd.concat(df_list, ignore_index=True)
    import numpy as np
    concatenated_df['reference_id'] = np.random.choice(concatenated_df['id'].unique())

    # Group by 'id' and sum up the values of 'prediction median'
    sum_per_id = concatenated_df.groupby('id')['prediction_median'].sum()

    # Use a random id's sum as reference
    reference_sum = sum_per_id.loc[concatenated_df['reference_id'].iloc[0]]

    # Express all other id's sums as a percentage of the reference sum
    sum_per_id_percentage = (sum_per_id / reference_sum) * 100

    # Print or use the results as needed
    print(max(sum_per_id_percentage))

