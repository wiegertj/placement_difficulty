import os
import pandas as pd

# Specify the base directory
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
filenames = loo_selection['verbose_name'].str.replace(".phy", "").tolist()
filenames = filenames[260:]
msa_counter = 0
base_directory = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/ebg_filter"
results = []
result_flat = []

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
    try:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    except ValueError:
        break
    import numpy as np
    concatenated_df['reference_id'] = np.random.choice(concatenated_df['id'].unique())

    unique_ids = concatenated_df['id'].unique()

    # Initialize a list to store percentage changes
    percentage_changes = []

    # Iterate through each id as the reference
    for reference_id in unique_ids:
        # Skip comparisons with itself
        other_ids = unique_ids[unique_ids != reference_id]

        # Calculate the sum of 'prediction_median' for the reference id
        reference_sum = concatenated_df[concatenated_df['id'] == reference_id]['prediction_median'].sum()

        # Calculate the sum of 'prediction_median' for other ids
        sum_per_id = concatenated_df[concatenated_df['id'].isin(other_ids)].groupby('id')['prediction_median'].sum()

        # Calculate percentage change for each id with respect to the reference id
        percentage_change = (sum_per_id / reference_sum)

        # Append to the list
        percentage_changes.append(percentage_change)

    # Create a DataFrame from the list of percentage changes
    result_df = pd.DataFrame(percentage_changes, index=unique_ids, columns=unique_ids)

    flattened_list = result_df.stack().dropna().tolist()
    result_flat.extend(flattened_list)




    # Display or use the result_df as needed
    print(result_df)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Assuming 'concatenated_df' is your DataFrame with columns 'id' and 'prediction_median'
# 'result_df' is the DataFrame generated from the previous code
# 'flattened_list' is the flattened list from the previous code
# Adjust the column names accordingly if they are different in your actual data

# ... (previous code)

# Flatten the result_df into a list and drop NaN values

# Create a histogram
plt.hist(result_flat, bins=20, color='g', edgecolor='black', density=True)  # Note: density=True for normalization

# Fit a Gaussian curve to the histogram
mu, std = norm.fit(result_flat)

# Plot the Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(0, 100, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Add labels, title, and legend
plt.xlabel('Percentage Change')
plt.ylabel('Density')
plt.title('Histogram of Percentage Change with Gaussian Fit')
plt.legend(['Gaussian Fit', 'Histogram'])

# Save the plot as EBG_NOISE.png
plt.savefig('EBG_NOISE.png')




