import pandas as pd

# Define the paths
path_1 = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/processed/features.csv"
path_new = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/parsimony_boot_199_new.csv"

import pandas as pd

# Load the DataFrames (adjust paths if needed)
path_new = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/parsimony_boot_199_new.csv"

df_1 = pd.read_csv(path_1)
df_new = pd.read_csv(path_new)
print(df_new["std_pars_bootsupp_parents"])
# Add suffix "_DEL" to all columns in df_new
df_new = df_new.add_suffix('_DEL')

# Perform an inner merge on 'dataset' and 'branchId'
merged_df = df_1.merge(df_new, how='inner', left_on=['dataset', 'branchId'], right_on=['dataset_DEL', 'branchId_DEL'])
print(merged_df["std_pars_bootstrap_support_parents"])
# Define the column mapping
column_mapping = {
    'mean_pars_bootstrap_support_parents': 'mean_pars_bootsupp_parents_DEL',
    'std_pars_bootstrap_support_parents': 'std_pars_bootsupp_parents_DEL',
    'skewness_bootstrap_pars_support_tree': 'skw_pars_bootsupp_parents_DEL',
    'min_pars_bootstrap_support_children_w': 'min_pars_bootsupp_child_w_DEL',
    'max_pars_bootstrap_support_children_w': 'max_pars_bootsupp_child_w_DEL',
    'std_pars_bootstrap_support_children': 'std_pars_bootstrap_support_children_DEL'
}

# Replace df_1 columns with corresponding values from df_new
for target_col, source_col in column_mapping.items():
    if source_col in merged_df.columns:
        merged_df[target_col] = merged_df[source_col]

# Drop all columns with "_DEL" suffix
merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_DEL')]
print(merged_df["std_pars_bootstrap_support_parents"])

# Save or view the resulting DataFrame
print(merged_df.head())
# Optionally save the DataFrame to a CSV file
# merged_df.to_csv("/path/to/output.csv", index=False)





