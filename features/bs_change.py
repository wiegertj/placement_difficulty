import pandas as pd

# Define the paths
path_1 = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/processed/features.csv"
path_new = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/parsimony_boot_199_new.csv"

import pandas as pd

# Load the DataFrames (adjust paths if needed)
path_1 = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/processed"
path_new = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/parsimony_boot_199_new.csv"

df_1 = pd.read_csv(path_1)
df_new = pd.read_csv(path_new)

# Define the column mapping from df_new to df_1
column_mapping = {
    'mean_pars_bootsupp_parents': 'mean_pars_bootstrap_support_parents',
    'std_pars_bootsupp_parents': 'std_pars_bootstrap_support_parents',
    'skw_pars_bootsupp_parents': 'skewness_bootstrap_pars_support_tree',
    'min_pars_bootsupp_child_w': 'min_pars_bootstrap_support_children_w',
    'max_pars_bootsupp_child_w': 'max_pars_bootstrap_support_children_w',
    'std_pars_bootstrap_support_children': 'std_pars_bootstrap_support_children'
}

# Reverse the mapping for renaming purposes
rename_mapping = {v: k for k, v in column_mapping.items()}

# Keep only df_1 columns
df_1_filtered = df_1[list(rename_mapping.keys())]

# Rename df_new columns for consistency with df_1
df_new_renamed = df_new.rename(columns=rename_mapping)

# Merge on the common columns (you can adjust `on` or `how` depending on your merge logic)
merged_df = df_1_filtered.merge(df_new_renamed, on=rename_mapping.keys(), how='inner')

# Replace df_1 values with those from df_new
for old_col, new_col in rename_mapping.items():
    if new_col in df_new.columns:
        merged_df[old_col] = merged_df[new_col]

# Drop unnecessary columns if needed
merged_df = merged_df[df_1_filtered.columns]

# Save or view the resulting DataFrame
print(merged_df.head())
# Optionally save the DataFrame to a CSV file
merged_df.to_csv("/hits/fast/cme/wiegerjs/placement_difficulty/features_new.csv", index=False)






