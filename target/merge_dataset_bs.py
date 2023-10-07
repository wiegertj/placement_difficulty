import warnings
import pandas as pd
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get targets and branch features
targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target/branch_supports.csv"),
                      usecols=lambda column: column != 'Unnamed: 0')

subst = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features", "subst_freq_stats_bs.csv"))

# Get MSA features
msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features",
                                        "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
msa_features = msa_features.drop_duplicates(subset=['dataset'], keep='first')

# Get tree features
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
tree_features = tree_features.drop_duplicates(subset=['dataset'], keep='first')
like = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "loglik.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
tree_features = tree_features.merge(like, on=["dataset"], how="inner")

# Get parsimony features
parsimony_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features/parsimony.csv"),
                                 usecols=lambda column: column != 'Unnamed: 0')
parsimony_features2 = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features.csv"),
                                  usecols=lambda column: column != 'Unnamed: 0')
parsimony_features3 = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features/parsimony_boot.csv"),
                                  usecols=lambda column: column != 'Unnamed: 0')
print(parsimony_features2.columns)
# Get split features
split_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                          "split_features.csv"), usecols=lambda column: column != 'Unnamed: 0')
split_features2 = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                          "split_features_tree.csv"), usecols=lambda column: column != 'Unnamed: 0')

msa_features.drop_duplicates(inplace=True, subset=["dataset"])
tree_features.drop_duplicates(inplace=True, subset=["dataset"])
parsimony_features.drop_duplicates(inplace=True, subset=["dataset", "branchId"])
parsimony_features2.drop_duplicates(inplace=True, subset=["dataset"])
parsimony_features3.drop_duplicates(inplace=True, subset=["dataset", "branchId"])
split_features2.drop_duplicates(inplace=True, subset=["dataset", "branchId"])
split_features.drop_duplicates(inplace=True, subset=["dataset", "branchId"])
subst.drop_duplicates(inplace=True, subset=["dataset"])

df_merged = targets.merge(msa_features, on=["dataset"], how="inner")
df_merged = df_merged.merge(tree_features, on=["dataset"], how="inner")
df_merged = df_merged.merge(parsimony_features, on=["dataset", "branchId"], how="inner")
df_merged = df_merged.merge(split_features2, on=["dataset", "branchId"], how="inner")
df_merged = df_merged.merge(split_features, on=["dataset", "branchId"], how="inner")
df_merged = df_merged.merge(subst, on=["dataset"], how="inner")
df_merged = df_merged.merge(parsimony_features2, on=["dataset"], how="inner")
#df_merged = df_merged.merge(parsimony_features3, on=["dataset", "branchId"], how="inner")

#df_merged['split_skw_ratio_topo'].fillna(-1, inplace=True)
#df_merged['split_skw_ratio_branch'].fillna(-1, inplace=True)
#df_merged['split_skw_entropy_diff'].fillna(-1, inplace=True)
df_merged['skew_branch_length_inner'].fillna(-1, inplace=True)
df_merged['skew_irs'].fillna(-1, inplace=True)
df_merged['skew_branch_length_tips'].fillna(-1, inplace=True)
df_merged['kur_gaps_msa'].fillna(-1, inplace=True)
df_merged['kur_entropy_msa'].fillna(-1, inplace=True)
df_merged['kurtosis_branch_length_tips'].fillna(-1, inplace=True)
df_merged['kurtosis_branch_length_inner'].fillna(-1, inplace=True)
df_merged['kurtosis_irs'].fillna(-1, inplace=True)
df_merged['kur_clo_sim'].fillna(-1, inplace=True)
df_merged['kur_eig_sim'].fillna(-1, inplace=True)
df_merged['irs_skw_ratio'].fillna(-1, inplace=True)
df_merged['skw_clo_sim_ratio'].fillna(-1, inplace=True)

step_size = 0.1
max_samples_per_interval = 6000

# Initialize an empty DataFrame to store the sampled data
sampled_data = pd.DataFrame()

# Iterate through the support intervals
for support_start in np.arange(0, 1, step_size):
    support_end = support_start + step_size

    # Filter the data within the current support interval
    interval_data = df_merged[(df_merged['support'] >= support_start) & (df_merged['support'] < support_end)]

    # Determine the number of samples to select
    num_samples = min(max_samples_per_interval, len(interval_data))

    # Randomly sample the data within the interval
    sampled_interval = interval_data.sample(n=num_samples, random_state=42)  # Adjust the random_state as needed
    print(support_start)
    print(num_samples)

    # Append the sampled interval to the result DataFrame
    sampled_data = pd.concat([sampled_data, sampled_interval])

# Display the sampled data
sampled_data.drop_duplicates(inplace=True, subset=["dataset", "branchId"])
print(sampled_data.shape)
print("Mean " + str(sampled_data["support"].median()))
print(sampled_data.columns)

sampled_data.to_csv(os.path.join(os.pardir, "data/processed/final/bs_support.csv"), index=False)

print(df_merged.shape)
