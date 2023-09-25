import warnings
import pandas as pd
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get targets and branch features
targets = pd.read_csv(os.path.join(os.pardir, "data/processed/target/branch_supports.csv"), usecols=lambda column: column != 'Unnamed: 0')

# Get MSA features
msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features",
                                        "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
msa_features = msa_features.drop_duplicates(subset=['dataset'], keep='first')

# Get tree features
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
tree_features = tree_features.drop_duplicates(subset=['dataset'], keep='first')

# Get parsimony features
parsimony_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features/parsimony.csv"), usecols=lambda column: column != 'Unnamed: 0')

# Get split features
split_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                  "split_features.csv"), usecols=lambda column: column != 'Unnamed: 0')

df_merged = targets.merge(msa_features, on=["dataset"], how="inner")
df_merged = df_merged.merge(tree_features, on=["dataset"], how="inner")
df_merged = df_merged.merge(parsimony_features, on=["dataset", "branchId"], how="inner")
df_merged = df_merged.merge(split_features, on=["dataset", "branchId"], how="inner")
df_merged.to_csv(os.path.join(os.pardir, "data/processed/final/bs_support.csv"), index=False)
df_merged['split_skw_ratio_topo'].fillna(-1, inplace=True)
df_merged['split_skw_ratio_branch'].fillna(-1, inplace=True)
df_merged['split_skw_entropy_diff'].fillna(-1, inplace=True)

print(df_merged.shape)

