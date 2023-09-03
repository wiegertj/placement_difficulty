import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df["verbose_name"] = difficulties_df["verbose_name"].str.replace(".phy", "")

msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features",
                                        "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
print("MSA feature count: " + str(msa_features.shape))
query_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index_col=False,
                             usecols=lambda column: column != 'Unnamed: 0')
print("Query feature count: " + str(query_features.shape))
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
tree_features_uncertainty = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"), index_col=False,
                                        usecols=lambda column: column != 'Unnamed: 0')
tree_features_uncertainty["dataset"] = tree_features_uncertainty["dataset"].str.replace(".newick", "")
tree_features = tree_features.merge(tree_features_uncertainty, on="dataset", how="inner")
print("Tree feature count after diff merging" + str(tree_features.shape))
print(tree_features.tail(10))

tree_features = tree_features.merge(difficulties_df[["verbose_name", "difficult"]], left_on="dataset",
                                    right_on="verbose_name", how="inner").drop(columns=["verbose_name"])
print(tree_features.tail(10))
print("Tree feature count: " + str(tree_features.shape))

merged_df = query_features.merge(msa_features, on='dataset', how='inner')
merged_df = merged_df.merge(tree_features, on="dataset", how="inner")

# add kmer features neotrop
neotrop = merged_df[merged_df['dataset'] == 'neotrop']
file_paths = ['neotrop_5000.csv']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

neotrop = neotrop.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
neotrop_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "neotrop_msa_dist.csv"),
                                index_col=False, usecols=lambda column: column != 'Unnamed: 0')
neotrop = neotrop.merge(neotrop_distances, on=["sampleId", "dataset"], how="inner")
neotrop_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "neotrop_10k_epa_result_entropy.csv"),
                                index_col=False, usecols=lambda column: column != 'Unnamed: 0')
neotrop = neotrop.merge(neotrop_entropies, on="sampleId", how="inner")

# add percetual hashing distances neotrop
file_path_hash = os.path.join(os.pardir, "data/processed/features", "neotrop_msa_perc_hash_dist.csv")
df = pd.read_csv(file_path_hash, usecols=lambda column: column != 'Unnamed: 0')
neotrop = neotrop.merge(df, on=["sampleId", "dataset"], how="inner")

# add kmer features bv

bv = merged_df[merged_df['dataset'] == 'bv']
file_paths = ['bv_5000.csv']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

bv = bv.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
bv_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "bv_msa_dist.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
bv = bv.merge(bv_distances, on=["sampleId", "dataset"], how="inner")
bv_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "bv_epa_result_entropy.csv"),
                           index_col=False, usecols=lambda column: column != 'Unnamed: 0')
bv = bv.merge(bv_entropies, on="sampleId", how="inner")

# add percetual hashing bv
file_path_hash = os.path.join(os.pardir, "data/processed/features", "bv_msa_perc_hash_dist.csv")
df = pd.read_csv(file_path_hash, usecols=lambda column: column != 'Unnamed: 0')
bv = bv.merge(df, on=["sampleId", "dataset"], how="inner")

# add kmer features tara

tara = merged_df[merged_df['dataset'] == 'tara']
file_paths = ['tara_5000.csv']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

tara = tara.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
tara_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tara_msa_dist.csv"), index_col=False,
                             usecols=lambda column: column != 'Unnamed: 0')
tara = tara.merge(tara_distances, on=["sampleId", "dataset"], how="inner")
tara_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "tara_epa_result_entropy.csv"),
                             index_col=False, usecols=lambda column: column != 'Unnamed: 0')
tara = tara.merge(tara_entropies, on="sampleId", how="inner")

# add percetual hashing tara
file_path_hash = os.path.join(os.pardir, "data/processed/features", "tara_msa_perc_hash_dist.csv")
df = pd.read_csv(file_path_hash, usecols=lambda column: column != 'Unnamed: 0')
tara = tara.merge(df, on=["sampleId", "dataset"], how="inner")

# add kmer features loo

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:

    loo = merged_df[merged_df['dataset'] == loo_dataset]
    file_path = loo_dataset + "_kmer15_03_1000.csv"
    try:
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        if df.shape[1] != 8:
            print("found old kmers " + loo_dataset)
            print("shape " + str(df.shape))
            continue
    except FileNotFoundError:
        print("Not found kmer: " + file_path + " skipped " + str(loo.shape))
        continue

    loo_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", loo_dataset + "_msa_dist.csv"),
                                index_col=False, usecols=lambda column: column != 'Unnamed: 0')
    loo_distances["dataset"] = loo_distances["dataset"].str.replace("_reference.fasta", "")
    df = df.merge(loo_distances, on=["sampleId", "dataset"], how="inner")
    loo_resuls_dfs.append(df)

loo_resuls_combined = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_resuls_dfs = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"),
                             index_col=False, usecols=lambda column: column != 'Unnamed: 0')
loo_resuls_combined = loo_resuls_combined.merge(loo_resuls_dfs, on=["sampleId", "dataset"], how="inner")
loo_resuls_combined = loo_resuls_combined.merge(query_features, on=["sampleId", 'dataset'], how='inner')
loo_resuls_combined = loo_resuls_combined.merge(tree_features, on='dataset', how='inner')
loo_resuls_combined = loo_resuls_combined.merge(msa_features, on='dataset', how='inner')

# add perc hashing distance loo

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "16p_msa_perc_hash_dist.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        if df.shape[1] != 38:
            print("Found old hash perc, skipped ")
            continue
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found Hash Perc: " + loo_dataset)

loo_hash_perc = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_hash_perc["dataset"] = loo_hash_perc["dataset"].str.replace("_reference.fasta", "")
loo_resuls_combined = loo_resuls_combined.merge(loo_hash_perc, on=["sampleId", 'dataset'], how='inner')

# add mutation rates

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_subst_freq_stats.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        if df.shape[1] != 7:
            print("Found old mutation rates")
            continue
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found Mutation Stats: " + loo_dataset)

loo_subst = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_resuls_combined = loo_resuls_combined.merge(loo_subst, on=["sampleId", 'dataset'], how='inner')

# add image comp

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_msa_im_comp.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found Image Comp Stats: " + loo_dataset)

loo_im_comp = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_resuls_combined = loo_resuls_combined.merge(loo_im_comp, on=["sampleId", 'dataset'], how='inner')

# final dataset
# combined_df = pd.concat([neotrop, bv, tara, loo_resuls_combined], axis=0, ignore_index=True)
combined_df = loo_resuls_combined

columns_with_nan = combined_df.columns[combined_df.isna().any()].tolist()

for col in columns_with_nan:
    num_nan = combined_df[col].isna().sum()
    print(f"Column '{col}' contains {num_nan} NaN values.")

combined_df['kur_gaps_msa'].fillna(-1, inplace=True)
combined_df['kur_gap_query'].fillna(-1, inplace=True)
combined_df['sk_kmer_sim25'].fillna(-1, inplace=True)
combined_df['kur_kmer_sim25'].fillna(-1, inplace=True)
combined_df['sk_dist_hu'].fillna(-1, inplace=True)
combined_df['sk_dist_lbp'].fillna(-1, inplace=True)
combined_df['kur_dist_lbp'].fillna(-1, inplace=True)
combined_df['sk_dist_pca'].fillna(-1, inplace=True)
combined_df['kur_dist_pca'].fillna(-1, inplace=True)
combined_df['skew_branch_length_tips'].fillna(-1, inplace=True)
combined_df['kurtosis_branch_length_tips'].fillna(-1, inplace=True)

print("Dropping NaN")
print(combined_df.shape)
print("After NaN")
combined_df.dropna(axis=0, inplace=True)
print(combined_df.shape)
print(combined_df.entropy)
combined_df["entropy"] = combined_df['entropy'].astype(float)

print(combined_df['dataset'].unique())
print(combined_df.shape)

print("Create uniform sample")
combined_df.loc[combined_df['entropy'] > 1, 'entropy'] = 1


def sample_rows(group):
    percentile = group["percentile"].iloc[0]
    print(percentile)
    if percentile <= 4.0:
        max_sample_size = min(2800, len(group))
    else:
        return group
    return group.sample(max_sample_size)


print(combined_df.shape)
bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
combined_df['percentile'] = pd.cut(combined_df['entropy'], bins=bin_edges, labels=False, include_lowest=True)
uniformly_distributed_df = combined_df.groupby('percentile', group_keys=False).apply(sample_rows)
print(uniformly_distributed_df["entropy"].median())
print(uniformly_distributed_df["entropy"].mean())
uniformly_distributed_df.drop(columns=["percentile"], inplace=True)
print(uniformly_distributed_df.shape)
uniformly_distributed_df.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"), index=False)
