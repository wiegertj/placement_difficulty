import os
import warnings
import pandas as pd

# Suppress FutureWarning from the str.replace() call
warnings.filterwarnings("ignore", "FutureWarning")


msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "m"
                                                                              "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
print("MSA feature count: " + str(msa_features.shape))
query_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index_col=False,
                             usecols=lambda column: column != 'Unnamed: 0')
print("Query feature count: " + str(query_features.shape))
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
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
        file_path = loo_dataset + "_msa_perc_hash_dist.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print("Not found Hash Perc: " + loo_dataset + " " + str(loo_dataset.shape))

loo_hash_perc = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_hash_perc["dataset"] = loo_hash_perc["dataset"].str.replace("_reference.fasta", "")
loo_resuls_combined = loo_resuls_combined.merge(loo_hash_perc, on=["sampleId", 'dataset'], how='inner')

# final dataset
#combined_df = pd.concat([neotrop, bv, tara, loo_resuls_combined], axis=0, ignore_index=True)
combined_df = loo_resuls_combined
print(combined_df['dataset'].unique())
print(combined_df.shape)
combined_df.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"), index=False)
