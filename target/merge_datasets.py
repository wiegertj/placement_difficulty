import pandas as pd
import os
msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
query_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
tree_features =  pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
merged_df = query_features.merge(msa_features, on='dataset', how='inner')
merged_df = merged_df.merge(tree_features, on="dataset", how="inner")



# add kmer features neotrop
neotrop = merged_df[merged_df['dataset'] == 'neotrop']
file_paths = ['neotrop_query10k_msa_query_kmer8_04_1000', 'neotrop_query10k_msa_query_kmer8_04_2000', 'neotrop_query10k_msa_query_kmer8_04_3000']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

neotrop = neotrop.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
neotrop_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "neotrop_query_msa_dist.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
neotrop = neotrop.merge(neotrop_distances, on=["sampleId", "dataset"], how="inner")
neotrop_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "neotrop_10k_epa_result_entropy.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
print(neotrop_entropies.columns)
neotrop = neotrop.merge(neotrop_entropies, on="sampleId", how="inner")

print(neotrop)


# add kmer features bv

bv = merged_df[merged_df['dataset'] == 'bv']
file_paths = ['bv_query_msa_query_kmer8_04_1000', 'bv_query_msa_query_kmer8_04_2000']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

bv = bv.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
bv_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "bv_query_msa_dist.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
bv = bv.merge(bv_distances, on=["sampleId", "dataset"], how="inner")
bv_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "bv_epa_result_entropy.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
bv = bv.merge(bv_entropies, on="sampleId", how="inner")
print(bv)



# add kmer features tara

tara = merged_df[merged_df['dataset'] == 'tara']
file_paths = ['tara_query_msa_query_kmer8_04_1000', 'tara_query_msa_query_kmer8_04_2000']
dataframes = []

for file_path in file_paths:
    file_path = os.path.join(os.pardir, "data/processed/features", file_path)
    df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

tara = tara.merge(kmer_features, on=["sampleId", "dataset"], how="inner")
tara_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tara_query_msa_dist.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
tara = tara.merge(tara_distances, on=["sampleId", "dataset"], how="inner")
tara_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "tara_epa_result_entropy.csv"), index_col=False, usecols=lambda column: column != 'Unnamed: 0')
tara = tara.merge(tara_entropies, on="sampleId", how="inner")
print(tara.shape)

# final dataset
combined_df = pd.concat([neotrop, bv, tara], axis=0, ignore_index=True)
print(combined_df.shape)
combined_df.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"), index=False)