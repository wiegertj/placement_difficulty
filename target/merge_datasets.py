import pandas as pd
import os
msa_features = os.path.join(os.pardir, "data/processed/features", "msa_features.csv")
query_features = os.path.join(os.pardir, "data/processed/features", "query_features.csv")
merged_df = query_features.merge(msa_features, on='dataset', how='left')

# add kmer features neotrop
neotrop = merged_df[merged_df['dataset'] == 'neotrop']
file_paths = ['neotrop_query10k_msa_query_kmer8_04_1000.csv', 'neotrop_query10k_msa_query_kmer8_04_2000.csv', 'neotrop_query10k_msa_query_kmer8_04_3000.csv']
dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

neotrop.merge(kmer_features, on="sampleId", how="left")

# add kmer features bv

bv = merged_df[merged_df['dataset'] == 'neotrop']
file_paths = ['neotrop_query10k_msa_query_kmer8_04_1000.csv', 'neotrop_query10k_msa_query_kmer8_04_2000.csv', 'neotrop_query10k_msa_query_kmer8_04_3000.csv']
dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

bv.merge(kmer_features, on="sampleId", how="left")


# add kmer features tara

tara = merged_df[merged_df['dataset'] == 'neotrop']
file_paths = ['neotrop_query10k_msa_query_kmer8_04_1000.csv', 'neotrop_query10k_msa_query_kmer8_04_2000.csv', 'neotrop_query10k_msa_query_kmer8_04_3000.csv']
dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

kmer_features = pd.concat(dataframes, ignore_index=True)

tara.merge(kmer_features, on="sampleId", how="left")

# final dataset
combined_df = pd.concat([neotrop, bv, tara], axis=0, ignore_index=True)
df.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"), index=False)