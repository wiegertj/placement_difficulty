import os
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Get difficulties
difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df = difficulties_df.drop_duplicates(subset=['verbose_name'], keep='first')
difficulties_df["verbose_name"] = difficulties_df["verbose_name"].str.replace(".phy", "")

# Get MSA features
msa_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features",
                                        "msa_features.csv"), index_col=False,
                           usecols=lambda column: column != 'Unnamed: 0')
msa_features = msa_features.drop_duplicates(subset=['dataset'], keep='first')

# Get query features
print("MSA feature count: " + str(msa_features.shape))
query_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features.csv"), index_col=False,
                             usecols=lambda column: column != 'Unnamed: 0')
query_features = query_features.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
print("Query feature count: " + str(query_features.shape))

# Get tree features
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
tree_features = tree_features.drop_duplicates(subset=['dataset'], keep='first')
print(tree_features.shape)

# Get tree uncertainty features
tree_features_uncertainty = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"),
                                        index_col=False,
                                        usecols=lambda column: column != 'Unnamed: 0')
tree_features_uncertainty = tree_features_uncertainty.drop_duplicates(subset=['dataset'], keep='first')
print(tree_features_uncertainty.shape)


tree_features_uncertainty["dataset"] = tree_features_uncertainty["dataset"].str.replace(".newick", "")
tree_features = tree_features.merge(tree_features_uncertainty, on="dataset", how="inner")
tree_features = tree_features.merge(difficulties_df[["verbose_name", "difficult"]], left_on="dataset",
                                    right_on="verbose_name", how="inner").drop(columns=["verbose_name"])

merged_df = query_features.merge(msa_features, on='dataset', how='inner')
merged_df = merged_df.merge(tree_features, on="dataset", how="inner")

# Get kmer features
loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:

    file_path = loo_dataset + "_kmer15_03_1000.csv"
    try:
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        if df.shape[1] != 8:
            print("found old kmers " + loo_dataset)
            print("shape " + str(df.shape))
            continue
        loo_distances = pd.read_csv(os.path.join(os.pardir, "data/processed/features", loo_dataset + "_msa_dist.csv"),
                                    index_col=False, usecols=lambda column: column != 'Unnamed: 0')
        loo_distances["dataset"] = loo_distances["dataset"].str.replace("_reference.fasta", "")
        df = df.merge(loo_distances, on=["sampleId", "dataset"], how="inner")
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print("Not found kmer: " + file_path + " skipped ")
        continue

loo_kmer_distances = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_kmer_distances = loo_kmer_distances.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
print("Kmer similarities shape: " + str(loo_kmer_distances.shape))

# Get entropies for leave one out
loo_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"),
                            index_col=False, usecols=lambda column: column != 'Unnamed: 0')
loo_entropies = loo_entropies.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')

print("LOO entropies shape:" + str(loo_entropies.shape))
loo_resuls_combined1 = loo_entropies.merge(loo_kmer_distances, on=["sampleId", "dataset"], how="inner")

loo_resuls_combined2 = loo_resuls_combined1.merge(query_features, on=["sampleId", 'dataset'], how='inner')
loo_resuls_combined2.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset_QUERY.csv"), index=False)

# Check for duplicates and print them if found
unique_values_combined1 = set(loo_resuls_combined1[['dataset', 'sampleId']].itertuples(index=False, name=None))
unique_values_combined2 = set(loo_resuls_combined2[['dataset', 'sampleId']].itertuples(index=False, name=None))
duplicates = loo_resuls_combined2[loo_resuls_combined2.duplicated(['dataset', 'sampleId'], keep=False)]
unique_duplicates = duplicates[['dataset', 'sampleId']].drop_duplicates()
duplicate_values_list = unique_duplicates.values.tolist()
print("List of Duplicate Values:")
for dataset, sampleId in duplicate_values_list:
    print(f"Dataset: {dataset}, SampleID: {sampleId}")

print("LOO shape after merging query features" + str(loo_resuls_combined2.shape))
loo_resuls_combined3 = loo_resuls_combined2.merge(tree_features, on='dataset', how='inner')
print("LOO shape after merging tree features" + str(loo_resuls_combined3.shape))
loo_resuls_combined4 = loo_resuls_combined3.merge(msa_features, on='dataset', how='inner')
print("LOO shape after merging MSA features" + str(loo_resuls_combined4.shape))

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "16p_msa_perc_hash_dist.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        if df.shape[1] != 45:
            print("Found old hash perc, skipped ")
            continue
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found Hash Perc: " + loo_dataset)

loo_hash_perc = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_hash_perc = loo_hash_perc.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')

loo_hash_perc["dataset"] = loo_hash_perc["dataset"].str.replace("_reference.fasta", "")
loo_resuls_combined = loo_resuls_combined4.merge(loo_hash_perc, on=["sampleId", 'dataset'], how='inner')

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
loo_subst = loo_subst.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
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
loo_im_comp = loo_im_comp.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
loo_resuls_combined = loo_resuls_combined.merge(loo_im_comp, on=["sampleId", 'dataset'], how='inner')

# add nearest sequence support

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_nearest_bootstrap.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found nearest sequence features: " + loo_dataset)

loo_nearest = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_nearest = loo_nearest.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
loo_resuls_combined = loo_resuls_combined.merge(loo_nearest, on=["sampleId", 'dataset'], how='inner')






loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_diff_site_stats.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        loo_resuls_dfs.append(df)
    except FileNotFoundError:
        print(file_path)
        print("Not found diff site stats: " + loo_dataset)

loo_diff = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_diff["temp"] = loo_diff["dataset"]
loo_diff["dataset"] = loo_diff["sampleId"]
loo_diff["sampleId"] = loo_diff["temp"]
loo_diff.drop(columns=["temp"], inplace=True, axis=1)

loo_diff = loo_diff.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
loo_resuls_combined = loo_resuls_combined.merge(loo_diff, on=["sampleId", 'dataset'], how='inner')





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
combined_df['sk_kmer_sim50'].fillna(-1, inplace=True)
combined_df['kur_kmer_sim50'].fillna(-1, inplace=True)
combined_df['sk_dist_hu'].fillna(-1, inplace=True)
combined_df['sk_dist_lbp'].fillna(-1, inplace=True)
combined_df['kur_dist_lbp'].fillna(-1, inplace=True)
combined_df['sk_dist_pca'].fillna(-1, inplace=True)
combined_df['kur_dist_pca'].fillna(-1, inplace=True)
combined_df['skew_branch_length_tips'].fillna(-1, inplace=True)
combined_df['kurtosis_branch_length_tips'].fillna(-1, inplace=True)
combined_df['skewness_nearest'].fillna(-1, inplace=True)
combined_df['kurtosis_nearest'].fillna(-1, inplace=True)
combined_df['sk_branch_len_nearest'].fillna(-1, inplace=True)
combined_df['kurt_branch_len_nearest'].fillna(-1, inplace=True)
combined_df['skw_fraction_char_rests8'].fillna(-1, inplace=True)
combined_df['kur_fraction_char_rests8'].fillna(-1, inplace=True)
combined_df['skw_fraction_char_rests7'].fillna(-1, inplace=True)
combined_df['kur_fraction_char_rests7'].fillna(-1, inplace=True)
combined_df['skw_fraction_char_rests5'].fillna(-1, inplace=True)
combined_df['kur_fraction_char_rests5'].fillna(-1, inplace=True)
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


# combined_df["avg_perc_ham_dist_msa"] = 0.0
# combined_df["std_perc_ham_dist_msa"] = 0.0
# combined_df["min_perc_ham_dist_msa"] = 0.0
# combined_df["max_perc_ham_dist_msa"] = 0.0
# combined_df["avg_kmer_simt_msa"] = 0.0
# combined_df["min_kmer_simt_msa"] = 0.0
# combined_df["max_kmer_simt_msa"] = 0.0
# combined_df["std_kmer_simt_msa"] = 0.0

# dataset_counter = 0
# for dataset in combined_df['dataset'].unique():
#   dataset_counter += 1
#  print(dataset_counter)
# subset = combined_df[combined_df['dataset'] == dataset]
# for index_aim, row_aim in subset.iterrows():
#   row_set = []
#  for index, row in subset.iterrows():
#     if index != index_aim:
#        row_set.append(row)
# avg_perc_ham_dist_msa = pd.DataFrame(row_set)['avg_perc_hash_ham_dist'].mean()
# std_perc_ham_dist_msa = pd.DataFrame(row_set)['std_perc_hash_ham_dist'].std()
#   min_perc_ham_dist_msa = pd.DataFrame(row_set)['min_perc_hash_ham_dist'].min()
#  max_perc_ham_dist_msa = pd.DataFrame(row_set)['max_perc_hash_ham_dist'].max()
# avg_kmer_sim_msa = pd.DataFrame(row_set)['mean_kmer_sim'].mean()
# std_kmer_sim_msa = pd.DataFrame(row_set)['std_kmer_sim'].std()
#    min_kmer_sim_msa = pd.DataFrame(row_set)['min_kmer_sim'].min()
#   max_kmer_sim_msa = pd.DataFrame(row_set)['max_kmer_sim'].max()
#  combined_df.at[index_aim, "avg_perc_ham_dist_msa"] = avg_perc_ham_dist_msa
# combined_df.at[index_aim, "std_perc_ham_dist_msa"] = std_perc_ham_dist_msa
# combined_df.at[index_aim, "min_perc_ham_dist_msa"] = min_perc_ham_dist_msa
# combined_df.at[index_aim, "max_perc_ham_dist_msa"] = max_perc_ham_dist_msa
# combined_df.at[index_aim, "avg_kmer_simt_msa"] = avg_kmer_sim_msa
# combined_df.at[index_aim, "std_kmer_simt_msa"] = std_kmer_sim_msa
# combined_df.at[index_aim, "min_kmer_simt_msa"] = min_kmer_sim_msa
# combined_df.at[index_aim, "max_kmer_simt_msa"] = max_kmer_sim_msa


def sample_rows(group):
    percentile = group["percentile"].iloc[0]
    print(percentile)
    if percentile <= 3.0:
        max_sample_size = min(2200, len(group))
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
