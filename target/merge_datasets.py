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

query_features200 = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features_200_r1.csv"), index_col=False,
                             usecols=lambda column: column != 'Unnamed: 0')
print(query_features200.shape)
query_features = query_features.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
query_features = pd.concat([query_features, query_features200])

#query_features_lik_diff = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "query_features_lik.csv"), index_col=False,
 #                            usecols=lambda column: column != 'Unnamed: 0')
#query_features = query_features.merge(query_features_lik_diff, on=["dataset", "sampleId"], how="inner")
print("Query feature count: " + str(query_features.shape))

# Get tree features
tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')
like = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "loglik.csv"), index_col=False,
                   usecols=lambda column: column != 'Unnamed: 0')
tree_features = tree_features.merge(like, on=["dataset"], how="inner")
tree_features_embed = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_embedd_stats.csv"),
                                  index_col=False,
                                  usecols=lambda column: column != 'Unnamed: 0')
tree_features_embed = tree_features_embed.drop_duplicates(subset=['dataset'], keep='first')
tree_features = tree_features.merge(tree_features_embed, on=["dataset"], how="inner")
tree_features = tree_features.drop_duplicates(subset=['dataset'], keep='first')
#like = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "loglik.csv"), index_col=False,
 #                  usecols=lambda column: column != 'Unnamed: 0')
#tree_features = tree_features.merge(like, on=["dataset"], how="inner")
print(tree_features.shape)

# Get tree uncertainty features
tree_features_uncertainty = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"),
                                        index_col=False,
                                        usecols=lambda column: column != 'Unnamed: 0')
tree_features_uncertainty = tree_features_uncertainty.drop_duplicates(subset=['dataset'], keep='first')
print(tree_features_uncertainty.shape)

tree_features_uncertainty["dataset"] = tree_features_uncertainty["dataset"].str.replace(".newick", "")
# tree_features = tree_features.merge(tree_features_uncertainty, on="dataset", how="inner")
# t#ree_features = tree_features.merge(difficulties_df[["verbose_name", "difficult"]], left_on="dataset",
#                                  right_on="verbose_name", how="inner").drop(columns=["verbose_name"])

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

        loo_distances_200_filepath = os.path.join(os.pardir, "data/processed/features",  loo_dataset + "_kmer15_03_200_r11000.csv")
        df_200 = pd.read_csv(loo_distances_200_filepath, usecols=lambda column: column != 'Unnamed: 0')

        loo_distances_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/features", loo_dataset + "_200_msa_dist_r1.csv"),
                                    index_col=False, usecols=lambda column: column != 'Unnamed: 0')
        loo_distances_msa["dataset"] = loo_distances_msa["dataset"].str.replace("_reference.fasta", "")
        df_200 = df_200.merge(loo_distances_msa,  on=["sampleId", "dataset"], how="inner")
        loo_resuls_dfs.append(df_200)



    except FileNotFoundError:
        print("Not found kmer: " + loo_distances_200_filepath + " skipped ")
        print("Not found kmer: " + os.path.join(os.pardir, "data/processed/features", loo_dataset + "_200_msa_dist.csv") + " skipped ")

        continue

loo_kmer_distances = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_kmer_distances = loo_kmer_distances.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
print("Kmer similarities shape: " + str(loo_kmer_distances.shape))

# Get entropies for leave one out
loo_entropies = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"),
                            index_col=False, usecols=lambda column: column != 'Unnamed: 0')
loo_entropies = loo_entropies.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')

loo_entropies200 = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy_200_r1.csv"),
                            index_col=False, usecols=lambda column: column != 'Unnamed: 0')

loo_entropies = pd.concat([loo_entropies, loo_entropies200])

print("LOO entropies shape:" + str(loo_entropies.shape))
loo_resuls_combined1 = loo_entropies.merge(loo_kmer_distances, on=["sampleId", "dataset"], how="inner")
print("LOO shape  before merging query features" + str(loo_resuls_combined1.shape))

loo_resuls_combined2 = loo_resuls_combined1.merge(query_features, on=["sampleId", 'dataset'], how='inner')
loo_resuls_combined2.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset_QUERY.csv"), index=False)
print("LOO shape befor dedup features" + str(loo_resuls_combined2.shape))

# Check for duplicates and print them if found
unique_values_combined1 = set(loo_resuls_combined1[['dataset', 'sampleId']].itertuples(index=False, name=None))
unique_values_combined2 = set(loo_resuls_combined2[['dataset', 'sampleId']].itertuples(index=False, name=None))
duplicates = loo_resuls_combined2[loo_resuls_combined2.duplicated(['dataset', 'sampleId'], keep=False)]
unique_duplicates = duplicates[['dataset', 'sampleId']].drop_duplicates()
duplicate_values_list = unique_duplicates.values.tolist()
#print("List of Duplicate Values:")
#for dataset, sampleId in duplicate_valueƒquerys_list:
 #   print(f"Dataset: {dataset}, SampleID: {sampleId}")

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

        file_path_200 = loo_dataset + "16p_200_r1_msa_perc_hash_dist.csv"
        file_path_200 = os.path.join(os.pardir, "data/processed/features", file_path_200)
        df_200 = pd.read_csv(file_path_200, usecols=lambda column: column != 'Unnamed: 0')
        loo_resuls_dfs.append(df_200)


    except FileNotFoundError:
        print(file_path)
        print("Not found Hash Perc: " + loo_dataset)

loo_hash_perc = pd.concat(loo_resuls_dfs, ignore_index=True)
loo_hash_perc = loo_hash_perc.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')

loo_hash_perc["dataset"] = loo_hash_perc["dataset"].str.replace("_reference.fasta", "")
loo_resuls_combined = loo_resuls_combined4.merge(loo_hash_perc, on=["sampleId", 'dataset'], how='inner')
# loo_resuls_combined = loo_resuls_combined4
# add mutation rates

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_subst_freq_stats.csv"
        file_path_200 = loo_dataset + "_subst_freq_stats_200_r1.csv"
        file_path = os.path.join(os.pardir, "data/processed/features", file_path)
        file_path_200 = os.path.join(os.pardir, "data/processed/features", file_path_200)
        df = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
        df200 = pd.read_csv(file_path_200, usecols=lambda column: column != 'Unnamed: 0')
        df200.drop(columns=["taxonId"], inplace=True, axis=1)

        if df.shape[1] != 7:
            print("Found old mutation rates")
            continue
        loo_resuls_dfs.append(df)
        df_copy = df.copy()
        df_copy["sampleId"] = df_copy["sampleId"] + "_200"
        loo_resuls_dfs.append(df_copy)
        loo_resuls_dfs.append(df200)
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
# loo_resuls_combined = loo_resuls_combined.merge(loo_im_comp, on=["sampleId", 'dataset'], how='inner')

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
# loo_resuls_combined = loo_resuls_combined.merge(loo_nearest, on=["sampleId", 'dataset'], how='inner')


loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_diff_site_stats_bad.csv"
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
#loo_resuls_combined = loo_diff
print(loo_resuls_combined.shape)
#loo_resuls_combined = loo_resuls_combined.merge(loo_diff, on=["sampleId", 'dataset'], how='inner')

loo_resuls_dfs = []
elements_to_delete = ['tara', 'bv', "neotrop"]
dataset_list = list(merged_df['dataset'].unique())
loo_datasets = [value for value in dataset_list if value not in elements_to_delete]

for loo_dataset in loo_datasets:
    try:
        file_path = loo_dataset + "_diff_site_stats_good.csv"
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
#loo_resuls_combined = loo_resuls_combined.merge(loo_diff, on=["sampleId", 'dataset'], how='inner')

# final dataset
# combined_df = pd.concat([neotrop, bv, tara, loo_resuls_combined], axis=0, ignore_index=True)
combined_df = loo_resuls_combined

columns_with_nan = combined_df.columns[combined_df.isna().any()].tolist()

columns_to_fill = ['kur_gaps_msa', 'kur_gap_query', 'sk_kmer_sim25', 'kur_kmer_sim25',
                   'sk_kmer_sim50', 'kur_kmer_sim50', 'sk_dist_hu', 'sk_dist_lbp',
                   'kur_dist_lbp', 'sk_dist_pca', 'kur_dist_pca', 'skew_branch_length_tips',
                   'kurtosis_branch_length_tips', 'skewness_nearest', 'kurtosis_nearest',
                   'sk_branch_len_nearest', 'kurt_branch_len_nearest', 'skw_fraction_char_rests8',
                   'kur_fraction_char_rests8', 'skw_fraction_char_rests7', 'kur_fraction_char_rests7',
                   'skw_fraction_char_rests5', 'kur_fraction_char_rests5', "kur_fraction_char_rests9",
                   "skw_fraction_char_rests9"]

for col in columns_with_nan:
    num_nan = combined_df[col].isna().sum()
    print(f"Column '{col}' contains {num_nan} NaN values.")

for column in columns_to_fill:
    if column in combined_df.columns:
        combined_df[column].fillna(-1, inplace=True)
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
##     if index != index_aim:
#       row_set.append(row)
#     avg_perc_ham_dist_msa = pd.DataFrame(row_set)['avg_perc_hash_ham_dist'].mean()
#    std_perc_ham_dist_msa = pd.DataFrame(row_set)['std_perc_hash_ham_dist'].std()
#   min_perc_ham_dist_msa = pd.DataFrame(row_set)['min_perc_hash_ham_dist'].min()
#  max_perc_ham_dist_msa = pd.DataFrame(row_set)['max_perc_hash_ham_dist'].max()
#     avg_kmer_sim_msa = pd.DataFrame(row_set)['mean_kmer_sim'].mean()
#    std_kmer_sim_msa = pd.DataFrame(row_set)['std_kmer_sim'].std()
#   min_kmer_sim_msa = pd.DataFrame(row_set)['min_kmer_sim'].min()
#  max_kmer_sim_msa = pd.DataFrame(row_set)['max_kmer_sim'].max()
# combined_df.at[index_aim, "avg_perc_ham_dist_msa"] = avg_perc_ham_dist_msa
# combined_df.at[index_aim, "std_perc_ham_dist_msa"] = std_perc_ham_dist_msa
# combined_df.at[index_aim, "min_perc_ham_dist_msa"] = min_perc_ham_dist_msa
#    combined_df.at[index_aim, "max_perc_ham_dist_msa"] = max_perc_ham_dist_msa
#   combined_df.at[index_aim, "avg_kmer_simt_msa"] = avg_kmer_sim_msa
#  combined_df.at[index_aim, "std_kmer_simt_msa"] = std_kmer_sim_msa
# combined_df.at[index_aim, "min_kmer_simt_msa"] = min_kmer_sim_msa
# combined_df.at[index_aim, "max_kmer_simt_msa"] = max_kmer_sim_msa##


def sample_rows(group):
    percentile = group["percentile"].iloc[0]
    print(percentile)
    max_sample_size = min(1800, group.shape[0])
    # if percentile <= 4.0:
    #   max_sample_size = 1000
    # else:
    #   print(group.shape)

    #  return group
    print(group.shape)
    return group.sample(max_sample_size)


print(combined_df.shape)

difficulties_df = combined_df.drop_duplicates(subset=['dataset', "sampleId"], keep='first')

print(combined_df.shape)
bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
combined_df['percentile'] = pd.cut(combined_df['entropy'], bins=bin_edges, labels=False, include_lowest=True)
uniformly_distributed_df = combined_df.groupby('percentile', group_keys=False).apply(sample_rows)
print(uniformly_distributed_df["entropy"].median())
print(uniformly_distributed_df["entropy"].mean())
uniformly_distributed_df.drop(columns=["percentile"], inplace=True)
print(uniformly_distributed_df.shape)
print(list(uniformly_distributed_df.columns))
combined_df = combined_df.drop_duplicates(subset=['dataset', 'sampleId'], keep='first')
combined_df = combined_df[combined_df['sampleId'].str.contains('_')]
print(combined_df.shape)

combined_df.to_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset_noboot_no_filter_r1.csv"), index=False)
