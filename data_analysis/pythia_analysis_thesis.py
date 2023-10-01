import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "diff_no_reest.csv"))
df["difference"] = df["difficulty_loo_no_reest"] - df["difficulty_loo_reest"]
print(max(df["difference"]))
import pandas as pd
from scipy.stats import spearmanr
#df = df[abs(df["difference"])>=0.0]



import matplotlib.pyplot as plt

# Assuming df is your DataFrame and "diff" is the column you want to plot
plt.hist(df['difference'], bins=20, edgecolor='k')  # Adjust the number of bins as needed

plt.xlabel("Difference")
plt.ylabel("Frequency")

plt.show()




#df = df[df["dataset"] == "10942_0"]
print(df.shape)
print(set(df["dataset"]))
feature_pairs = [
    ('avg_rfdist_parsimony_base_val', 'avg_rfdist_parsimony_loo_val'),
    ('proportion_unique_topos_parsimony_base_val', 'proportion_unique_topos_parsimony_loo_val'),
    ('proportion_invariant_base_val', 'proportion_invariant_loo_val'),
    ('num_patterns/num_taxa_base_val', 'num_patterns/num_taxa_loo_val'),
    ('bollback_base_val', 'bollback_loo_val'),
    ('proportion_gaps_base_val', 'proportion_gaps_loo_val'),
    ('pattern_entropy_base_val', "pattern_entropy_loo_val"),
    ('entropy_base_val', 'entropy_loo_val'),
    ('num_patterns/num_sites_base_val', 'num_patterns/num_sites_loo_val'),
    ('num_sites/num_taxa_base_val', "num_sites/num_taxa_loo_val"),

    ('avg_rfdist_parsimony_base_exp', 'avg_rfdist_parsimony_loo_exp'),
    ('proportion_unique_topos_parsimony_base_exp', 'proportion_unique_topos_parsimony_loo_exp'),
    ('proportion_invariant_base_exp', 'proportion_invariant_loo_exp'),
    ('num_patterns/num_taxa_base_exp', 'num_patterns/num_taxa_loo_exp'),
    ('bollback_base_exp', 'bollback_loo_exp'),
    ('proportion_gaps_base_exp', 'proportion_gaps_loo_exp'),
    ('pattern_entropy_base_exp', "pattern_entropy_loo_exp"),
    ('entropy_base_exp', 'entropy_loo_exp'),
    ('num_patterns/num_sites_base_exp', 'num_patterns/num_sites_loo_exp')
    # Add more feature pairs as needed
]



for base_feature, loo_feature in feature_pairs:
    new_column_name = f'{base_feature}_diff'
    print(base_feature)
    print(loo_feature)
    df[new_column_name] = df[base_feature] - df[loo_feature]


# Assuming df is your DataFrame
columns_to_exclude = ["dataset", "sampleId", "difficulty_loo_no_reest", "difficulty_loo_reest"]

# Filter the columns to include only those not in the exclusion list
columns_to_include = [col for col in df.columns if col not in columns_to_exclude]

# Calculate Spearman correlations and p-values
correlations = {}

for column in columns_to_include:
    spearman_corr, p_value = spearmanr(df["difference"], df[column])
    correlations[column] = {"Spearman Correlation": spearman_corr, "p-value": p_value}

# Sort correlations by Spearman Correlation in descending order
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]["Spearman Correlation"]), reverse=True)

high_correlation_pairs = []

# Iterate through sorted_correlations and check if the correlation exceeds the threshold
for column, corr_info in sorted_correlations:
    if abs(corr_info["Spearman Correlation"]) >= 0.5:
        high_correlation_pairs.append((column, corr_info["Spearman Correlation"]))

# Create scatter plots for high correlation pairs and save them
for column, correlation in high_correlation_pairs:
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.scatterplot(data=df, x="difference", y=column)
    plt.title(f'Scatter Plot for {column}\nSpearman Correlation: {correlation:.2f}')
    plt.xlabel('Difference')
    plt.ylabel(column)
    plt.grid(True)
    plt.savefig(f'PYTHIA_{column}.png'.replace("/","_"))  # Save the scatter plot with a unique filename
    plt.close()

# Print the sorted list of correlations and p-values
for column, values in sorted_correlations:
    print(f"{column}:")
    print(f"  Spearman Correlation: {values['Spearman Correlation']}")
    print(f"  p-value: {values['p-value']}")
    print()
