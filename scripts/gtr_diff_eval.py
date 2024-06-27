import os
import subprocess
from ete3 import Tree
import pandas as pd

# Load the CSV file
file_path = "/hits/fast/cme/wiegerjs/model_test_res.csv"
df = pd.read_csv(file_path)

# Filter out rows where best_model_bic == 'error'
filtered_df = df[df['best_model_bic'] != 'error']
filtered_df = filtered_df[filtered_df['best_model_bic'] != 'GTR+G']
filtered_df = filtered_df[filtered_df['best_model_bic'] != 'GTR+G+I']
# Select the first 200 rows from the filtered DataFrame
sampled_df = filtered_df.head(200)

# Iterate over the sampled rows and print msa_name and best_model_bic
results = []
for index, row in sampled_df.iterrows():
    print(f"msa_name: {row['msa_name']}, best_model_bic: {row['best_model_bic']}")
    tree_filepath = os.path.join(os.pardir, "data/raw/reference_tree", f"{row['msa_name']}.newick")
    tree_filepath_alt = os.path.join(os.pardir, "scripts/", f"{row['msa_name']}_modelfinder.raxml.bestTree")

    tree1 = Tree(tree_filepath)
    tree2 = Tree(tree_filepath_alt)

    # Calculate the Robinson-Foulds distance
    rf, max_rf, common_leaves, parts_t1, parts_t2, discarded_t1, discarded_t2 = tree1.robinson_foulds(tree2, unrooted_trees=True)

    # Normalize the RF distance
    nrf = rf / max_rf
    results.append({'msa_name': row['msa_name'], 'nrf': nrf})

    print(nrf
          )
results_df = pd.DataFrame(results)

# Save the results DataFrame as a CSV file
output_file_path = "nrf_results.csv"
results_df.to_csv(output_file_path, index=False)

print(results_df['nrf'].mean())
print(results_df['nrf'].std())
print(f"Results saved to {output_file_path}")
df_filtered = results_df[results_df['nrf'] >= 0.3]
df_filtered.to_csv('nrf_results_filtered.csv')
import matplotlib.pyplot as plt

# Plot a histogram of the NRF values
plt.figure(figsize=(10, 6))
plt.hist(results_df['nrf'], bins=20, edgecolor='black')
plt.xlabel('Normalized Robinson-Foulds (NRF) Distance')
plt.ylabel('Frequency')
plt.title('Histogram of NRF Distances')
plt.grid(True)
plt.tight_layout()
plt.savefig('nrf_results.png')