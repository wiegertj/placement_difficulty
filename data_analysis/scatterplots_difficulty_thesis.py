import os
import pandas as pd
import numpy as np
df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df_diff_new = pd.read_csv(os.path.join(os.pardir, "data", "treebase_difficulty_new.csv"))
df_diff_new["dataset"] = df_diff_new["name"].str.replace(".phy","")
df = df.drop("difficult", axis=1)
df = df.merge(df_diff_new, on=["dataset"], how="inner")
df.drop(axis=1, columns=['dataset', 'sampleId', "current_closest_taxon_perc_ham"], inplace=True)
print(df.shape)

step_size = 0.1
samples_per_range = 1000

# Initialize an empty DataFrame to store the sampled data
sampled_df = pd.DataFrame(columns=df.columns)

for value in np.arange(0.1, 1.1, step_size):
    # Filter the DataFrame to select rows within the current range
    filtered_range = df[(df['entropy'] >= value) & (df['entropy'] < value + step_size)]

    # Check if there are enough samples in the filtered range
    if len(filtered_range) >= samples_per_range:
        # Sample 100 values from the filtered range
        sampled_range = filtered_range.sample(n=samples_per_range, random_state=42)  # You can change the random_state
    else:
        # Take all available samples in this range
        sampled_range = filtered_range

    # Append the sampled data to the resulting DataFrame
    sampled_df = pd.concat([sampled_df, sampled_range])

def get_random_row(group):
    return group.sample(1)

# Group the DataFrame by the "dataset" column and apply the function to get random rows
sampled_df = sampled_df.groupby("dataset", group_keys=False).apply(get_random_row)

filtered_df = sampled_df
original_df = sampled_df

spearman_correlations = []

import matplotlib.pyplot as plt
import numpy as np

degree = 4


# Create scatterplots for each column in cols
#for column in ["sk_sup_tree", "mean_sup_tree", "abs_weighted_distance_major_modes_supp", "sk_clo_sim", "mean_rf_tree"]:


Q1 = original_df["sk_sup_tree"].quantile(0.25)
Q3 = original_df["sk_sup_tree"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Determine lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Create a boolean mask for filtering
mask = (original_df["sk_sup_tree"] >= lower_bound) & (
        original_df["sk_sup_tree"] <= upper_bound)

# Step 4: Apply the mask to the DataFrame
filtered_df = original_df[mask]
entropy_column = original_df['difficulty']

# Fit a polynomial of the specified degree to the data
x_data = filtered_df["difficulty"]
y_data = filtered_df["sk_sup_tree"]
# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=14)  # Set tick size to 13

# Fit a polynomial of degree 4
coefficients = np.polyfit(x_data, y_data, 4)
polynomial = np.poly1d(coefficients)

# Generate x values for the trendline
x_fit = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate corresponding y values for the trendline
y_fit = polynomial(x_fit)


# Plot the red trendline of degree 4
plt.plot(x_fit, y_fit, color='red', label='Trendline of degree 4', linewidth=3.0)

# Show the legend
plt.legend()

# Plot the trend line in red with thickness 3
plt.plot(x_fit, y_fit, 'r-', linewidth=3)
plt.ylabel('Skewness Bootstrap Support', fontsize=15)
plt.xlabel('Difficulty', fontsize=15)
# Set plot label    s and title
plt.tick_params(axis='both', labelsize=14)

plt.title(f'Scatterplot of Difficulty vs. Skewness Bootstrap Support',fontsize=15)
plt.savefig(os.path.join(os.pardir, "data/visualizations/" + "skewness_entropy.png"))
# Save or display the plot
plt.show()




# Create scatterplots for each column in cols
#for column in ["sk_sup_tree", "mean_sup_tree", "abs_weighted_distance_major_modes_supp", "sk_clo_sim", "mean_rf_tree"]:


Q1 = original_df["sk_clo_sim"].quantile(0.25)
Q3 = original_df["sk_clo_sim"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Determine lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Create a boolean mask for filtering
mask = (original_df["sk_clo_sim"] >= lower_bound) & (
        original_df["sk_clo_sim"] <= upper_bound)

# Step 4: Apply the mask to the DataFrame
filtered_df = original_df[mask]
entropy_column = original_df['difficulty']

# Fit a polynomial of the specified degree to the data
x_data = filtered_df["difficulty"]
y_data = filtered_df["sk_clo_sim"]
# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=14)  # Set tick size to 13

# Fit a polynomial of degree 4
coefficients = np.polyfit(x_data, y_data, 4)
polynomial = np.poly1d(coefficients)

# Generate x values for the trendline
x_fit = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate corresponding y values for the trendline
y_fit = polynomial(x_fit)


# Plot the red trendline of degree 4
plt.plot(x_fit, y_fit, color='red', label='Trendline of degree 4', linewidth=3.0)

# Show the legend
plt.legend()

# Plot the trend line in red with thickness 3
plt.plot(x_fit, y_fit, 'r-', linewidth=3)
plt.ylabel('Skewness Closeness Centrality', fontsize=15)
plt.xlabel('Difficulty', fontsize=15)
# Set plot label    s and title
plt.tick_params(axis='both', labelsize=14)

plt.title(f'Scatterplot of Difficulty vs. Skewness Closeness Centrality',fontsize=15)
plt.savefig(os.path.join(os.pardir, "data/visualizations/" + "sk_clo_diff.png"))
# Save or display the plot
plt.show()







# Create scatterplots for each column in cols
#for column in ["sk_sup_tree", "mean_sup_tree", "abs_weighted_distance_major_modes_supp", "sk_clo_sim", "mean_rf_tree"]:


Q1 = original_df["mean_sup_tree"].quantile(0.25)
Q3 = original_df["mean_sup_tree"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Determine lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Create a boolean mask for filtering
mask = (original_df["mean_sup_tree"] >= lower_bound) & (
        original_df["mean_sup_tree"] <= upper_bound)

# Step 4: Apply the mask to the DataFrame
filtered_df = original_df[mask]
entropy_column = original_df['difficulty']

# Fit a polynomial of the specified degree to the data
x_data = filtered_df["difficulty"]
y_data = filtered_df["mean_sup_tree"]
# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=14)  # Set tick size to 13

# Fit a polynomial of degree 4
coefficients = np.polyfit(x_data, y_data, 4)
polynomial = np.poly1d(coefficients)

# Generate x values for the trendline
x_fit = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate corresponding y values for the trendline
y_fit = polynomial(x_fit)


# Plot the red trendline of degree 4
plt.plot(x_fit, y_fit, color='red', label='Trendline of degree 4', linewidth=3.0)

# Show the legend
plt.legend()

# Plot the trend line in red with thickness 3
plt.plot(x_fit, y_fit, 'r-', linewidth=3)
plt.ylabel('Mean Bootstrap Support', fontsize=15)
plt.xlabel('Difficulty', fontsize=15)
# Set plot label    s and title
plt.tick_params(axis='both', labelsize=14)

plt.title(f'Scatterplot of Difficulty vs. Mean Bootstrap Support',fontsize=15)
plt.savefig(os.path.join(os.pardir, "data/visualizations/" + "mbs.png"))
# Save or display the plot
plt.show()



# Create scatterplots for each column in cols
#for column in ["sk_sup_tree", "mean_sup_tree", "abs_weighted_distance_major_modes_supp", "sk_clo_sim", "mean_rf_tree"]:


Q1 = original_df["mean_rf_tree"].quantile(0.25)
Q3 = original_df["mean_rf_tree"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Determine lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Create a boolean mask for filtering
mask = (original_df["mean_rf_tree"] >= lower_bound) & (
        original_df["mean_rf_tree"] <= upper_bound)

# Step 4: Apply the mask to the DataFrame
filtered_df = original_df[mask]
entropy_column = original_df['difficulty']

# Fit a polynomial of the specified degree to the data
x_data = filtered_df["difficulty"]
y_data = filtered_df["mean_rf_tree"]
# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=14)  # Set tick size to 13

# Fit a polynomial of degree 4
coefficients = np.polyfit(x_data, y_data, 4)
polynomial = np.poly1d(coefficients)

# Generate x values for the trendline
x_fit = np.linspace(x_data.min(), x_data.max(), 100)

# Calculate corresponding y values for the trendline
y_fit = polynomial(x_fit)


# Plot the red trendline of degree 4
plt.plot(x_fit, y_fit, color='red', label='Trendline of degree 4', linewidth=3.0)

# Show the legend
plt.legend()

# Plot the trend line in red with thickness 3
plt.plot(x_fit, y_fit, 'r-', linewidth=3)
plt.ylabel('Mean RF-Distance Bootstraps', fontsize=15)
plt.xlabel('Difficulty', fontsize=15)
# Set plot label    s and title
plt.tick_params(axis='both', labelsize=14)

plt.title(f'Scatterplot of Difficulty vs. Mean RF-Distance Bootstraps',fontsize=15)
plt.savefig(os.path.join(os.pardir, "data/visualizations/" + "mrf.png"))
# Save or display the plot
plt.show()