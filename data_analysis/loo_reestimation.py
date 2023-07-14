import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "norm_rf_loo.csv"))

# get haag-score
df['dataset'] = df['dataset_sampleId'].str.split('_').str[:2].str.join('_') + ".phy"
df_difficulties = pd.read_csv(os.path.join(os.pardir, "data", "treebase_difficulty.csv"))
df = df.merge(df_difficulties, how='inner', right_on="verbose_name", left_on="dataset")
print(df.columns)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Normalized RF Distance and Branch Length Distance')

# Scatter plots
axs[0].scatter(df['difficult'], df['norm_rf_dist'], label='Normalized RF Distance', color='blue')
axs[1].scatter(df['difficult'], df['bsd'], label='Branch Length Distance', color='orange')

degree = 5

for i, col in enumerate([('norm_rf_dist', 'Normalized RF Distance'), ('bsd', 'Felsenstein Branch Length Distance')]):
    # Polynomial regression
    coefficients = np.polyfit(df['difficult'], df[col[0]], degree)
    x_range = np.linspace(min(df['difficult']), max(df['difficult']), num=100)
    y_range = np.polyval(coefficients, x_range)

    # Plot the polynomial curve
    axs[i].plot(x_range, y_range, color='red')

    # Set axis labels
    axs[i].set_xlabel('Difficulty')
    axs[i].set_ylabel(col[0])
    axs[i].set_ylabel(col[1])  # Add custom y-axis label here

# Remove legends
axs[0].legend([])
axs[1].legend([])

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
