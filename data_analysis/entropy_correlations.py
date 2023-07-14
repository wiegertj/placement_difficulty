import pandas as pd
import os
df = pd.read_csv(os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv"))
df_difficulties = pd.read_csv(os.path.join(os.pardir, "data", "treebase_difficulty.csv"))
df_difficulties["verbose_name"] = df_difficulties["verbose_name"].str.replace(".phy", "")
df = df.merge(df_difficulties, how='inner', right_on="verbose_name", left_on="dataset")
df = df[["dataset", "difficult", "entropy"]]
mean_df = df.groupby('dataset').mean().reset_index()

import numpy as np
import matplotlib.pyplot as plt

# Assuming your DataFrame with mean values is named 'mean_df'

# Extract the 'difficulty' and 'entropy' columns as NumPy arrays
difficulty = mean_df['difficult'].values
entropy = mean_df['entropy'].values

# Perform polynomial regression of degree 5
coefficients = np.polyfit(difficulty, entropy, 5)

# Generate values for x-axis based on difficulty range
x = np.linspace(min(difficulty), max(difficulty), 100)

# Evaluate the polynomial curve using the coefficients and x values
y = np.polyval(coefficients, x)

# Plot the scatter plot
plt.scatter(difficulty, entropy)

# Plot the polynomial curve
plt.plot(x, y, color='red')

# Set axis labels
plt.xlabel('Difficulty')
plt.ylabel('Entropy')

# Set plot title
plt.title('Polynomial Regression: Entropy vs. Difficulty')

# Add legend
plt.legend()

# Display the plot
plt.show()

