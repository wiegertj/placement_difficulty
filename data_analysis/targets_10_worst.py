import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.join(os.pardir, "data/processed/target", "loo_result_entropy.csv")
df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histogram for "entropy" column
axes[0].hist(df['entropy'], bins=40, color='skyblue')
axes[0].set_title("Histogram of Entropy")

# Histogram for "lwr_drop" column
axes[1].hist(df['lwr_drop'], bins=40, color='salmon')
axes[1].set_title("Histogram of Lwr Drop")

# Histogram for "branch_dist_best_two_placements" column
axes[2].hist(df['branch_dist_best_two_placements'], bins=40, color='green')
axes[2].set_title("Histogram of Branch Distance Best Two Placements")

plt.tight_layout()
plt.show()
