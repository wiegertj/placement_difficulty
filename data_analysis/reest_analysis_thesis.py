import os
import pandas as pd
import matplotlib.pyplot as plt

df_noreest = pd.read_csv(os.path.join(os.pardir, "data/reest", "loo_result_noreest.csv"))
df_reest = pd.read_csv(os.path.join(os.pardir, "data/reest", "loo_result_reest.csv"))
df_reest["entropy_reest"] = df_reest["entropy"]
df_merged = df_noreest.merge(df_reest, on=["dataset", "sampleId"], how="inner")
df_merged["entropy_diff"] = df_merged["entropy_reest"] - df_merged["entropy_noreest"]

plt.hist(df_merged["entropy_diff"], bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.title('Histogram of Difference Entropy (Reestimation) and Entropy (No Reestimation) ', fontsize=13)
plt.xlabel('Entropy (Reestimation) - Entropy (No Reestimation)', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.show()