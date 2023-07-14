import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df = df.drop(axis=1, columns=['dataset', 'sampleId'])
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(18, 15))
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, fmt=".1f", ax=ax)
heatmap.set_xticklabels(df.columns, rotation=90)
heatmap.set_yticklabels(df.columns, rotation=0)
heatmap.set_title('Pearson Correlation Matrix')

heatmap.figure.savefig(os.path.join(os.pardir, "data/visualizations", "heatmap.png"))