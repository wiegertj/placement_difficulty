import pandas as pd
import os
df_nomt = pd.read_csv(os.pardir + "/data/bootstrap_times_iqtree_nomt.csv")
df_mt = pd.read_csv(os.pardir + "/data/bootstrap_times_iqtree_mt.csv")

df = df_nomt.merge(df_mt, on=["dataset"], how="inner")
print(df.shape)
df['elapsed_time'] = df[['elapsed_time_mt', 'elapsed_time_nomt']].min(axis=1)
df.to_csv("iqtree_final.csv")
print(df)