import os
import pandas as pd

list_foldernames = pd.read_csv(os.pardir + "/data/folder_names_raxml.csv")["folder_name"].values.tolist()
counter=0
results = []
df_tmps = []

base_path = "/hits/fast/cme/wiegerjs/placement_difficulty/tests/"
# Loop over each subdirectory (folder) within the specified path

for folder_name in list_foldernames:

    folder_path = base_path + str(folder_name) + "/" + str(folder_name) + ".csv"
    if os.path.exists(folder_path):
        counter += 1
        print(counter)
        df_tmp = pd.read_csv(folder_path)
        df_tmps.append(df_tmp)

result_df = pd.concat(df_tmps, ignore_index=True)
print(result_df.shape)

truth = pd.read_csv(os.path.join(os.pardir, "data/processed/target/branch_supports_raxml_data.csv"))
truth["support"] = truth["support"] * 100

df_merged = result_df.merge(truth, on=["dataset", "branchId"], how="inner")
print(df_merged.shape)

df_merged["prediction_o70"] = 0
df_merged["prediction_o80"] = 0
df_merged.loc[df_merged["prediction_over_70"] >= 0.5, "prediction_o70"] = 1
df_merged.loc[df_merged["prediction_over_80"] >= 0.5, "prediction_o80"] = 1

df_merged["o70_true"] = 0
df_merged["o80_true"] = 0
df_merged.loc[df_merged["support"] >= 70, "o70_true"] = 1
df_merged.loc[df_merged["support"] >= 80, "o80_true"] = 1

df_merged["prediction_error"] = df_merged["support"] - df_merged["prediction_median"]
df_merged.to_csv("raxml_data_results.csv")