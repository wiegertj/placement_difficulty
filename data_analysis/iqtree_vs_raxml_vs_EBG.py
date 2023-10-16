import os
import pandas as pd

file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/prediction"
all_dataframes = []

all_dataframes = []
counter = 0
for root, dirs, files in os.walk(file_path):
    # Skip the "ebg_tmp" directory and its contents
    if "ebg_tmp" in dirs:
        dirs.remove("ebg_tmp")
    for filename in files:
        if filename.endswith('.csv'):
            counter += 1
            print(counter)
            file_pathname = os.path.join(root, filename)
            df = pd.read_csv(file_pathname)
            all_dataframes.append(df)

combined_dataframe = pd.concat(all_dataframes, ignore_index=True)

combined_dataframe.to_csv("ebg_prediction_test.csv")