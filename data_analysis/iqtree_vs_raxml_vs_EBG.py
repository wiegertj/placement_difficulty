import os
import pandas as pd

file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/prediction"
all_dataframes = []

all_dataframes = []

for root, dirs, files in os.walk(file_path):
    for filename in files:
        if filename.endswith('.csv'):
            file_pathname = os.path.join(root, filename)
            df = pd.read_csv(file_pathname)
            all_dataframes.append(df)

combined_dataframe = pd.concat(all_dataframes, ignore_index=True)

combined_dataframe.to_csv("ebg_prediction_test.csv")