import pandas as pd

# Define the paths
path_1 = "/hits/fast/cme/wiegerjs/EBG_train/EBG_train/data/processed"
path_new = "/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/features/bs_features/parsimony_boot_199_new.csv"

# Load and print columns for the first file
try:
    df_1 = pd.read_csv(path_1)
    print(f"Columns in {path_1}:")
    print(df_1.columns)
except Exception as e:
    print(f"Error reading file {path_1}: {e}")

print("-" * 40)

# Load and print columns for the second file
try:
    df_new = pd.read_csv(path_new)
    print(f"Columns in {path_new}:")
    print(df_new.columns)
except Exception as e:
    print(f"Error reading file {path_new}: {e}")
