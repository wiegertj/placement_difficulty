import json
import os
import pandas as pd
from scipy.stats import entropy

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

def extract_entropy(jplace_file) -> pd.DataFrame:
    print(jplace_file)
    entropies = []
    with open(jplace_file, 'r') as f:
        jplace_data = json.load(f)
        for placement in jplace_data['placements']:
            sample_name = placement['n'][0]
            probabilities = placement['p']
            like_weight_ratios = [tup[2] for tup in probabilities]

            entropy_val = entropy(like_weight_ratios)
            entropies.append((sample_name, entropy_val))

    df = pd.DataFrame(entropies, columns=['SampleID', 'Entropy'])
    df['Entropy'] = min_max_normalize(df['Entropy'])
    return df

if __name__ == '__main__':

    for file in ["neotrop_10k_epa_result.jplace", "bv_epa_result.jplace", "tara_epa_result.jplace"]:
        jplace_file_path = os.path.join(os.pardir, "data/raw/placements", file)
        df = extract_entropy(jplace_file_path)
        df.to_csv(os.path.join(os.pardir, "data/processed/target", file.replace(".jplace", "") + "_entropy.csv"))








