import pandas as pd
import os

tree_features = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), index_col=False,
                            usecols=lambda column: column != 'Unnamed: 0')

row = tree_features[tree_features["dataset"] == "test"].head(1)

avg_blength = row["avg_blength"].item()
assert round(avg_blength, 4) == 0.1922

max_blength = row["max_blength"].item()
assert max_blength == 0.4

min_blength = row["min_blength"].item()
assert min_blength == 0.05

std_blength = row["std_blength"].item()
assert round(std_blength, 4) == 0.0983
#(((A:0.1,B:0.2):0.3,(C:0.15,D:0.12):0.25):0.4,((E:0.18,F:0.22):0.3,(G:0.1,(H:0.05,(I:0.08,J:0.11):0.12):0.15):0.28):0.35);
#93,test,0.1922222222222222,0.4,0.05,0.09829409135376209,1.01,0.13100000000000003,0.05,0.22,0.05486346689738081,0.23888888888888893,0.0,0.4,0.12584161120675114

#'max_blength', 'min_blength', 'std_blength', 'tree_depth',
 #                              'average_branch_length_tips', 'min_branch_length_tips', 'max_branch_length_tips', 'std_branch_length_tips', 'average_branch_length_inner',
  #                             'min_branch_length_inner', 'max_branch_length_inner', 'std_branch_length_inner'


