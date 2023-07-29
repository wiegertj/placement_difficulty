import os
from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df.drop(columns=['sampleId', "lwr_drop", "min_blength", "branch_dist_best_two_placements"], inplace=True)

data_frame = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
dataset_sample = data_frame['verbose_name'].str.replace(".phy", "").sample(40)
holdout_datasets = df[df["dataset"].isin(dataset_sample)]
df = df[~df["dataset"].isin(dataset_sample)]
X_test = holdout_datasets.drop(axis=1, columns=["entropy", "dataset"])
y_test = holdout_datasets["entropy"]
print("Number of test samples: " + str(len(y_test)))
print(dataset_sample)
X_train = df.drop(axis=1, columns=["entropy", "dataset"])
y_train = df["entropy"]

tpot = TPOTRegressor(generations=5, population_size=20, cv=5,
                     random_state=42, verbosity=2)
tpot.fit(X_train, y_train)

y_pred = tpot.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")
# Export the optimized pipeline code
tpot.export('tpot_optimized_pipeline.py')
