import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df.drop(columns=["branch_dist_best_two_placements", "dataset", "sampleId"], inplace=True)
print("Median Entropy: ")
print(df["entropy"].median())
print(df.columns)
# Assuming you have a DataFrame 'df' with 'entropy', 'lwr_drop', and other features

# Splitting the data into features and target variables
X = df.drop(columns=['entropy', 'lwr_drop'])
y = df[['entropy', 'lwr_drop']]

# Splitting data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base estimator (Gradient Boosting Regressor)
base_estimator = GradientBoostingRegressor()

# Create a RegressorChain model
chain = RegressorChain(base_estimator, order=[0, 1])

# Define the parameter grid for GridSearchCV
param_grid = {
    'base_estimator__n_estimators': [200, 500],
    'base_estimator__max_depth': [5, 10, 15],
    'base_estimator__learning_rate': [0.1, 0.01]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(chain, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the GridSearchCV object on the training data
grid_search.fit(X_train, y_train)

# Get the best model from the GridSearchCV
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate RMSE for entropy and lwr_drop
rmse_entropy = sqrt(mean_squared_error(y_test['entropy'], y_pred[:, 0]))
rmse_lwr_drop = sqrt(mean_squared_error(y_test['lwr_drop'], y_pred[:, 1]))

# Print RMSE for entropy and lwr_drop
print(f'RMSE for entropy: {rmse_entropy:.2f}')
print(f'RMSE for lwr_drop: {rmse_lwr_drop:.2f}')
