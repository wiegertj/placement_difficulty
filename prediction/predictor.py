import math
import os
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
X = df.drop(axis=1, columns=["entropy"]) # Select all columns except the last column (features)
#X = X[['min_perc_hash_ham_dist', 'max_perc_hash_ham_dist',
 #      'avg_perc_hash_ham_dist', 'std_perc_hash_ham_dist', 'dataset', 'sampleId']]
print(X.columns)

y = df["entropy"]   # Select the last column (target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mse_ = mean_squared_error(y_test, np.zeros(len(y_test))) # RMSE on just predicting most common => 0
rmse_ = math.sqrt(mse_)
print("Baseline 0 RMSE: " + str(rmse_))

mse_ = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train)) # RMSE on just predicting most common => 0
rmse_ = math.sqrt(mse_)
print("Baseline Mean RMSE: " + str(rmse_))

model = RandomForestRegressor( n_estimators = 300,
    max_depth = 20,
    max_features= 10,
    min_samples_split= 10,
    min_samples_leaf=5, n_jobs=-1)

rfe = RFE(estimator=model, n_features_to_select=10)  # Adjust the number of features as needed
rfe.fit(X_train.drop(axis=1, columns=['dataset', 'sampleId']), y_train)
print(rfe.support_)
selected_features = X_train.drop(axis=1, columns=['dataset', 'sampleId']).columns[rfe.support_]
print(selected_features)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

param_grid = {
    'n_estimators': [250, 350, 500],
    'max_depth': [10, 20],
    'max_features': [5, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [10, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

best_params = grid_search.best_params_
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# MSE of entropy.py prediction on testset
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error on test set: {rmse}")
feature_importances = best_model.feature_importances_

# Scale and print feature importances
scaler = MinMaxScaler()
normalized_importances = scaler.fit_transform(feature_importances.reshape(-1, 1)).flatten()
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': normalized_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances (Normalized):")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")
