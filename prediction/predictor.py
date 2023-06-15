import math
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.express as px

# Read CSV file into a pandas DataFrame
df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df.drop(axis=1, columns=['dataset', 'sampleId'], inplace=True)

# Split the dataset into features (X) and target (y)
X = df.drop(axis=1, columns=["Entropy", "std_ham_dist", "min_ham_dist", "max_ham_dist", "avg_ham_dist"]) # Select all columns except the last column (features)
y = df["Entropy"]   # Select the last column (target)


print(df[df["Entropy"] == 0].shape)
print(df.shape)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForestRegressor model
model = RandomForestRegressor()
#model = GradientBoostingRegressor()

param_grid_GBR = {
    'n_estimators': [300],
    'learning_rate': [0.05],
    'max_depth': [10]
}


# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [400],
    'max_depth': [10],
    'min_samples_split': [5]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the best model on the holdout test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(y_test)
print(y_pred)

print(f"Root Mean Squared Error on test set: {rmse}")
feature_importances = best_model.feature_importances_


scaler = MinMaxScaler()
normalized_importances = scaler.fit_transform(feature_importances.reshape(-1, 1)).flatten()

# Create a DataFrame with feature names and importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': normalized_importances})

# Sort the feature importances in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted and normalized feature importances
print("Feature Importances (Normalized):")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

best_params = grid_search.best_params_

print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")