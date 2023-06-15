import math
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
df.drop(axis=1, columns=['dataset', 'sampleId'], inplace=True)

# Split the dataset into features (X) and target (y)
#X = df[["gap_fraction"]]

# Tara as test
df_test = df[df["tree_depth"] == 3.8795620000000017]
df_train = df[df["tree_depth"] != 3.8795620000000017]

X_train = df_train.drop(axis=1, columns=["Entropy"])
y_train = df_train["Entropy"]

X_test = df_test.drop(axis=1, columns=["Entropy"])
y_test = df_test["Entropy"]

#X = df.drop(axis=1, columns=["Entropy"]) # Select all columns except the last column (features)
#y = df["Entropy"]   # Select the last column (target)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
#model = GradientBoostingRegressor()

param_grid_GBR = {
    'n_estimators': [300],
    'learning_rate': [0.05],
    'max_depth': [10]
}

param_grid = {
    'n_estimators': [300],
    'max_depth': [20],
    'max_features': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [5]
}

# GridSearch
grid_search = GridSearchCV(model, param_grid, cv=8)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

best_params = grid_search.best_params_
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# MSE of entropy prediction on testset
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
