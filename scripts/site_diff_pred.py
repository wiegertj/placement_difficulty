import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
# Read the CSV file
file_path = '/Users/juliuswiegert/Downloads/site_diff_features.csv'
df = pd.read_csv(file_path,  usecols=lambda column: column != 'Unnamed: 0')

# Drop the 'msa_name' and 'dataset' columns
columns_to_drop = ['msa_name', 'dataset', "mean_entropy", "min_entropy", "skewness_entropy",	"kurt_entropy",	"std_dev_entropy"]
df = df.drop(columns=columns_to_drop, errors='ignore')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(-1, inplace=True)


# Check if 'diff_diff' column is present in the DataFrame

# Separate features (X) and target variable (y)
X = df.drop(columns=['diff_diff'])
y = df['diff_diff']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a DecisionTreeRegressor
regression_tree = DecisionTreeRegressor()

# Fit the model on the training data
regression_tree.fit(X_train, y_train)

# Make predictions on the test data
predictions = regression_tree.predict(X_test)
X_test["pred"] = predictions
X_test["true"] = y_test
# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

from sklearn.metrics import mean_absolute_error, median_absolute_error

# ... (previous code for prediction)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate Median Absolute Error
median_ae = median_absolute_error(y_test, predictions)
print("Median Absolute Error:", median_ae)

feature_importances = regression_tree.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
sorted_importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or use the sorted feature importances
print(sorted_importance_df.head(20))

print("---"*20)

file_path = '/Users/juliuswiegert/Downloads/site_diff_features.csv'
df = pd.read_csv(file_path,  usecols=lambda column: column != 'Unnamed: 0')

# Drop the 'msa_name' and 'dataset' columns
columns_to_drop = ['msa_name', "mean_entropy", "min_entropy", "skewness_entropy",	"kurt_entropy",	"std_dev_entropy"]
df = df.drop(columns=columns_to_drop, errors='ignore')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(-1, inplace=True)

# Convert the categories to integers (0, 1)

df["class"] = 0.0
df.loc[df['diff_diff'] > 0.0, 'class'] = 1
df.loc[df['diff_diff'] < 0.0, 'class'] = -1

# Split the data into training and testing sets
#X_classifier = df.drop(columns=['diff_diff', 'class'])
#y_classifier = df['class']

df["group"] = df['dataset'].astype('category').cat.codes.tolist()

sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
test = df[df['group'].isin(sample_dfs)]
train = df[~df['group'].isin(sample_dfs)]

print(test.shape)
print(train.shape)

X_train = train.drop(axis=1, columns=["diff_diff", "dataset", "group", "class"])
y_train = train["class"]

X_test = test.drop(axis=1, columns=["diff_diff", "dataset", "group", "class"])
y_test = test["class"]

# Train a DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their potential values
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Assuming X_train, y_train, X_test, and y_test are defined

param_grid = {
    'criterion': ['gini'],
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [10, 30, 50, 100],
    'min_samples_split': [5, 10, 20, 40],
    'min_samples_leaf': [4, 8, 12]
}

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(classifier, param_grid, cv=10, scoring='accuracy')

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions on the test data using the best model
best_classifier = grid_search.best_estimator_
predictions_classifier = best_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions_classifier)
print("Classifier Accuracy:", accuracy)

# Get feature importances from the best model
feature_importances = best_classifier.feature_importances_

# Create a DataFrame with feature names and their importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
sorted_importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or use the sorted feature importances
print(sorted_importance_df.head(20))

# Evaluate the classifier (you may want to remove this if you've already evaluated it above)
accuracy = accuracy_score(y_test, predictions_classifier)
balanced_accuracy_score_ = balanced_accuracy_score(y_test, predictions_classifier)
print("Classifier Accuracy:", accuracy)
print("Classifier BAC:", balanced_accuracy_score_)

# Add predictions as a column to the test data and save to CSV
X_test["pred_class"] = predictions_classifier
X_test.to_csv("testdata.csv", index=False)
