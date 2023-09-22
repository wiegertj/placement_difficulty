import pandas as pd
import os
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df = difficulties_df[["verbose_name", "difficult"]]

difficulties_df = difficulties_df.drop_duplicates(subset=['verbose_name'], keep='first')
difficulties_df["dataset"] = difficulties_df["verbose_name"].str.replace(".phy", "")
difficulties_df.drop(columns=["verbose_name"], axis=1, inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns


df_tree = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_uncertainty = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_uncertainty["dataset"] = df_uncertainty["dataset"].str.replace(".newick","")
df_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_merged = df_msa.merge(df_tree, on=["dataset"], how="inner")
df_merged = df_merged.merge(df_uncertainty, on=["dataset"], how="inner")
#subst_stats = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "subst_freq_stats.csv"))
#df_merged = df_merged.merge(subst_stats, on=["dataset"], how="inner")
#df_merged = df_merged.merge(difficulties_df, on=["dataset"], how="inner")
print(df_merged.shape)

for column2 in df_merged.drop(columns=["dataset","mean_support", "max_support","std_support", "skewness_support", "kurt_support", 'min_rf', 'max_rf', 'mean_rf', 'std_dev_rf', 'skewness_rf',
       'kurtosis_rf']).columns:
    column = "skewness_support"
    correlation, p_value = spearmanr(df_merged[column2], df_merged["skewness_support"])
    if abs(correlation) >= 0.5:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=column2, y="skewness_support", data=df_merged)
        print(column2 + "_" + "skewness_support" + "_" + str(correlation))
        # Calculate Spearman rank correlation and p-value
        correlation = round(correlation, 4)
        p_value = round(p_value, 8)

         #Add Spearman rank correlation and p-value to the title
        plt.title(f'{column2} and {column}\nSpearman Correlation: {correlation:.4f}, p-value:{p_value:.4f}')



    # Add a trendline with confidence
    #sns.regplot(x=column2, y=column, data=df, scatter=False, ci=95, color='red', order=4)

        plt.xlabel(column2)
        plt.ylabel(column)
        plt.savefig(f'1111111{column2}_vs_{column}.png')  # Save the figure
        print("Saved")




# Extract the "mean_support" column values
mean_support_values = df_merged["mean_support"]

# Create a histogram
plt.hist(mean_support_values, bins=20, edgecolor='k')  # Adjust the number of bins as needed
plt.xlabel("mean_support")
plt.ylabel("Frequency")
plt.title("Histogram of skewness_support")
plt.grid(True)
plt.savefig("mean_support.png")

print(df_merged["mean_support"].mean())
#df_merged.drop(columns=["dataset"], inplace=True, axis=1)


df_merged.fillna(0, inplace=True)


X = df_merged.drop(columns=["dataset","mean_support", "max_support","std_support", "skewness_support", "kurt_support", 'min_rf', 'max_rf', 'mean_rf', 'std_dev_rf', 'skewness_rf',
       'kurtosis_rf'])
print(X.columns)
y = df_merged["mean_support"]

X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.2)

# Initialize the Random Forest regressor

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to search
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50),
    }

    # Initialize CV RMSE
    cv_rmse = 0.0

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    for train_idx, val_idx in kf.split(X_train_full):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # Create dataset objects
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train the model
        model = lgb.train(params, train_data, valid_sets=[train_data, val_data], verbose_eval=100)

        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate RMSE for this fold
        fold_rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Update CV RMSE
        cv_rmse += fold_rmse / 5  # Divide by the number of folds (5 in this case)

    return cv_rmse

# Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)


# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train the final model using the best hyperparameters
best_model = lgb.LGBMRegressor(**best_params)

best_model.fit(X_train_full, y_train_full)

# Evaluate on holdout set
y_pred_holdout = best_model.predict(X_holdout)
rmse_holdout = mean_squared_error(y_holdout, y_pred_holdout, squared=False)
print(f"RMSE on holdout set: {rmse_holdout}")

# Print feature importances
feature_importances = best_model.feature_importances_
feature_names = X.columns

# Create a dictionary to store feature names and their importances
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort feature importances in descending order
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print feature importances
print("Feature Importances:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance:.4f}")