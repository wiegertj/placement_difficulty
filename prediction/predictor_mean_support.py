import pandas as pd
import os
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty.csv")
difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
difficulties_df = difficulties_df.drop_duplicates(subset=['verbose_name'], keep='first')
difficulties_df["verbose_name"] = difficulties_df["verbose_name"].str.replace(".phy", "")

df_tree = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree.csv"))
df_uncertainty = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "tree_uncertainty.csv"))
df_uncertainty["dataset"] = df_uncertainty["dataset"].str.replace(".newick","")
df_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"))
df_merged = df_msa.merge(df_tree, on=["dataset"], how="inner")
df_merged = df_merged.merge(df_uncertainty, on=["dataset"], how="inner")
subst_stats = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "subst_freq_stats.csv"))
#df_merged = df_merged.merge(subst_stats, on=["dataset"], how="inner")
print(df_merged.shape)
print(df_merged["mean_support"].mean())

X = df.drop(columns=["mean_sup_tree"])
y = df["mean_sup_tree"]

# Define the objective function for Optuna
def objective(trial):
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Create dataset objects
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train the model
    model = lgb.train(params, train_data, valid_sets=[train_data, val_data], verbose_eval=100)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Calculate RMSE
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse

# Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train the final model using the best hyperparameters
best_model = lgb.LGBMRegressor(**best_params)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on holdout set
y_pred_holdout = best_model.predict(X_val)
rmse_holdout = mean_squared_error(y_val, y_pred_holdout, squared=False)
print(f"RMSE on holdout set: {rmse_holdout}")