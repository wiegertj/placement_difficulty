import json
import math
import os
import pickle
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def rand_forest_entropy(holdout_trees=0, rfe=False, rfe_feature_n=10):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))

    if holdout_trees == 0:
        X = df.drop(axis=1, columns=["entropy", "lwr_drop", "branch_dist_best_two_placements"])
        y = df["entropy"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
        rmse_zero = math.sqrt(mse_zero)
        print("Baseline prediting 0 RMSE: " + str(rmse_zero))

        mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
        rmse_mean = math.sqrt(mse_mean)
        print("Baseline predicting mean RMSE: " + str(rmse_mean))
    else:
        data_frame = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
        dataset_sample = data_frame['verbose_name'].str.replace(".phy", "").sample(40)
        holdout_datasets = df[df["dataset"].isin(dataset_sample)]
        df = df[~df["dataset"].isin(dataset_sample)]
        X_test = holdout_datasets.drop(axis=1, columns=["entropy", "lwr_drop", "branch_dist_best_two_placements"])
        y_test = holdout_datasets["entropy"]
        print("Number of test samples: " + str(len(y_test)))
        print(dataset_sample)
        X_train = df.drop(axis=1, columns=["entropy", "lwr_drop", "branch_dist_best_two_placements"])
        y_train = df["entropy"]

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=10, min_samples_split=20, min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'sampleId']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'sampleId']).columns[rfe.support_]
        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    model = RandomForestRegressor(n_jobs=8)

    param_grid = {
        'n_estimators': [100, 250, 350],
        'max_depth': [5, 10, 20],
        'max_features': [5, 10],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [10, 20]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset', 'sampleId'])
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # MSE of entropy.py prediction on testset
    y_pred = best_model.predict(X_test.drop(axis=1, columns=['dataset', 'sampleId']))
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")
    feature_importances = best_model.feature_importances_

    # Scale and print feature importances
    scaler = MinMaxScaler()
    normalized_importances = scaler.fit_transform(feature_importances.reshape(-1, 1)).flatten()
    importance_df = pd.DataFrame({'Feature': X_train.drop(axis=1, columns=['dataset', 'sampleId']).columns, 'Importance': normalized_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()

    if holdout_trees == 0:
        name = "rf_sample_holdout_02"
    else:
        name = "rf_tree_holdout_" + str(holdout_trees)

    if rfe:
        name = name + "_rfe_" + str(rfe_feature_n)

    plot_filename = os.path.join(os.pardir, "data/prediction", "feature_importances_" + name + ".png")
    plt.savefig(plot_filename)

    print("Feature Importances (Normalized):")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    with open(os.path.join(os.pardir, "data/prediction", "standard_" + name + ".pkl"), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(os.pardir, "data/prediction", "standard_rf_params_" + name + ".txt"), 'w') as f:
        best_params_str = json.dumps(best_params)
        f.write(best_params_str)

rand_forest_entropy(rfe=True, holdout_trees=0)
rand_forest_entropy(holdout_trees=40, rfe=True)
