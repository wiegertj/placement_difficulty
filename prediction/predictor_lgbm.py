import json
import math
import shap
import lightgbm as lgb

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


def rand_forest_entropy(holdout_trees=0, rfe=False, rfe_feature_n=10, shapley_calc=True, targets=[]):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
    df.drop(columns=["lwr_drop", "branch_dist_best_two_placements"], inplace=True)
    print("Median Entropy: ")
    print(df["entropy"].median())


    if targets == []:
        target = "entropy"
    else:
        target = targets

    if holdout_trees == 0:
        X = df.drop(axis=1, columns=target)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
        rmse_zero = math.sqrt(mse_zero)
        print("Baseline prediting 0 RMSE: " + str(rmse_zero))

        mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
        rmse_mean = math.sqrt(mse_mean)
        print("Baseline predicting mean RMSE: " + str(rmse_mean))
    else:
        data_frame = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
        dataset_sample = data_frame['verbose_name'].str.replace(".phy", "").sample(30)
        holdout_datasets = df[df["dataset"].isin(dataset_sample)]
        df = df[~df["dataset"].isin(dataset_sample)]
        X_test = holdout_datasets.drop(axis=1, columns=target)
        y_test = holdout_datasets[target]
        print("Number of test samples: " + str(len(y_test)))
        print(dataset_sample)
        X_train = df.drop(axis=1, columns=target)
        y_train = df[target]

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=10, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'sampleId']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'sampleId']).columns[rfe.support_]
        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset', 'sampleId'])
        X_test = X_test.drop(axis=1, columns=['dataset', 'sampleId'])

    # Define parameter grid for grid search
    param_grid = {
        'boosting_type': ['gbdt', 'dart'],  # You can add more options
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.8, 0.9],
        'n_estimators': [100, 200, 300]
    }

    # Create LightGBM model
    model = lgb.LGBMRegressor(n_jobs=20)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               n_jobs=-1)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best parameters and best estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate best model on test data

    # MSE of entropy.py prediction on testset
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    feature_importance = model.feature_importance()

    # Print feature importances
    for feature, importance in zip(X_train.columns, feature_importance):
        print(f'{feature}: {importance}')

    # Scale and print feature importances
    scaler = MinMaxScaler()
    normalized_importances = scaler.fit_transform(feature_importance.reshape(-1, 1)).flatten()
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': normalized_importances})
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



    X_test_["prediction"] = y_pred
    X_test_["entropy"] = y_test
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + ".csv"))

    if shapley_calc:
        X_test = X_test_[(abs(X_test_['entropy'] - X_test_['prediction']) < 0.05) & (
                (X_test_['entropy'] < 0.1) | (X_test_['entropy'] > 0.9))]

        explainer = shap.Explainer(model, X_test.drop(columns=["entropy", "prediction"]), check_additivity=False)
        shap_values = explainer(X_test.drop(columns=["entropy", "prediction"]), check_additivity=False)

        shap.summary_plot(shap_values, X_test.drop(columns=["entropy", "prediction"]), plot_type="bar")
        plt.savefig(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + "shap.png"))

        # Create the waterfall plot for the sample with the highest prediction
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[9], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_9_treeholdout.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[15], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_15_treeholdout.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[100], max_display=10)  # Limit the display to 10 features

        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_100treeholdout.png")


rand_forest_entropy(rfe=False, holdout_trees=0, shapley_calc =False, targets=[])
# rand_forest_entropy(holdout_trees=40, rfe=False)
# rand_forest_entropy(rfe=False, holdout_trees=30, shapley_calc =False, targets=[])
# rand_forest_entropy(holdout_trees=40, rfe=True)
