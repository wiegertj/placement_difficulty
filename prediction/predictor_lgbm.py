import math
import shap
import lightgbm as lgb
from verstack import LGBMTuner

import os
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def light_gbm_regressor(rfe=False, rfe_feature_n=10, shapley_calc=True, targets=[]):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "final_dataset.csv"))
    df.drop(columns=["lwr_drop", "branch_dist_best_two_placements"], inplace=True)
    print("Median Entropy: ")
    print(df["entropy"].median())
    print(df.columns)
    print(df.shape)

    if targets == []:
        target = "entropy"
    else:
        target = targets

    X = df.drop(axis=1, columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)
    print("Baseline predicting mean RMSE: " + str(rmse_mean))

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

    #param_grid = {
     #   'boosting_type': ['gbdt'],
      #  'num_leaves': [75, 100],
       # 'max_depth': [5, 10],
       # 'learning_rate': [0.05],
       # 'n_estimators': [850, 1000],
       # 'min_child_samples': [20, 50]
    #}

    tuner = LGBMTuner(metric='rmse')  # <- the only required argument
    tuner.fit(X_train, y_train)
    # check the optimization log in the console.
    #pred = tuner.predict(test)

    #model = lgb.LGBMRegressor(n_jobs=40)

    #grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    #grid_search.fit(X_train, y_train)

    #best_params = grid_search.best_params_
    #best_model = grid_search.best_estimator_
    #print(best_params)

    y_pred = tuner.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    model = tuner.fitted_model

    feature_importance = model.feature_importances_

    for feature, importance in zip(X_train.columns, feature_importance):
        print(f'{feature}: {importance}')

    scaler = MinMaxScaler()
    normalized_importances = scaler.fit_transform(feature_importance.reshape(-1, 1)).flatten()
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': normalized_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()

    name = "rf_tree_holdout_20"
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
        # X_test = X_test_[(abs(X_test_['entropy'] - X_test_['prediction']) < 0.05) & (
        #       (X_test_['entropy'] < 0.1) | (X_test_['entropy'] > 0.9))]
        X_test = X_test_
        X_test = X_test.sort_values(by="entropy")
        explainer = shap.Explainer(model, X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId"]),
                                   check_additivity=False)
        shap_values = explainer(X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId"]),
                                check_additivity=False)

        shap.summary_plot(shap_values, X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId"]),
                          plot_type="bar")
        plt.savefig(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + "shap.png"))

        # Create the waterfall plot for the sample with the highest prediction
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[0], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_0_treeholdout.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[1500], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_1500_treeholdout.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[-300], max_display=10)  # Limit the display to 10 features

        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("waterfall_plot_2500treeholdout.png")


light_gbm_regressor(rfe=False, shapley_calc=False, targets=[])

