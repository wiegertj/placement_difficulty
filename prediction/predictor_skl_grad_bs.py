import math
import sys
import random
import shap
import lightgbm as lgb
import os
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback
from sklearn.ensemble import GradientBoostingRegressor


def quantile_loss(y_true, y_pred, quantile):
    """

    Parameters
    ----------
    y_true : 1d ndarray
        Target value.

    y_pred : 1d ndarray
        Predicted value.

    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_true - y_pred
    return mean(np.maximum(quantile * residual, (quantile - 1) * residual))



def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe


def light_gbm_regressor(rfe=False, rfe_feature_n=20, shapley_calc=True):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "bs_support.csv"))
    print("Median Support: ")
    print(df["support"].median())
    df.columns = df.columns.str.replace(':', '_')
    df.fillna(-1, inplace=True)
    df.replace([np.inf, -np.inf], -1, inplace=True)

    print(df.columns)
    print(df.shape)

    df["group"] = df['dataset'].astype('category').cat.codes.tolist()

    target = "support"
    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    y_test = test[target]

    #X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, test_size=0.2,
     #                                                                              random_state=42)
    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)
    print("Baseline predicting mean RMSE: " + str(rmse_mean))

    if rfe:
        model = GradientBoostingRegressor(n_estimators=250, max_depth=10, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'branchId', 'group']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'branchId', 'group']).columns[rfe.support_]
        selected_features = selected_features.append(pd.Index(['group']))

        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset', 'branchId'])
        X_test = X_test.drop(axis=1, columns=['dataset', 'branchId'])

    #####################################################################################################################

    def objective_median(trial):
        #callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {
            'loss': 'quantile',
            'alpha': 0.5,
            'n_iter_no_change': 5,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of boosting stages to be used
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # Learning rate
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Maximum depth of the individual trees
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            # Minimum samples required to split an internal node
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            # Minimum number of samples required to be at a leaf node
        }

        val_scores = []

        gkf = GroupKFold(n_splits=6)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val)#, reference=train_data)
            # KEIN VALIDSETS?
            model = GradientBoostingRegressor(**params)
            model = model.fit(X_train_tmp, y_train_tmp)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.5)
            print("score: " + str(val_score))

            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_median, n_trials=50)

    best_params_median = study.best_params
    best_score_median = study.best_value

    print(f"Best Params: {best_params_median}")
    print(f"Best MAPE training: {best_score_median}")

    #train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)
    model = GradientBoostingRegressor(**best_params_median)
    final_model_median = model.fit(X_train.drop(axis=1, columns=["group"]), y_train)
    #final_model_median = lgb.train(best_params_median, train_data)

    y_pred = final_model_median.predict(X_test.drop(axis=1, columns=["group"]))

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae:.2f}")

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE on test set: {mape}")

    mdae = median_absolute_error(y_test, y_pred)
    print(f"MDAE on test set: {mdae}")

    mbe = MBE(y_test, y_pred)
    print(f"MBE on test set: {mbe}")


    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference

    # Save the plot as an image file (e.g., PNG)
    plt.savefig("residual_plot.png")

    feature_importance = final_model_median.feature_importance(importance_type='gain')

    importance_df = pd.DataFrame(
        {'Feature': X_train.drop(axis=1, columns=["group"]).columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    scaler = MinMaxScaler()
    importance_df['Importance'] = scaler.fit_transform(importance_df[['Importance']])
    importance_df = importance_df.nlargest(30, 'Importance')

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()

    name = "8000"
    if rfe:
        name = name + "_rfe_" + str(rfe_feature_n)

    plot_filename = os.path.join(os.pardir, "data/prediction", "feature_importances_" + name + ".png")
    plt.savefig(plot_filename)

    print("Feature Importances (Normalized):")
    for index, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    X_test_["prediction"] = y_pred
    X_test_["support"] = y_test
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + ".csv"))

    if shapley_calc:
        # X_test = X_test_[(abs(X_test_['entropy'] - X_test_['prediction']) < 0.05) & (
        #       (X_test_['entropy'] < 0.1) | (X_test_['entropy'] > 0.9))]
        X_test = X_test_
        X_test = X_test.sort_values(by="entropy")
        explainer = shap.Explainer(final_model_median,
                                   X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId", "group"]),
                                   check_additivity=False)
        shap_values = explainer(X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId", "group"]),
                                check_additivity=False)

        shap.summary_plot(shap_values, X_test.drop(columns=["entropy", "prediction", "dataset", "sampleId", "group"]),
                          plot_type="bar")
        plt.savefig(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + "shap.png"))

        plt.figure(figsize=(10, 6))

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[0], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm_0.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[1500], max_display=10)  # Limit the display to 10 features
        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm_1500.png")

        plt.figure(figsize=(10, 6))  # Adjust width and height as needed

        # Create the waterfall plot
        shap.initjs()  # Initialize JavaScript visualization
        shap.plots.waterfall(shap_values[-300], max_display=10)  # Limit the display to 10 features

        plt.xlabel("SHAP Value", fontsize=14)  # Adjust x-axis label font size
        plt.ylabel("Feature", fontsize=14)  # Adjust y-axis label font size
        plt.xticks(fontsize=12)  # Adjust x-axis tick font size
        plt.yticks(fontsize=12)  # Adjust y-axis tick font size
        plt.tight_layout()  # Adjust layout to prevent overlapping elements
        plt.savefig("lgbm-300.png")

    ######################################################################################################
    def objective_lower_bound(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {
            'loss': 'quantile',
            'alpha': 0.05,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of boosting stages to be used
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # Learning rate
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Maximum depth of the individual trees
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'n_iter_no_change': 5,

            # Minimum samples required to split an internal node
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            # Minimum number of samples required to be at a leaf node
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            # KEIN VALIDSETS?
            model = GradientBoostingRegressor(**params)
            model = model.fit(X_train_tmp.drop(axis=1, columns=["group"]), y_train_tmp)
            #model = lgb.train(params, train_data)#, valid_sets=[val_data])
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.05)
            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lower_bound, n_trials=50)

    best_params_lower_bound = study.best_params
    best_score_lower_bound = study.best_value

    print(f"Best Params: {best_params_lower_bound}")
    print(f"Best Quantile Loss: {best_score_lower_bound}")

    #train_data = lgb.Dataset()
    model = GradientBoostingRegressor(**best_params_lower_bound)
    final_model_lower_bound = model.fit(X_train.drop(axis=1, columns=["group"]), y_train)

    y_pred_lower = final_model_lower_bound.predict(X_test.drop(axis=1, columns=["group"]))
    print("Quantile Loss on Holdout: " + str(quantile_loss(y_test, y_pred_lower, 0.05)))

    #########################################################################################################

    def objective_upper_bound(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {
            'loss': 'quantile',
            'alpha': 0.95,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of boosting stages to be used
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # Learning rate
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Maximum depth of the individual trees
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'n_iter_no_change': 5,

            # Minimum samples required to split an internal node
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            # Minimum number of samples required to be at a leaf node
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            # KEIN VALIDSETS?
            model = GradientBoostingRegressor(**params)
            model = model.fit(X_train_tmp, y_train_tmp)
           # model = lgb.train(params, train_data)#, valid_sets=[val_data])
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.95)
            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_upper_bound, n_trials=100)

    best_params_upper_bound = study.best_params
    best_score_upper_bound = study.best_value

    print(f"Best Params: {best_params_upper_bound}")
    print(f"Best Quantile Loss: {best_score_upper_bound}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)
    model = GradientBoostingRegressor(**best_params_upper_bound)
    final_model_upper_bound = model.fit(X_train.drop(axis=1, columns=["group"]), y_train)
    #äfinal_model_upper_bound = lgb.train(best_params_upper_bound, train_data)

    y_pred_upper = final_model_upper_bound.predict(X_test.drop(axis=1, columns=["group"]))
    print("Quantile Loss on Holdout: " + str(quantile_loss(y_test, y_pred_upper, 0.95)))

    result_df = pd.DataFrame({'upper_bound': y_pred_upper, 'lower_bound': y_pred_lower, 'pred': y_pred, 'support': y_test})
    result_df.to_csv(os.path.join(os.pardir, "data/prediction", "bs_support_pred_quant.csv"))

light_gbm_regressor(rfe=True, shapley_calc=False)
