import math
import pickle
import sys
import random
import shap
import lightgbm as lgb
import os
import sklearn.metrics as metrics
from scipy.stats import entropy

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
    median_absolute_error, log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback


def light_gbm_regressor(rfe=False, rfe_feature_n=10, shapley_calc=True):
    df_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"), usecols=lambda column: column != 'Unnamed: 0')
    df_target = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "split_prediction.csv"), usecols=lambda column: column != 'Unnamed: 0')
    print("unique datasets: ")
    print(len(df_target["dataset"].unique()))
    df = df_msa.merge(df_target, on=["dataset"], how="inner")
    print("unique datasets after msa: ")
    print(len(df["dataset"].unique()))
    parsimony_features2 = pd.read_csv(
        os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features_no_model.csv"),
        usecols=lambda column: column != 'Unnamed: 0')

    difficulties_path = os.path.join(os.pardir, "data/treebase_difficulty_new.csv")
    difficulties_df = pd.read_csv(difficulties_path, index_col=False, usecols=lambda column: column != 'Unnamed: 0')
    difficulties_df = difficulties_df.drop_duplicates(subset=['name'], keep='first')
    difficulties_df["dataset"] = difficulties_df["name"].str.replace(".phy", "")
    difficulties_df = difficulties_df[["dataset", "difficulty"]]

    df = df.merge(difficulties_df, on=["dataset"], how="inner")
    print("unique datasets after diff: ")
    print(len(df["dataset"].unique()))
    df = df.merge(parsimony_features2, on=["dataset"], how="inner")
    print("unique datasets after topo: ")
    print(len(df["dataset"].unique()))
    value_counts = df['inML'].value_counts()
    print(value_counts)
    df["group"] = df['dataset'].astype('category').cat.codes.tolist()
    print(df.columns)
    print(df.shape)

    print("####"*10)
    print("Baseline")

    val_preds_binary_baseline = (df["pars_support_cons"] > 0.7).astype(int)

    accuracy = accuracy_score(df["inML"], val_preds_binary_baseline)
    print(accuracy)
    precision = precision_score(df["inML"], val_preds_binary_baseline)
    recall = recall_score(df["inML"], val_preds_binary_baseline)
    f1 = f1_score(df["inML"], val_preds_binary_baseline)
    roc_auc = roc_auc_score(df["inML"], val_preds_binary_baseline)

    plt.scatter(df["inML"], val_preds_binary_baseline)

    # Add labels to the axes
    plt.xlabel("df['inML']")
    plt.ylabel("pars_support_cons")

    # Save the scatterplot to a file
    plt.savefig("scatterplot_baseline.png")


    print("####"*10)


    df.to_csv(os.path.join(os.pardir, "data/processed/features/split_features/all_data.csv"))

    target = "inML"

    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    X_train = X_train[["dataset",'pars_support_cons', 'std_pars_supp_parents', 'min_pars_supp_child_w',
       'std_pars_supp_child_w', 'min_pars_supp_child', 'mean_pars_supp_child',
       'std_pars_supp_child', 'irs_std_right', 'irs_skw_right',
       'avg_rf_no_boot', 'group']]
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    X_test = X_test[["dataset", 'pars_support_cons', 'std_pars_supp_parents', 'min_pars_supp_child_w',
       'std_pars_supp_child_w', 'min_pars_supp_child', 'mean_pars_supp_child',
       'std_pars_supp_child', 'irs_std_right', 'irs_skw_right',
       'avg_rf_no_boot', 'group']]
    y_test = test[target]

    X_train.fillna(-1, inplace=True)

    X_train.replace([np.inf, -np.inf], -1, inplace=True)

    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)
    print("Baseline predicting mean RMSE: " + str(rmse_mean))

    mse = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    r_squared = r2_score(y_test, np.zeros(len(y_test)) + mean(y_train))
    print(f"R-squared on test set: {r_squared:.2f}")

    mae = mean_absolute_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    print(f"MAE on test set: {mae:.2f}")

    mape = median_absolute_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    print(f"MdAE on test set: {mape}")

    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=20, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n,
                  step=0.1)  # Adjust the number of features as needed
        rfe.fit(X_train.drop(axis=1, columns=['dataset', 'group']), y_train)
        print(rfe.support_)
        selected_features = X_train.drop(axis=1, columns=['dataset', 'group']).columns[rfe.support_]
        selected_features = selected_features.append(pd.Index(['group']))

        print("Selected features for RFE: ")
        print(selected_features)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    X_test_ = X_test
    if not rfe:
        X_train = X_train.drop(axis=1, columns=['dataset'])
        X_test = X_test.drop(axis=1, columns=['dataset'])

    def objective(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]





        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',  # Use 'binary_logloss' for binary classification
            'num_iterations': trial.suggest_int('num_iterations', 10, 200),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 300),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.1),
            'bagging_freq': 0,
            'verbosity': -1
            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
        }

        val_scores = []

        gkf = GroupKFold(n_splits=5)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            print("In prediction")
            print(X_train_tmp.columns)
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            # KEIN VALIDSETS?
            model = lgb.train(params, train_data)  # , valid_sets=[val_data])
            val_preds = model.predict(X_val)
            val_preds_binary = (val_preds > 0.5).astype(int)

            # val_score = mean_squared_error(y_val, val_preds)
            # val_score = math.sqrt(val_score)
            val_score = accuracy_score(y_val, val_preds_binary)
            val_scores.append(val_score)

        return np.mean(val_scores)  # sum(val_scores) / len(val_scores) #median?

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "binary_logloss"
    best_params["bagging_freq"] = 0
    best_score = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best F1 training: {best_score}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model = lgb.train(best_params, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", "branch_predictor.pkl")
    #with open(model_path, 'wb') as file:
     #   pickle.dump(final_model, file)

    y_pred = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    # Convert probabilities to class labels (binary classification)
    y_pred_binary = (y_pred > 0.5).astype(int)
    entropy_values = [entropy([p, 1 - p], base=2) for p in y_pred]

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)


    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference

    # Save the plot as an image file (e.g., PNG)
    plt.savefig("residual_plot.png")

    feature_importance = final_model.feature_importance(importance_type='gain')

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
    X_test_["uncertainty"] = entropy_values
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", "prediction_results_classifier" + name + ".csv"))

    if shapley_calc:
        # X_test = X_test_[(abs(X_test_['entropy'] - X_test_['prediction']) < 0.05) & (
        #       (X_test_['entropy'] < 0.1) | (X_test_['entropy'] > 0.9))]
        X_test = X_test_
        X_test = X_test.sort_values(by="entropy")
        explainer = shap.Explainer(final_model,
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




light_gbm_regressor(rfe=False, shapley_calc=False)
