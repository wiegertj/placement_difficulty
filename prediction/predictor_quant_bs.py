import math
import sys
import random
import shap
import lightgbm as lgb
import os
import pickle

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import sp_version, parse_version
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback


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

    df = df[["dataset", "branchId", "support", "parsimony_boot_support",
             "parsimony_support",
             "avg_subst_freq",
             "length_relative",
             "length",
             "avg_rel_rf_boot",
             "max_subst_freq",
             "skw_pars_bootsupp_tree",
             "cv_subst_freq",
             "bl_ratio",
             "max_pars_bootsupp_child_w",
             "sk_subst_freq",
             "mean_pars_bootsupp_parents",
             "max_pars_supp_child_w",
             "std_pars_bootsupp_parents",
             "min_pars_supp_child",
             "min_pars_supp_child_w",
             "rel_num_children",
             "mean_pars_supp_child_w",
             "std_pars_bootsupp_child",
             "mean_clo_sim_ratio",
             "min_pars_bootsupp_child_w"]]

    column_name_mapping = {
        "parsimony_boot_support": "parsimony_bootstrap_support",
        "parsimony_support": "parsimony_support",
        "avg_subst_freq": "mean_substitution_frequency",
        "length_relative": "norm_branch_length",
        "length": "branch_length",
        "avg_rel_rf_boot": "mean_norm_rf_distance",
        "max_subst_freq": "max_substitution_frequency",
        "skw_pars_bootsupp_tree": "skewness_bootstrap_pars_support_tree",
        "cv_subst_freq": "cv_substitution_frequency",
        "bl_ratio": "branch_length_ratio_split",
        "max_pars_bootsupp_child_w": "max_pars_bootstrap_support_children_w",
        "sk_subst_freq": "skw_substitution_frequency",
        "mean_pars_bootsupp_parents": "mean_pars_bootstrap_support_parents",
        "max_pars_supp_child_w": "max_pars_support_children_weighted",
        "std_pars_bootsupp_parents": "std_pars_bootstrap_support_parents",
        "min_pars_supp_child": "min_pars_support_children",
        "min_pars_supp_child_w": "min_pars_support_children_weighted",
        "rel_num_children": "number_children_relative",
        "mean_pars_supp_child_w": "mean_pars_support_children_weighted",
        "std_pars_bootsupp_child": "std_pars_bootstrap_support_children",
        "mean_clo_sim_ratio": "mean_closeness_centrality_ratio",
        "min_pars_bootsupp_child_w": "min_pars_bootstrap_support_children_w"
    }

    # Rename the columns in the DataFrame
    df = df.rename(columns=column_name_mapping)

    # df_diff = pd.read_csv(os.path.join(os.pardir, "data/treebase_difficulty_new.csv"))
    # df_diff["name"] = df_diff["name"].str.replace(".phy", "")
    # df = df.merge(df_diff, left_on="dataset", right_on="name", how="inner")
    # df.drop(columns=["datatype", "name"], axis=1, inplace=True)

    df.fillna(-1, inplace=True)
    df.replace([np.inf, -np.inf], -1, inplace=True)
    print("Median Support: ")
    print(df["support"].median())
    df.columns = df.columns.str.replace(':', '_')

    print(df.columns)
    print(df.shape)

    df["group"] = df['dataset'].astype('category').cat.codes.tolist()

    target = "support"

    ######
    # list_to_delete = ["17984_0", "10965_0", "17331_0", "18577_0", "21602_10"]
    # df = df[~df['dataset'].isin(list_to_delete)]

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
    loo_selection = loo_selection[:200]
    filenames = loo_selection["dataset"].values.tolist()

    test = df[df['dataset'].isin(filenames)]
    train = df[~df['dataset'].isin(filenames)]

    # print(test.shape)
    # print(train.shape)

    #####

    # sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    # test = df[df['group'].isin(sample_dfs)]
    # train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    y_test = test[target]

    # X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, test_size=0.2,
    #                                                                              random_state=42)
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

    ######################################

    def objective_median(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': 0.5,
            'num_iterations': trial.suggest_int('num_iterations', 100, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.5),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            "verbosity": -1

            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
        }

        val_scores = []

        gkf = GroupKFold(n_splits=6)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val)  # , reference=train_data)
            # KEIN VALIDSETS?
            model = lgb.train(params, train_data)  # , valid_sets=[val_data])
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.5)
            print("score: " + str(val_score))

            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_median, n_trials=50)

    best_params = study.best_params
    best_params["objective"] = "quantile"
    best_params["metric"] = "quantile"
    best_params["boosting_type"] = "gbdt"
    best_params["bagging_freq"] = 0
    best_params["alpha"] = 0.5
    best_params["verbosity"] = -1
    best_score_median = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best MAPE training: {best_score_median}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model = lgb.train(best_params, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", "mean_model90_test_nonuni.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)

    y_pred_median = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    mse = mean_squared_error(y_test, y_pred_median)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    r_squared = r2_score(y_test, y_pred_median)
    print(f"R-squared on test set: {r_squared:.2f}")

    mae = mean_absolute_error(y_test, y_pred_median)
    print(f"MAE on test set: {mae:.2f}")

    mape = median_absolute_error(y_test, y_pred_median)
    print(f"MdAE on test set: {mape}")

    residuals = y_test - y_pred_median

    plt.scatter(y_pred_median, residuals)
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

    print(importance_df)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()

    sys.exit()

    #####################################################################################################################
    # X_test = X_test[["parsimony_support", "length", 'min_pars_supp_child_w', 'split_std_ratio_branch', 'group']]
    # X_train = X_train[["parsimony_support", "length", 'min_pars_supp_child_w', 'split_std_ratio_branch', 'group']]
    X_test = X_test[["parsimony_bootstrap_support", "parsimony_support", "mean_substitution_frequency",
                     "branch_length", "norm_branch_length", "max_substitution_frequency", 'group']]
    X_train = X_train[["parsimony_bootstrap_support", "parsimony_support", "mean_substitution_frequency",
                       "branch_length", "norm_branch_length", "max_substitution_frequency", 'group']]

    def objective_lower_bound(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {

            'alpha': trial.suggest_float('alpha', 0.0001, 0.1),

        }

        val_scores = []

        gkf = GroupKFold(n_splits=3)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val)  # , reference=train_data)
            # KEIN VALIDSETS?
            solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

            model = QuantileRegressor(**params, quantile=0.05, solver=solver)

            model = model.fit(X_train_tmp, y_train_tmp)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.05)
            print("score: " + str(val_score))

            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lower_bound, n_trials=6)

    best_params_lo = study.best_params
    best_score_lo = study.best_value

    print(f"Best Params: {best_params_lo}")
    print(f"Best MAPE training: {best_score_lo}")

    model_lo = QuantileRegressor(**best_params_lo, quantile=0.05, solver=solver).fit(
        X_train.drop(axis=1, columns=["group"]), y_train)

    model_path = os.path.join(os.pardir, "data/processed/final", "low_model90_test.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model_lo, file)

    y_pred_lo = model_lo.predict(X_test.drop(axis=1, columns=["group"]))
    quant_loss_lo = quantile_loss(y_test, y_pred_lo, 0.05)
    print(f"Quantile Loss Holdout: {quant_loss_lo}")
    mse = mean_squared_error(y_test, y_pred_lo)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    mae = mean_absolute_error(y_test, y_pred_lo)
    print(f"MAE on test set: {mae:.2f}")

    mape = mean_absolute_percentage_error(y_test, y_pred_lo)
    print(f"MAPE on test set: {mape}")

    mdae = median_absolute_error(y_test, y_pred_lo)
    print(f"MDAE on test set: {mdae}")

    mbe = MBE(y_test, y_pred_lo)
    print(f"MBE on test set: {mbe}")

    ######################################################################################################
    def objective_higher_bound(trial):
        # callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {

            'alpha': trial.suggest_float('alpha', 0.0001, 0.1),

        }

        val_scores = []

        gkf = GroupKFold(n_splits=3)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val)  # , reference=train_data)
            # KEIN VALIDSETS?
            model = QuantileRegressor(**params, quantile=0.85, solver=solver).fit(X_train_tmp, y_train_tmp)
            val_preds = model.predict(X_val)
            val_score = quantile_loss(y_val, val_preds, 0.85)
            print("score: " + str(val_score))

            val_scores.append(val_score)

        return sum(val_scores) / len(val_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_higher_bound, n_trials=6)
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

    best_params_hi = study.best_params
    best_score_hi = study.best_value

    print(f"Best Params: {best_params_hi}")
    print(f"Best Q training: {best_score_hi}")

    model_hi = QuantileRegressor(**best_params_hi, quantile=0.85, solver=solver).fit(
        X_train.drop(axis=1, columns=["group"]), y_train)

    model_path = os.path.join(os.pardir, "data/processed/final", "high_model90_test.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model_hi, file)

    y_pred_hi = model_hi.predict(X_test.drop(axis=1, columns=["group"]))

    mse = mean_squared_error(y_test, y_pred_hi)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    mae = mean_absolute_error(y_test, y_pred_hi)
    print(f"MAE on test set: {mae:.2f}")

    mape = mean_absolute_percentage_error(y_test, y_pred_hi)
    print(f"MAPE on test set: {mape}")

    mdae = median_absolute_error(y_test, y_pred_hi)
    print(f"MDAE on test set: {mdae}")

    mbe = MBE(y_test, y_pred_hi)
    print(f"MBE on test set: {mbe}")

    quant_loss_lo = quantile_loss(y_test, y_pred_lo, 0.05)
    print(f"Quantile Loss Holdout: {quant_loss_lo}")

    quant_loss_hi = quantile_loss(y_test, y_pred_hi, 0.85)
    print(f"Quantile Loss Holdout: {quant_loss_hi}")

    X_test_["prediction_median"] = y_pred_median
    X_test_["prediction_low"] = y_pred_lo
    X_test_["prediction_hi"] = y_pred_hi
    X_test_["support"] = y_test
    X_test_["pred_error"] = y_test - y_pred_median
    X_test_["pi_width"] = y_pred_hi - y_pred_lo

    X_test_.to_csv(os.path.join(os.pardir, "data/processed/final", "pred_interval_90_final.csv"))


light_gbm_regressor(rfe=False, shapley_calc=False)
