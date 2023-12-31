import math
import sys
import random
import shap
import lightgbm as lgb
import os
import optuna
from sklearn.linear_model import QuantileRegressor
import numpy as np
from sklearn.utils.fixes import sp_version, parse_version
import pickle
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
    print("Median Support: ")
    print(df["support"].median())
    df.columns = df.columns.str.replace(':', '_')

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
        X_train = X_train.drop(axis=1, columns=['dataset'])
        X_test = X_test.drop(axis=1, columns=['dataset'])

    ############################################LR#########################################################################

    from sklearn.linear_model import Ridge

    def objective_lr(trial):
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1000)

        val_scores_lr = []

        gkf = GroupKFold(n_splits=6)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_tmp)
            X_valid_scaled = scaler.fit_transform(X_val)
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train_tmp)

            val_preds = model.predict(X_valid_scaled)
            val_score = mean_absolute_error(y_val, val_preds)
            val_scores_lr.append(val_score)

        return sum(val_scores_lr) / len(val_scores_lr)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lr, n_trials=100)

    best_params_lr = study.best_params
    best_score_lr = study.best_value
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(axis=1, columns=["group"]))
    X_test_scaled = scaler.fit_transform(X_test.drop(axis=1, columns=["group"]))
    final_model_lr = Ridge(best_params_lr["alpha"])
    final_model_lr.fit(X_train_scaled, y_train)
    y_pred = final_model_lr.predict(X_test_scaled)

    mae_lr = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae_lr:.2f}")

    #####################################################################################################################

    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 32)
        min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)

        val_scores_rf = []

        gkf = GroupKFold(n_splits=6)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_tmp)
            X_valid_scaled = scaler.fit_transform(X_val)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_scaled, y_train_tmp)

            val_preds = model.predict(X_valid_scaled)
            val_score = mean_absolute_error(y_val, val_preds)
            val_scores_rf.append(val_score)

        return sum(val_scores_rf) / len(val_scores_rf)

    # Perform 100 Optuna trials
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_rf, n_trials=100)

    # Get the best hyperparameters
    best_params_rf = study.best_params
    print("Best Hyperparameters:", best_params_rf)

    # Train the final model with the best hyperparameters using the full dataset
    final_model_rf = RandomForestRegressor(**best_params_rf, random_state=42)
    X_train_scaled = MinMaxScaler().fit_transform(X_train.drop(axis=1, columns=["group"]))
    final_model_rf.fit(X_train_scaled, y_train)

    # Evaluate the final model on the test set
    X_test_scaled = MinMaxScaler().fit_transform(X_test.drop(axis=1, columns=["group"]))
    y_test_pred = final_model_rf.predict(X_test_scaled)
    mae_rf = mean_absolute_error(y_test, y_test_pred)
    print(f"MAE on test set: {mae_rf:.2f}")

    df_res = pd.DataFrame([(mae_lr, mae_rf)], columns=["mae_r", "mae_rf"])

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       "df_res.csv")):
        df_res.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                "df_res.csv")), index=False)
    else:
        df_res.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                   "df_res.csv"),
                      index=False,
                      mode='a', header=False)

    return

    #####################################################################################################################

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
    study.optimize(objective_median, n_trials=100)

    best_params_median = study.best_params
    best_params_median["objective"] = "quantile"
    best_params_median["metric"] = "quantile"
    best_params_median["boosting_type"] = "gbdt"
    best_params_median["bagging_freq"] = 0
    best_params_median["alpha"] = 0.5
    best_params_median["verbosity"] = -1
    best_score_median = study.best_value

    print(f"Best Params: {best_params_median}")
    print(f"Best MAPE training: {best_score_median}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model_median = lgb.train(best_params_median, train_data)

    model_path = os.path.join(os.pardir, "data/processed/final", "median_model90_test_skl.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model_median, file)

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


for i in range(0, 9):
    light_gbm_regressor(rfe=False, shapley_calc=False)

#########    #########
#########    #########
