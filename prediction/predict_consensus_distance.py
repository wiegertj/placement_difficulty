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
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)
    diff = (y_true-y_pred)
    mbe = diff.mean()
    return mbe
def light_gbm_regressor(rfe=False, rfe_feature_n=20, shapley_calc=True):
    df_msa = pd.read_csv(os.path.join(os.pardir, "data/processed/features", "msa_features.csv"))

    #pred topo
    df_difference = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features", "cons_comp_target.csv")) #1000
    df_difference.drop(axis=1, columns=["quartet"])


    # pred cons
    #df_difference = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features", "cons_comp1000.csv")) #20.000



    #df_difference2 = pd.read_csv(os.path.join(os.pardir, "data/processed/features/bs_features", "cons_comp_500parsboot.csv"))
    #df_difference2["dataset"] = df_difference2["dataset"].str.replace(".newick", "")
    #df_difference2["quartet_2"] = df_difference2["quartet"]
    #df_difference2["nrf_2"] = df_difference2["nrf"]
    #df_difference2.drop(axis=1, columns=["nrf", "quartet"], inplace=True)
    print(df_difference.shape)
    #print(df_difference.mean())
    #print(np.std(df_difference))
    import matplotlib.pyplot as plt

    # Assuming you have a DataFrame called 'df_difference'

    # Extract the "nrf" column
    nrf_values = df_difference["nrf"]

    # Create a histogram
    plt.hist(nrf_values, bins=20, edgecolor='k')  # You can adjust the number of bins as needed

    # Add labels and title
    plt.xlabel('nrf Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of nrf Values')

    # Save the histogram as an image file (e.g., PNG)
    plt.savefig('nrf_histogram.png')

    # Show the histogram (optional)
    plt.show()
    print(df_difference[df_difference["nrf"] < 0.1].shape)

    parsimony_features2 = pd.read_csv(
        os.path.join(os.pardir, "data/processed/features/bs_features/pars_top_features_no_model.csv"),
        usecols=lambda column: column != 'Unnamed: 0')

    df_difference["dataset"] = df_difference["dataset"].str.replace(".newick", "")
   # df_difference = df_difference.merge(df_difference2, on=["dataset"], how="inner")
    df = df_msa.merge(df_difference, on=["dataset"], how="inner")
    df = df.merge(parsimony_features2, on=["dataset"], how="inner")
    print("Median Support: ")
    #print(df["quartet"].median())
    df.columns = df.columns.str.replace(':', '_')

    print(df.columns)
    print(df.shape)

    df["group"] = df['dataset'].astype('category').cat.codes.tolist()


    target = "nrf"

    #X = df.drop(axis=1, columns=target)
    #y = df[target]

    #X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, df["group"], test_size=0.2,
     #                                                                              random_state=12)

    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    #X_train = X_train.drop(axis=1, columns=["quartet"])
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)
    #X_test = X_test.drop(axis=1, columns=["quartet"])

    y_test = test[target]



    X_train.fillna(-1, inplace=True)
    X_train.replace([np.inf, -np.inf], -1, inplace=True)




    mse_zero = mean_squared_error(y_test, np.zeros(len(y_test)))
    rmse_zero = math.sqrt(mse_zero)
    print("Baseline prediting 0 RMSE: " + str(rmse_zero))

    mse_mean = mean_squared_error(y_test, np.zeros(len(y_test)) + mean(y_train))
    rmse_mean = math.sqrt(mse_mean)
    print("Baseline predicting mean RMSE: " + str(rmse_mean))

    mse = mean_squared_error(y_test,  np.zeros(len(y_test)) + mean(y_train))
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    r_squared = r2_score(y_test,  np.zeros(len(y_test)) + mean(y_train))
    print(f"R-squared on test set: {r_squared:.2f}")

    mae = mean_absolute_error(y_test,  np.zeros(len(y_test)) + mean(y_train))
    print(f"MAE on test set: {mae:.2f}")

    mape = median_absolute_error(y_test,  np.zeros(len(y_test)) + mean(y_train))
    print(f"MdAE on test set: {mape}")


    if rfe:
        model = RandomForestRegressor(n_jobs=-1, n_estimators=250, max_depth=20, min_samples_split=20,
                                      min_samples_leaf=10)
        rfe = RFE(estimator=model, n_features_to_select=rfe_feature_n, step=0.1)  # Adjust the number of features as needed
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
        #callbacks = [LightGBMPruningCallback(trial, 'l1')]

        params = {
            'objective': 'regression',
            'metric': 'l1',
            'num_iterations': trial.suggest_int('num_iterations', 100, 500),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 300),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            #'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.1),
            'bagging_freq': 0,
            'verbosity': -1
            #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            # KEIN VALIDSETS?
            model = lgb.train(params, train_data)#, valid_sets=[val_data])
            val_preds = model.predict(X_val)
            #val_score = mean_squared_error(y_val, val_preds)
            #val_score = math.sqrt(val_score)
            val_score = mean_absolute_error(y_val, val_preds)
            val_scores.append(val_score)

        return np.mean(val_scores)#sum(val_scores) / len(val_scores) #median?

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_score = study.best_value

    print(f"Best Params: {best_params}")
    print(f"Best MAPE training: {best_score}")

    train_data = lgb.Dataset(X_train.drop(axis=1, columns=["group"]), label=y_train)

    final_model = lgb.train(best_params, train_data)

    y_pred = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error on test set: {rmse}")

    r_squared = r2_score(y_test, y_pred)
    print(f"R-squared on test set: {r_squared:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test set: {mae:.2f}")

    mape = median_absolute_error(y_test, y_pred)
    print(f"MdAE on test set: {mape}")


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
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", "prediction_results" + name + ".csv"))

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
