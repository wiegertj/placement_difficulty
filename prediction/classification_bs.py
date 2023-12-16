import math
import pickle
import sys
import random
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
    median_absolute_error, log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from optuna.integration import LightGBMPruningCallback


def light_gbm_regressor(cutoff, rfe=False, rfe_feature_n=10, shapley_calc=True):
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "bs_support_500.csv"))
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

    df_raxml = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "df_pred.csv"), usecols=lambda
        column: column != 'Unnamed: 0' and column != 'Unnamed: 0_x' and column != 'Unnamed: 0_y')
    df_raxml_filtered = df_raxml[["dataset", "branchId", "support", "parsimony_bootstrap_support",
                                  "parsimony_support",
                                  "mean_substitution_frequency",
                                  "norm_branch_length",
                                  "branch_length",
                                  "mean_norm_rf_distance",
                                  "max_substitution_frequency",
                                  "skewness_bootstrap_pars_support_tree",
                                  "cv_substitution_frequency",
                                  "branch_length_ratio_split",
                                  "max_pars_bootstrap_support_children_w",
                                  "skw_substitution_frequency",
                                  "mean_pars_bootstrap_support_parents",
                                  "max_pars_support_children_weighted",
                                  "std_pars_bootstrap_support_parents",
                                  "min_pars_support_children",
                                  "min_pars_support_children_weighted",
                                  "number_children_relative",
                                  "mean_pars_support_children_weighted",
                                  "std_pars_bootstrap_support_children",
                                  "mean_closeness_centrality_ratio",
                                  "min_pars_bootstrap_support_children_w"]]
    df_raxml_filtered["support"] = df_raxml_filtered["support"] / 100
    print(df.shape)
    df = pd.concat([df, df_raxml_filtered])
    print(df.shape)






    df_reg_pred = df.drop(columns=["dataset", "support"], axis=1)
    with open("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/final/100median_model_final_ebg.pkl",
              'rb') as model_file:
        regression_median = pickle.load(model_file)

    with open("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/final/100low_model_10_final_ebg.pkl",
              'rb') as model_file:
        regression_lower10 = pickle.load(model_file)

    with open("/hits/fast/cme/wiegerjs/placement_difficulty/data/processed/final/100low_model_5_final_ebg.pkl",
              'rb') as model_file:
        regression_lower5 = pickle.load(model_file)

    #df["median_pred"] = regression_median.predict(df_reg_pred)
    #df["lower_bound_10"] = regression_lower10.predict(df_reg_pred)
    #df["lower_bound_5"] = regression_lower5.predict(df_reg_pred)










    print("Median Support: ")
    print(df["support"].median())
    df.columns = df.columns.str.replace(':', '_')
    df["is_valid"] = 0
    print(df["support"])
    print(df["parsimony_bootstrap_support"])
    df.loc[df['support'] > cutoff, 'is_valid'] = 1
    print(df["is_valid"].value_counts())
    print(df.columns)
    print(df.shape)

    print("####" * 10)
    print("Baseline")
    accuracy_best = -10



    print("####" * 10)

    df["group"] = df['dataset'].astype('category').cat.codes.tolist()
    target = "is_valid"
    df.drop(columns=["support"], inplace=True, axis=1)
    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]
    print(test["parsimony_bootstrap_support"])
    val_preds_binary_baseline = (test["parsimony_bootstrap_support"] > cutoff).astype(int)
    accuracy_baseline = balanced_accuracy_score(test["is_valid"], val_preds_binary_baseline)
    f1_baseline = f1_score(test["is_valid"], val_preds_binary_baseline)
    roc_baseline = roc_auc_score(test["is_valid"], val_preds_binary_baseline)
    precision_baseline = precision_score(test["is_valid"], val_preds_binary_baseline)
    recall_baseline = recall_score(test["is_valid"], val_preds_binary_baseline)

    #data = {"acc": 0,
     #       "pre": 0,
      #      "rec": 0,
       #     "roc_auc": 0,
        #    "f1": 0,
         #   "acc_baseline": accuracy_baseline,
          #  "pre_baseline": precision_baseline,
            #"rec_baseline": recall_baseline,
           # "roc_auc_baseline": roc_baseline,
            #"f1_baseline": f1_baseline
            #}
    #data_list = [data]

    #time_dat = pd.DataFrame(data_list)

    #if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
     #                                  f"classifier_metrics{cutoff}.csv")):
      #  time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
       #                                           f"classifier_metrics{cutoff}.csv")), index=False)
    #else:
     #   time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
      #                               f"classifier_metrics{cutoff}.csv"),
       #                 index=False,
        #                mode='a', header=False)

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
    loo_selection = loo_selection[:180]
    filenames = loo_selection["dataset"].values.tolist()

    test = df[df['dataset'].isin(filenames)]
    train = df[~df['dataset'].isin(filenames)]


    sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
    test = df[df['group'].isin(sample_dfs)]
    train = df[~df['group'].isin(sample_dfs)]

    X_train = train.drop(axis=1, columns=target)
    y_train = train[target]

    X_test = test.drop(axis=1, columns=target)

    y_test = test[target]

    X_train.fillna(-1, inplace=True)

    X_train.replace([np.inf, -np.inf], -1, inplace=True)

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
            'num_iterations': trial.suggest_int('num_iterations', 10, 300),
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 300),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.3),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_uniform('lambda_l1', 1e-5, 1.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 1e-5, 1.0),
            'min_split_gain': trial.suggest_uniform('min_split_gain', 1e-5, 0.3),
            'bagging_freq': 0,
            'verbosity': -1
            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
        }

        val_scores = []

        gkf = GroupKFold(n_splits=10)
        for train_idx, val_idx in gkf.split(X_train.drop(axis=1, columns=['group']), y_train, groups=X_train["group"]):
            X_train_tmp, y_train_tmp = X_train.drop(axis=1, columns=['group']).iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.drop(axis=1, columns=['group']).iloc[val_idx], y_train.iloc[val_idx]

            train_data = lgb.Dataset(X_train_tmp, label=y_train_tmp)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            # KEIN VALIDSETS?
            model = lgb.train(params, train_data)  # , valid_sets=[val_data])
            val_preds = model.predict(X_val)
            val_preds_binary = (val_preds >= 0.5).astype(int)

            # val_score = mean_squared_error(y_val, val_preds)
            # val_score = math.sqrt(val_score)
            val_score = f1_score(y_val, val_preds_binary)
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

    model_path = os.path.join(os.pardir, "data/processed/final", f"final_class_{str(cutoff)}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)

    y_pred = final_model.predict(X_test.drop(axis=1, columns=["group"]))

    # Convert probabilities to class labels (binary classification)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    used_probability = np.where(y_pred_binary == 1, y_pred, 1 - y_pred)
    not_probability = 1 - used_probability
    X_test_["used_probability"] = used_probability
    X_test_["not_probability"] = not_probability
    X_test_["entropy"] = -1

    for index, row in X_test_.iterrows():
        entropy_row = entropy([row["used_probability"], row["not_probability"]], base=2)  # Compute Shannon entropy
        X_test_.loc[index, 'entropy'] = entropy_row

    # Calculate classification metrics
    accuracy = balanced_accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')

    data = {"acc": accuracy,
            "pre": precision,
           "rec": recall,
        "roc_auc": roc_auc,
            "f1": f1,
            "acc_baseline": accuracy_baseline,
            "pre_baseline": precision_baseline,
            "rec_baseline": recall_baseline,
            "roc_auc_baseline": roc_baseline,
            "f1_baseline": f1_baseline
            }
    data_list = [data]

    time_dat = pd.DataFrame(data_list)

    if not os.path.isfile(os.path.join(os.pardir, "data/processed/features/bs_features",
                                       f"classifier_metrics{cutoff}_bacc_500.csv")):
        time_dat.to_csv(os.path.join(os.path.join(os.pardir, "data/processed/features/bs_features",
                                                  f"classifier_metrics{cutoff}_bacc_500.csv")), index=False)
    else:
        time_dat.to_csv(os.path.join(os.pardir, "data/processed/features/bs_features",
                                     f"classifier_metrics{cutoff}_bacc_500.csv"),
                        index=False,
                        mode='a', header=False)

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
    X_test_["prediction_binary"] = y_pred_binary
    X_test_["support"] = y_test
    X_test_.to_csv(os.path.join(os.pardir, "data/prediction", f"prediction_results_classifier{cutoff}" + name + ".csv"))

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

for cutoff in [0.8]:
    for i in range(0,10):
        light_gbm_regressor(cutoff, rfe=False, rfe_feature_n=10, shapley_calc=False)