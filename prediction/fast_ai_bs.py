import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
from pytorch_forecasting.metrics.quantile import QuantileLoss
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

loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
loo_selection = loo_selection[:200]
filenames = loo_selection["dataset"].values.tolist()

test = df[df['dataset'].isin(filenames)]
train = df[~df['dataset'].isin(filenames)]


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


# print(test.shape)
# print(train.shape)

#####

# sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
# test = df[df['group'].isin(sample_dfs)]
# train = df[~df['group'].isin(sample_dfs)]

# print(test.shape)
# print(train.shape)

#####

# sample_dfs = random.sample(df["group"].unique().tolist(), int(len(df["group"].unique().tolist()) * 0.2))
# test = df[df['group'].isin(sample_dfs)]
# train = df[~df['group'].isin(sample_dfs)]
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=["support"], axis=1), train["support"], test_size=0.2, random_state=42)

X_train = X_train.drop(axis=1, columns=["dataset", "branchId", "group"]).to_numpy()
X_val = X_val.drop(axis=1, columns=["dataset", "branchId", "group"]).to_numpy()
X_test = test.drop(axis=1, columns=[target, "dataset", "branchId", "group"]).to_numpy()
y_test = test[target].to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
scaler = MinMaxScaler()

X_test = scaler.fit_transform(X_test)
scaler = MinMaxScaler()

X_val = scaler.fit_transform(X_val)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)



# Define the model
model = nn.Sequential(
    nn.Linear(22, 80),
    nn.ReLU(),
    nn.Linear(80, 50),
    nn.ReLU(),
    nn.Linear(50, 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 1)
)
quantiles = [0.125]
loss_fn = QuantileLoss(quantiles=quantiles)

# loss function and optimizer
#loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.1)

n_epochs = 200   # number of epochs to run
batch_size = 25  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
patience = 10  # Number of epochs with no improvement to wait before early stopping
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Decrease LR by half every 10 epochs

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch

    scheduler.step()

    model.eval()
    y_pred = model(X_val)
    mse = loss_fn(y_pred, y_val)
    mse = float(mse)

    # Check if validation loss (MSE) has improved
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        count = 0  # Reset the patience counter
    else:
        count += 1

    if count >= patience:
        print(f"Early stopping after {epoch} epochs without improvement.")
        break

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
model.eval()
with torch.no_grad():
    y_pred_low = model(X_test)

test["prediction_low"] = y_pred_low










# Define the model
model = nn.Sequential(
    nn.Linear(22, 80),
    nn.ReLU(),
    nn.Linear(80, 50),
    nn.ReLU(),
    nn.Linear(50, 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 1)
)
quantiles = [0.875]
loss_fn = QuantileLoss(quantiles=quantiles)

# loss function and optimizer
#loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.1)

n_epochs = 200   # number of epochs to run
batch_size = 25  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
patience = 10  # Number of epochs with no improvement to wait before early stopping
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Decrease LR by half every 10 epochs

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch

    scheduler.step()

    model.eval()
    y_pred = model(X_val)
    mse = loss_fn(y_pred, y_val)
    mse = float(mse)

    # Check if validation loss (MSE) has improved
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        count = 0  # Reset the patience counter
    else:
        count += 1

    if count >= patience:
        print(f"Early stopping after {epoch} epochs without improvement.")
        break

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
model.eval()
with torch.no_grad():
    y_pred_high = model(X_test)

test["prediction_hi"] = y_pred_high










# Define the model
model = nn.Sequential(
    nn.Linear(22, 80),
    nn.ReLU(),
    nn.Linear(80, 50),
    nn.ReLU(),
    nn.Linear(50, 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 1)
)
quantiles = [0.5]
loss_fn = QuantileLoss(quantiles=quantiles)

# loss function and optimizer
#loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.1)

n_epochs = 200   # number of epochs to run
batch_size = 25  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
patience = 10  # Number of epochs with no improvement to wait before early stopping
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Decrease LR by half every 10 epochs

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch

    scheduler.step()

    model.eval()
    y_pred = model(X_val)
    mse = loss_fn(y_pred, y_val)
    mse = float(mse)

    # Check if validation loss (MSE) has improved
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        count = 0  # Reset the patience counter
    else:
        count += 1

    if count >= patience:
        print(f"Early stopping after {epoch} epochs without improvement.")
        break

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
model.eval()
with torch.no_grad():
    y_pred_median = model(X_test)

test["prediction_median"] = y_pred_median










test.to_csv("pytorch_bs_pred.csv")