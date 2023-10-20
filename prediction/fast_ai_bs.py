import os
import numpy as np
import pandas as pd
from fastai.tabular.all import *
from sklearn.metrics import mean_absolute_error, mean_squared_error


def light_gbm_regressor():
    # Load your dataset (modify the path as needed)
    df = pd.read_csv(os.path.join(os.pardir, "data/processed/final", "bs_support.csv"))

    loo_selection = pd.read_csv(os.path.join(os.pardir, "data/loo_selection.csv"))
    loo_selection["dataset"] = loo_selection["verbose_name"].str.replace(".phy", "")
    loo_selection = loo_selection[:200]
    filenames = loo_selection["dataset"].values.tolist()

    test = df[df['dataset'].isin(filenames)]
    print(test)
    y_test = test["support"].values
    #test = test.drop(columns=["support"], axis=1)
    train = df[~df['dataset'].isin(filenames)]
    print(train)
    test = test.drop(columns=["dataset", "branchId"], axis=1)
    train = train.drop(columns=["dataset", "branchId"], axis=1)

    # Create a TabularPandas object
    procs = [Categorify, FillMissing, Normalize]
    cont_names = [col for col in train.columns if col != 'support']

    data = TabularPandas(train, procs=procs, cat_names=[], cont_names=cont_names, y_names="support")

    # Create DataLoaders
    dls = data.dataloaders(bs=64)  # Adjust batch size as needed

    # Create a regression learner with MAE as the loss function
    learn = tabular_learner(dls, layers=[200, 100], metrics=[mean_absolute_error])

    # Train the model
    learn.fit_one_cycle(5)  # Modify the number of epochs as needed

    procs = [Categorify, FillMissing, Normalize]
    cont_names = [col for col in train.columns if col != 'support']

    test_data = data.new(test)

    test_dataloader = learn.dls.test_dl(test_data)

    # Use the 'get_preds' method on the learner 'learn' to get predictions for the test data
    preds = learn.get_preds(dl=test_dataloader)[0]

    # Calculate MAE on the test set
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error on test set: {mae}")


# Call the function to run the Fastai Tabular Learner
light_gbm_regressor()
