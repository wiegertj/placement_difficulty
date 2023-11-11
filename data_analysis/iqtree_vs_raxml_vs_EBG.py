import math
import os
import sys
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/EBG_TEST_HALF"
all_dataframes = []

counter = 0
for root, dirs, files in os.walk(file_path):
    # Skip the "ebg_tmp" directory and its contents
    if "ebg_tmp" in dirs:
        dirs.remove("ebg_tmp")
    for filename in files:
        if filename.endswith('.csv'):
            counter += 1
            print(counter)
            file_pathname = os.path.join(root, filename)
            filename_data = filename.replace("ebg_test", "")
            df = pd.read_csv(file_pathname)
            df["dataset"] = filename_data
            all_dataframes.append(df)
combined_dataframe = pd.concat(all_dataframes, ignore_index=True)

combined_dataframe.to_csv("ebg_prediction_test_half.csv")
print(combined_dataframe.shape)
sys.exit()

#################################### TIME ####################################
from skimage.metrics import mean_squared_error

#from data_analysis.analyse_pi import MBE
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
time_iq_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/bootstrap_times_iqtree_mt.csv"))
time_ebg_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/benchmark_ebg.csv"))
time_raxml_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/inference_times_standard_20.csv"))
time_rbs = pd.read_csv(os.path.join(os.pardir, "tests/final_times/benchmark_rapid_bootstrap_deimos.csv"))
boot = pd.read_csv(os.path.join(os.pardir, "tests/final_times/bootstrap_times_standard.csv"))

time_merged = time_iq_tree.merge(time_ebg_tree, on=["dataset"], how="inner")
#time_merged = time_merged.merge(time_rbs, on=["dataset"], how="inner")
time_merged = time_merged.merge(time_raxml_tree, on=["dataset"], how="inner")
time_merged = time_merged.merge(boot, on=["dataset"], how="inner")
time_merged["msa_size"] = time_merged["len_x"] * time_merged["num_seq_x"]
time_merged["elapsed_time_ebg_inf"] = time_merged["elapsed_time_ebg"] + time_merged["elapsed_time_inference"]
time_merged['isDNA'] = [1] * 180 + [0] * (len(time_merged) - 180)
time_merged_rbs = time_merged.merge(time_rbs,  on=["dataset"], how="inner")

print(time_merged.shape)
count_smaller = len(time_merged[time_merged['elapsed_time_ebg_inf'] < time_merged['elapsed_time_iqtree']])
print(count_smaller)
# Calculate the percentage
percentage_smaller = (count_smaller / len(time_merged))

# Print the result
print(f"The percentage of rows where elapsed_time_ebg_inf is smaller than elapsed_time_iqtree: {percentage_smaller}%")
time_merged["diff_iqebg"] = (time_merged['elapsed_time_iqtree'] - time_merged['elapsed_time_ebg_inf'])
time_merged["perc_diff"] = time_merged["diff_iqebg"] / time_merged['elapsed_time_iqtree']
time_merged_speedup = time_merged[time_merged["msa_size"] >= 0]
print(time_merged_speedup.shape)
time_merged["speedup_to_iq"] = (time_merged['elapsed_time_iqtree'] / time_merged['elapsed_time_ebg_inf'])
print(time_merged[["speedup_to_iq", "elapsed_time_iqtree", "elapsed_time_ebg_inf"]] )
print(time_merged["speedup_to_iq"].median())

time_merged_speedup["speedup_to_iq"] = (time_merged_speedup['elapsed_time_iqtree'] / time_merged_speedup['elapsed_time_ebg_inf'])
print(time_merged_speedup[["speedup_to_iq", "elapsed_time_iqtree", "elapsed_time_ebg_inf"]] )
print(time_merged_speedup["speedup_to_iq"].median())
import numpy as np
time_merged["perc_diff"] = time_merged["perc_diff"].replace([np.inf, -np.inf], np.nan)

time_merged = time_merged.dropna(subset=['perc_diff'])

print(time_merged["perc_diff"].values.tolist())
percentage_difference = time_merged["perc_diff"].median()
# Print the result
print(f"On average, 'elapsed_time_ebg_inf' is smaller by {percentage_difference}% compared to 'elapsed_time_iqtree'.")
time_merged['elapsed_time_sbs'] = time_merged['elapsed_time_sbs'] * 1.1
sum_iq = time_merged['elapsed_time_iqtree'].sum()
sum_boot = time_merged['elapsed_time_sbs'].sum()
sum_ebg = time_merged['elapsed_time_ebg'].sum()
#sum_rbs = time_merged['elapsed_time_rbs'].sum()
sum_rebg_inf = time_merged['elapsed_time_ebg_inf'].sum()

columns = ['Prediction', 'Prediction + Inference','IQTree UFB' , "SBS"]
sums = [sum_ebg,sum_rebg_inf, sum_iq , sum_boot]
plt.bar(columns, sums)
plt.xlabel('Methods')
plt.ylabel('Total Elapsed Time (seconds)')
plt.title('Total Elapsed Time for Different Methods')
plt.tight_layout()
plt.show()
sorted_time_merged = time_merged.sort_values(by='msa_size')

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Sort the DataFrame by 'msa_size'

window_size = 20  # Adjust this value as needed
time_merged_rbs = time_merged_rbs.sort_values(by='msa_size')
time_merged_rbs['moving_avg_ebg_rbs_1'] = time_merged_rbs[time_merged_rbs['isDNA'] == 1]['elapsed_time_rbs'].rolling(window=window_size).median()
time_merged_rbs['moving_avg_ebg_rbs_0'] = time_merged_rbs[time_merged_rbs['isDNA'] == 0]['elapsed_time_rbs'].rolling(window=5).median()

# Calculate the moving average for each data series when 'isDNA' == 1
sorted_time_merged['moving_avg_iqtree_1'] = sorted_time_merged[sorted_time_merged['isDNA'] == 1]['elapsed_time_iqtree'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg'] = sorted_time_merged['elapsed_time_ebg'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_sbs'] = sorted_time_merged['elapsed_time_sbs'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_inf_1'] = sorted_time_merged[sorted_time_merged['isDNA'] == 1]['elapsed_time_ebg_inf'].rolling(window=window_size).median()

sorted_time_merged['moving_avg_iqtree_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0]['elapsed_time_iqtree'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0]['elapsed_time_ebg'].rolling(window=window_size).median()
sorted_time_merged['moving_avg_ebg_inf_0'] = sorted_time_merged[sorted_time_merged['isDNA'] == 0]['elapsed_time_ebg_inf'].rolling(window=window_size).median()

# Plot the smoothed lines using the moving average for 'isDNA' == 1
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size'], sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_iqtree_1'], label='UFBoot2 (DNA)', color='blue')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size'], sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_iqtree_0'], label='UFBoot2 (AA)', color='blue', linestyle="dashed")

#plt.plot(sorted_time_merged['moving_avg_sbs'], label='SBS', color='red')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 1]['msa_size'], sorted_time_merged[sorted_time_merged['isDNA'] == 1]['moving_avg_ebg_inf_1'], label='EBG + inference (DNA)', color='green')
plt.plot(sorted_time_merged[sorted_time_merged['isDNA'] == 0]['msa_size'], sorted_time_merged[sorted_time_merged['isDNA'] == 0]['moving_avg_ebg_inf_0'], label='EBG + inference (AA)', color='green', linestyle="dashed")
plt.plot(sorted_time_merged["msa_size"],sorted_time_merged["moving_avg_ebg"], label='EBG', color='green', linestyle="dotted")

plt.plot(sorted_time_merged["msa_size"],sorted_time_merged["moving_avg_sbs"], label='SBS', color='red')

# Plot the smoothed lines using the moving average for 'isDNA' == 1
#plt.plot(time_merged_rbs[time_merged_rbs['isDNA'] == 1]['msa_size'], time_merged_rbs[time_merged_rbs['isDNA'] == 1]['moving_avg_ebg_rbs_1'], label='RBS (DNA)', color='purple')
#plt.plot(time_merged_rbs[time_merged_rbs['isDNA'] == 0]['msa_size'], time_merged_rbs[time_merged_rbs['isDNA'] == 0]['moving_avg_ebg_rbs_0'], label='RBS (AA)', color='purple', linestyle="dashed")


# Add labels, legend, and other plot details
plt.xlabel('# sequences × sequence length')
plt.yscale('log', base=10)  # Set the base to 10 for powers of 10
plt.xscale('log', base=10)
plt.ylabel('Moving average elapsed time (window size = 20)')
plt.legend()
plt.tight_layout()
plt.savefig("runtimes.png")
plt.show()







from numpy import median
from sklearn.metrics import mean_absolute_error, median_absolute_error, accuracy_score, f1_score, roc_auc_score

df_iqtree = pd.read_csv(os.path.join(os.pardir, "tests/final_preds/iq_boots.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "tests/final_preds/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
#df_normal = pd.read_csv(os.path.join(os.pardir, "data/branch_supports.csv"))
#df_raxml = pd.read_csv(os.path.join(os.pardir, "data/raxml_classic_supports.csv"))
################## ################## ################## ################## ################## ##################

df_raxml = pd.read_csv(os.path.join(os.pardir, "data/raxml_classic_supports.csv"))
df_merged_raxml = df_raxml.merge(df_ebg_prediction, on=["dataset", "branchId"], how="inner")
df_merged_raxml = df_merged_raxml.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"], how="inner")

df_merged_raxml["bound_dist_5"] = abs(df_merged_raxml["prediction_lower5"] - df_merged_raxml["prediction_median"])
print(df_merged_raxml.shape)
df_merged_raxml = df_merged_raxml[df_merged_raxml["bound_dist_5"] <= 22]
print(df_merged_raxml.shape)

print("EBG")
mae = mean_absolute_error(df_merged_raxml["true_support"], df_merged_raxml["prediction_median"])
mdae = median_absolute_error(df_merged_raxml["true_support"], df_merged_raxml["prediction_median"])
mbe = MBE(df_merged_raxml["true_support"], df_merged_raxml["prediction_median"])
rmse = math.sqrt(mean_squared_error(df_merged_raxml["true_support"], df_merged_raxml["prediction_median"]))
print("MAE " + str(mae))
print("mdae " + str(mdae))
print("mbe " + str(mbe))
print("rmse " + str(rmse))
print("RB")
mae = mean_absolute_error(df_merged_raxml["true_support"], df_merged_raxml["support_raxml_classic"])
mdae = median_absolute_error(df_merged_raxml["true_support"], df_merged_raxml["support_raxml_classic"])
mbe = MBE(df_merged_raxml["true_support"], df_merged_raxml["support_raxml_classic"])
rmse = math.sqrt(mean_squared_error(df_merged_raxml["true_support"], df_merged_raxml["support_raxml_classic"]))
print("+++"*50)
print("MAE " + str(mae))
print("mdae " + str(mdae))
print("mbe " + str(mbe))
print("rmse " + str(rmse))

df_merged_raxml['ebg_over_80'] = (df_merged_raxml['prediction_bs_over_80'] >= 0.5).astype(int)
df_merged_raxml['ebg_over_70'] = (df_merged_raxml['prediction_bs_over_70'] >= 0.5).astype(int)
df_merged_raxml['ebg_over_75'] = (df_merged_raxml['prediction_bs_over_75'] >= 0.5).astype(int)
df_merged_raxml['ebg_over_85'] = (df_merged_raxml['prediction_bs_over_85'] >= 0.5).astype(int)

df_merged_raxml['rb_over_80'] = (df_merged_raxml['support_raxml_classic'] >= 80).astype(int)
df_merged_raxml['rb_over_70'] = (df_merged_raxml['support_raxml_classic'] >= 70).astype(int)
df_merged_raxml['rb_over_75'] = (df_merged_raxml['support_raxml_classic'] >= 75).astype(int)
df_merged_raxml['rb_over_85'] = (df_merged_raxml['support_raxml_classic'] >= 85).astype(int)

df_merged_raxml['support_over_85'] = (df_merged_raxml['true_support'] >= 85).astype(int)
df_merged_raxml['support_over_80'] = (df_merged_raxml['true_support'] >= 80).astype(int)
df_merged_raxml['support_over_75'] = (df_merged_raxml['true_support'] >= 75).astype(int)
df_merged_raxml['support_over_70'] = (df_merged_raxml['true_support'] >= 70).astype(int)

print("---------80-----------")
print("acc")
accuracy = accuracy_score(df_merged_raxml["support_over_80"], df_merged_raxml["ebg_over_80"])
print(accuracy)
accuracy = accuracy_score(df_merged_raxml["support_over_80"], df_merged_raxml["rb_over_80"])
print(accuracy)
print("f1")
f1 = f1_score(df_merged_raxml["support_over_80"], df_merged_raxml["ebg_over_80"])
print(f1)
f1 = f1_score(df_merged_raxml["support_over_80"], df_merged_raxml["rb_over_80"])
print(f1)
print("roc")
roc = roc_auc_score(df_merged_raxml["support_over_80"], df_merged_raxml["ebg_over_80"])
print(roc)
roc = roc_auc_score(df_merged_raxml["support_over_80"], df_merged_raxml["rb_over_80"])
print(roc)#
print("---------70-----------")
print("acc")
accuracy = accuracy_score(df_merged_raxml["support_over_70"], df_merged_raxml["ebg_over_70"])
print(accuracy)
accuracy = accuracy_score(df_merged_raxml["support_over_70"], df_merged_raxml["rb_over_70"])
print(accuracy)
print("f1")
f1 = f1_score(df_merged_raxml["support_over_70"], df_merged_raxml["ebg_over_70"])
print(f1)
f1 = f1_score(df_merged_raxml["support_over_70"], df_merged_raxml["rb_over_70"])
print(f1)
print("roc")
roc = roc_auc_score(df_merged_raxml["support_over_70"], df_merged_raxml["ebg_over_70"])
print(roc)
roc = roc_auc_score(df_merged_raxml["support_over_70"], df_merged_raxml["rb_over_70"])
print(roc)
print("---------85-----------")
print("acc")
accuracy = accuracy_score(df_merged_raxml["support_over_85"], df_merged_raxml["ebg_over_85"])
print(accuracy)
accuracy = accuracy_score(df_merged_raxml["support_over_85"], df_merged_raxml["rb_over_85"])
print(accuracy)
print("f1")
f1 = f1_score(df_merged_raxml["support_over_85"], df_merged_raxml["ebg_over_85"])
print(f1)
f1 = f1_score(df_merged_raxml["support_over_85"], df_merged_raxml["rb_over_85"])
print(f1)
print("roc")
roc = roc_auc_score(df_merged_raxml["support_over_85"], df_merged_raxml["ebg_over_85"])
print(roc)
roc = roc_auc_score(df_merged_raxml["support_over_85"], df_merged_raxml["rb_over_85"])
print(roc)
print("---------75-----------")
print("acc")
accuracy = accuracy_score(df_merged_raxml["support_over_75"], df_merged_raxml["ebg_over_75"])
print(accuracy)
accuracy = accuracy_score(df_merged_raxml["support_over_75"], df_merged_raxml["rb_over_75"])
print(accuracy)
print("f1")
f1 = f1_score(df_merged_raxml["support_over_75"], df_merged_raxml["ebg_over_75"])
print(f1)
f1 = f1_score(df_merged_raxml["support_over_75"], df_merged_raxml["rb_over_75"])
print(f1)
print("roc")
roc = roc_auc_score(df_merged_raxml["support_over_75"], df_merged_raxml["ebg_over_75"])
print(roc)
roc = roc_auc_score(df_merged_raxml["support_over_75"], df_merged_raxml["rb_over_75"])
print(roc)

print("+++"*50)
################## ################## ################## ################## ################## ##################



df_merged = df_ebg_prediction.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"], how="inner")
df_merged["iq_tree_error"] = df_merged["true_support"].values - (df_merged["iq_support"].values)
#df_merged["raxml_error"] = df_merged["support"].values*100 - df_merged["support_raxml_classic"].values


#df_merged = df_ebg_prediction.merge(df_normal, on=["branchId", "dataset"], how="inner")
#df_merged = df_merged.merge(df_raxml, on=["dataset", "branchId"], how="inner")
df_merged["bound_dist_10"] = abs(df_merged["prediction_lower10"] - df_merged["prediction_median"])
df_merged["bound_dist_5"] = abs(df_merged["prediction_lower5"] - df_merged["prediction_median"])

df_merged["pred_error_ebg"] = abs(df_merged["true_support"].values - df_merged["prediction_ebg_tool"])

import matplotlib.pyplot as plt
import numpy as np



# Assuming you have a DataFrame named 'df_merged'
bound_dist_10 = df_merged["bound_dist_10"]
bound_dist_5 = df_merged["bound_dist_5"]
df_merged["pred_error_ebg"] = pow(df_merged["true_support"] - df_merged["prediction_ebg_tool"],2)
pred_error_ebg = df_merged["pred_error_ebg"]

print("+++++++NEW")
print(df_merged.shape)
print(df_merged[df_merged["bound_dist_5"] <= 22].shape)
print("+++++++NEW")

# Define the interval bins
bins = np.arange(0, 41, 1)

# Use the `pd.cut` function to create interval categories for 'bound_dist_10' and 'bound_dist_5'
interval_categories_10 = pd.cut(bound_dist_10, bins)
interval_categories_5 = pd.cut(bound_dist_5, bins)

# Group the data by intervals and calculate the mean 'pred_error_ebg' in each interval
no_group10 = pred_error_ebg.groupby(interval_categories_10).count()
print(no_group10)
no_group5 = pred_error_ebg.groupby(interval_categories_5).count()
grouped_10 = pred_error_ebg.groupby(interval_categories_10).sum() / no_group10
print(grouped_10)
grouped_5 = pred_error_ebg.groupby(interval_categories_5).sum() / no_group5
grouped_10 = np.sqrt(grouped_10)
grouped_5 = np.sqrt(grouped_5)
print(grouped_10)

# Create the bar chart with two sets of bars
plt.figure(figsize=(10, 6))

# Bar width for each set of bars
bar_width = 0.35

# Calculate the x-axis positions for the bars
x = np.arange(len(grouped_10))

# Plot the bars for 'bound_dist_10'
plt.bar(x - bar_width/2, grouped_10.values, width=bar_width, label='10% lower bound')
#plt.xticks([i + bar_width / 2 for i in x], np.arange(1, 41))

# Plot the bars for 'bound_dist_5'
plt.bar(x + bar_width/2, grouped_5.values, width=bar_width, label='5% lower bound')

# Your existing code
plt.xlabel('Distance to median prediction', fontsize=14)  # Set the fontsize for x-label
plt.ylabel('Median absolute error', fontsize=14)  # Set the fontsize for y-label
plt.grid(axis='y')
plt.xticks(rotation=45, ha='right', fontsize=12)  # Set the fontsize for x-ticks

plt.yticks(fontsize=12)  # Set the fontsize for y-ticks
plt.legend(fontsize=12)  # Set the fontsize for the legend
plt.tight_layout()

# Increase the size of the overall figure for better readability
fig = plt.gcf()
fig.set_size_inches(10, 6)  # Adjust the figure size as needed

plt.savefig("uncertainty_regression")
plt.show()



plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.hist(df_merged["bound_dist_10"], bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.title('Histogram of bound_dist_10')
plt.xlabel('bound_dist_10')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.scatter(df_merged["bound_dist_10"], df_merged["pred_error_ebg"], alpha=0.05)  # Adjust marker size and alpha (transparency) as needed
plt.title('Scatterplot of dist_lower_10 vs. pred_error')
plt.xlabel('dist_lower_10')
plt.ylabel('pred_error')
plt.grid(True)
plt.show()

df_merged["pred_error_ebg_abs"] = abs(df_merged["true_support"].values - df_merged["prediction_ebg_tool"])
df_merged["iq_tree_error_abs"] = abs(df_merged["true_support"].values - (df_merged["iq_support"].values - 15))
#df_merged["raxml_error_abs"] = abs(df_merged["support"].values*100 - df_merged["support_raxml_classic"].values)

print(df_merged.shape)

print("########################"*5)
print("IQTREE Error")
print(mean(df_merged["iq_tree_error_abs"]))
print(median(df_merged["iq_tree_error_abs"]))
print("########################"*5)
#print("Prediction Error RAxML")
#print(mean(df_merged["raxml_error_abs"]))
#print(median(df_merged["raxml_error_abs"]))
print("########################"*5)
print("Prediction EBG")
print(mean(df_merged["pred_error_ebg_abs"]))
print(median(df_merged["pred_error_ebg_abs"]))
print(math.sqrt(mean_squared_error(df_merged["true_support"], df_merged["prediction_ebg_tool"])))
print("########################"*5)

within_range_5 = df_merged[(df_merged['true_support'] >= df_merged['prediction_lower5'])]
percentage_within_range = (len(within_range_5) / len(df_merged)) * 100
print("Within range 5: "+str(percentage_within_range))

within_range_10 = df_merged[(df_merged['true_support'] >= df_merged['prediction_lower10'])]
percentage_within_range = (len(within_range_10) / len(df_merged)) * 100
print("Within range 10: "+str(percentage_within_range))

print("########################"*5)

df_merged['ebg_over_80'] = (df_merged['prediction_bs_over_80'] >= 0.5).astype(int)
df_merged['ebg_over_70'] = (df_merged['prediction_bs_over_70'] >= 0.5).astype(int)
df_merged['ebg_over_75'] = (df_merged['prediction_bs_over_75'] >= 0.5).astype(int)
df_merged['ebg_over_85'] = (df_merged['prediction_bs_over_85'] >= 0.5).astype(int)

df_merged['support_over_85'] = (df_merged['true_support'] >= 85).astype(int)
df_merged['support_over_80'] = (df_merged['true_support'] >= 80).astype(int)
df_merged['support_over_75'] = (df_merged['true_support'] >= 75).astype(int)
df_merged['support_over_70'] = (df_merged['true_support'] >= 70).astype(int)

from scipy.stats import entropy

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_80"], 1 - row["prediction_bs_over_80"]]), base=2)  # Compute Shannon entropy
    df_merged.loc[index, 'entropy_ebg_over_80'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_85"], 1 - row["prediction_bs_over_85"]]), base=2)  # Compute Shannon entropy
    df_merged.loc[index, 'entropy_ebg_over_85'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_75"], 1 - row["prediction_bs_over_75"]]), base=2)  # Compute Shannon entropy
    df_merged.loc[index, 'entropy_ebg_over_75'] = entropy_row

for index, row in df_merged.iterrows():
    entropy_row = entropy(np.array([row["prediction_bs_over_70"], 1 - row["prediction_bs_over_70"]]), base=2)  # Compute Shannon entropy
    df_merged.loc[index, 'entropy_ebg_over_70'] = entropy_row


import numpy as np
import matplotlib.pyplot as plt

intervals = np.arange(0, 1.1, 0.1)
fractions_80 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_80'] >= lower_bound) & (df_merged['entropy_ebg_over_80'] < upper_bound)
    fraction = np.mean(df_merged['support_over_80'][mask] == df_merged['ebg_over_80'][mask])
    fractions_80.append(fraction)

fractions_85 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_85'] >= lower_bound) & (df_merged['entropy_ebg_over_85'] < upper_bound)
    fraction = np.mean(df_merged['support_over_85'][mask] == df_merged['ebg_over_85'][mask])
    fractions_85.append(fraction)

fractions_75 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_75'] >= lower_bound) & (df_merged['entropy_ebg_over_75'] < upper_bound)
    fraction = np.mean(df_merged['support_over_75'][mask] == df_merged['ebg_over_75'][mask])
    fractions_75.append(fraction)

fractions_70 = []
for i in range(len(intervals) - 1):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    mask = (df_merged['entropy_ebg_over_70'] >= lower_bound) & (df_merged['entropy_ebg_over_70'] < upper_bound)
    fraction = np.mean(df_merged['support_over_70'][mask] == df_merged['ebg_over_70'][mask])
    fractions_70.append(fraction)
print(fractions_70)
# Set the width of the bars
bar_width = 0.2
uncertainty_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Set the width of the bars
bar_width = 0.2

# Create x-axis values for each set of bars
x = np.arange(len(uncertainty_levels))

# Create the bar chart
plt.bar(x - 1.5*bar_width, fractions_70, width=bar_width, label='t=0.70')
plt.bar(x - 0.5*bar_width, fractions_75, width=bar_width, label='t=0.75')
plt.bar(x + 0.5*bar_width, fractions_80, width=bar_width, label='t=0.80')
plt.bar(x + 1.5*bar_width, fractions_85, width=bar_width, label='t=0.85')
plt.axhline(y=0.8, color='black', linestyle='--', label='0.8')
plt.axhline(y=0.9, color='blue', linestyle='--', label='0.9')
# Set the x-axis labels
plt.xlabel('Uncertainty')
plt.xticks(x, [f"{u:.1f}" for u in uncertainty_levels])

# Set the y-axis label
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))

# Set the title
plt.legend()

plt.tight_layout()
plt.savefig("uncertainty")
# Add a legend
plt.show()
# Show the

df_merged['iq_support_over_95'] = (df_merged['iq_support'] >= 95).astype(int)
print(df_merged.shape)
print(df_merged.shape)
print("acc")
accuracy = accuracy_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
print(accuracy)
accuracy = accuracy_score(df_merged["support_over_80"], df_merged["iq_support_over_95"])
print(accuracy)
print("f1")
f1 = f1_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
print(f1)
f1 = f1_score(df_merged["support_over_80"], df_merged["iq_support_over_95"])
print(f1)
print("roc")
roc = roc_auc_score(df_merged["support_over_80"], df_merged["ebg_over_80"])
print(roc)
roc = roc_auc_score(df_merged["support_over_80"], df_merged["iq_support_over_95"])
print(roc)
print("Prediction EBG")
print(mean(df_merged["pred_error_ebg_abs"]))
print(median(df_merged["pred_error_ebg_abs"]))
print(math.sqrt(mean_squared_error(df_merged["true_support"], df_merged["prediction_ebg_tool"])))
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, label, ax):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

# Create a single figure and axis for both ROC curves
fig, ax = plt.subplots()

# Plot ROC curve for the "EBG" accuracy score
plot_roc_curve(df_merged["support_over_80"], df_merged["ebg_over_80"], "Prediction", ax)

# Plot ROC curve for the "IQ Support" accuracy score
plot_roc_curve(df_merged["support_over_80"], df_merged["iq_support_over_95"], "IQ Support", ax)

# Customize the plot
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc='lower right')

# Display the plot
plt.show()
# Add labels and a legend
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Show the plot
plt.show()

################## ################## ################## ################## ################## ##################
