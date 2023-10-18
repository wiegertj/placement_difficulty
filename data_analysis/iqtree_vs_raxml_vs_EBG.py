import os
import sys
from statistics import mean
import matplotlib.pyplot as plt

import pandas as pd




file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/tests"
all_dataframes = []

counter = 0
for root, dirs, files in os.walk(file_path):
 #   # Skip the "ebg_tmp" directory and its contents
    if "ebg_tmp" in dirs:
        dirs.remove("ebg_tmp")
    for filename in files:
        if filename.endswith('.csv'):
            counter += 1
            print(counter)
            file_pathname = os.path.join(root, filename)
            df = pd.read_csv(file_pathname)
            all_dataframes.append(df)

combined_dataframe = pd.concat(all_dataframes, ignore_index=True)

combined_dataframe.to_csv("ebg_prediction_test.csv")
print(combined_dataframe.shape)
sys.exit()

time_iq_tree = pd.read_csv(os.path.join(os.pardir, "data/pars_boot_times_iqrtree.csv"))
time_ebg_tree = pd.read_csv(os.path.join(os.pardir, "data/pars_boot_times_ebg.csv"))
time_raxml_tree = pd.read_csv(os.path.join(os.pardir, "data/pars_boot_times_raxml_classic.csv"))

time_merged = time_iq_tree.merge(time_ebg_tree, on=["num_seq", "len"], how="inner")
time_merged = time_merged.merge(time_raxml_tree, on=["num_seq", "len"], how="inner")

print(time_merged.shape)

sum_iq = time_merged['elapsed_time_iq'].sum()
sum_ebg = time_merged['elapsed_time_ebg'].sum()
sum_raxml = time_merged['elapsed_time_raxml'].sum() / 2
print("1:" + str(sum_iq/sum_ebg) + ":" + str(sum_raxml/sum_ebg))

# Create a bar chart for the sums
columns = ['Prediction', 'IQTree UFB', 'RAxML Rapid Bootstrap']
sums = [sum_ebg, sum_iq , sum_raxml]

plt.bar(columns, sums)

plt.xlabel('Methods')
plt.ylabel('Total Elapsed Time (seconds)')
plt.title('Total Elapsed Time for Different Methods')

# Show the plot
plt.show()

# Create a line chart to show the development of the three times over time (len)
plt.plot(sorted(time_merged['len']), sorted(time_merged['elapsed_time_iq']), label='IQTree')
plt.plot(sorted(time_merged['len']), sorted(time_merged['elapsed_time_ebg']), label='Prediction')
plt.plot(sorted(time_merged['len']), sorted(time_merged['elapsed_time_raxml']), label='RAxML Rapid BS')

plt.xlabel('Sequence Length')
plt.ylabel('Elapsed Time (seconds)')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Create a line chart to show the development of the three times over time (len)
#plt.plot(sorted(time_merged['num_seq']), sorted(time_merged['elapsed_time_iq']), label='IQTree')
plt.plot(sorted(time_merged['num_seq']), sorted(time_merged['elapsed_time_ebg']), label='Prediction')
#plt.plot(sorted(time_merged['num_seq']), sorted(time_merged['elapsed_time_raxml']), label='RAxML Rapid BS')

plt.xlabel('Number of Sequences')
plt.ylabel('Elapsed Time (seconds)')

# Add a legend
plt.legend()

# Show the plot
plt.show()


from numpy import median
from sklearn.metrics import mean_absolute_error, median_absolute_error

df_iqtree = pd.read_csv(os.path.join(os.pardir, "data/iq_boots.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "data/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace("ebg_test", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
df_normal = pd.read_csv(os.path.join(os.pardir, "data/pred_interval_75_final.csv"))
df_normal["prediction_normal"] = df_normal["prediction_median"] * 100
df_raxml = pd.read_csv(os.path.join(os.pardir, "data/raxml_classic_supports.csv"))

df_merged = df_ebg_prediction.merge(df_normal, on=["branchId", "dataset"], how="inner")
df_merged = df_merged.merge(df_raxml, on=["dataset", "branchId"], how="inner")
df_merged = df_merged.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"], how="inner")

df_merged["error_from_ebg"] = (df_merged["prediction_normal"] - df_merged["prediction_ebg_tool"])
print(mean(df_merged["error_from_ebg"]))
df_merged["pred_error_ebg"] = (df_merged["support"]*100 - df_merged["prediction_ebg_tool"])
print(mean(df_merged["pred_error_ebg"]))

print("########################"*5)

print(df_merged.shape)
df_merged["pred_error_raxml"] = (df_merged["support"]*100 - df_merged["support_raxml_classic"])
print(mean(df_merged["pred_error_raxml"]))

df_merged["iq_tree_error"] = df_merged["support"].values*100 - df_merged["iq_support"].values - 15

print("########################"*5)

print("Prediction Error RAxML")
print("MAE")
print(mean_absolute_error(df_merged["support"].values*100, df_merged["support_raxml_classic"].values))
print("MAE")
print(median_absolute_error(df_merged["support"].values*100, df_merged["support_raxml_classic"].values))

print("########################"*5)

print("Prediction EBG")
print("MAE")
print(mean_absolute_error(df_merged["support"].values*100, df_merged["prediction_ebg_tool"].values))
print("MAE")
print(median_absolute_error(df_merged["support"].values*100, df_merged["prediction_ebg_tool"].values))
print("in 75%")
within_range = df_merged[(df_merged['support']*100 >= df_merged['prediction_lower_75']) & (df_merged['support']*100 <= df_merged['prediction_upper_75'])]
percentage_within_range = (len(within_range) / len(df_merged)) * 100
print(percentage_within_range)
df_merged["pi_width_75"] = abs(df_merged['prediction_lower_75']-df_merged['prediction_upper_75'])
data_to_plot = df_merged['prediction_upper_75']
print("Prediction Interval")
print(mean(df_merged["pi_width_75"]))
# Create a histogram
plt.hist(data_to_plot, bins=8, color='skyblue', edgecolor='black')

# Add labels and a title
plt.xlabel('Lower Bound')
plt.ylabel('Frequency')
plt.title('Histogram of Lower Bound 75')

# Show the plot
plt.show()
print("########################"*5)

print("Prediction IQTree")
print("MAE")
print(mean_absolute_error(df_merged["support"].values*100, df_merged["iq_support"].values - 15))
print("MAE")
print(median_absolute_error(df_merged["support"].values*100, df_merged["iq_support"].values - 15))

# Plot histograms for both columns on the same plot
plt.hist(df_merged["pred_error_raxml"], alpha=0.5, label='RAxML Rapid Bootstrap', bins=30)
plt.hist(df_merged["pred_error_ebg"], alpha=0.5, label='Prediction', bins=30)
plt.hist(df_merged["iq_tree_error"], alpha=0.5, label='IQTree', bins=30)


# Add labels and a legend
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Show the plot
plt.show()