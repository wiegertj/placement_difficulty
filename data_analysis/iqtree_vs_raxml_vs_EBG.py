import os
import sys
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd




file_path = "/hits/fast/cme/wiegerjs/placement_difficulty/BENCHMARK_EBG"
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

combined_dataframe.to_csv("ebg_prediction_test.csv")
print(combined_dataframe.shape)
sys.exit()

#################################### TIME ####################################

time_iq_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/bootstrap_times_iqtree_mt.csv"))
time_ebg_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/benchmark_ebg.csv"))
time_raxml_tree = pd.read_csv(os.path.join(os.pardir, "tests/final_times/inference_times_standard_20.csv"))
boot = pd.read_csv(os.path.join(os.pardir, "tests/final_times/bootstrap_times_standard.csv"))

time_merged = time_iq_tree.merge(time_ebg_tree, on=["dataset"], how="inner")
time_merged = time_merged.merge(time_raxml_tree, on=["dataset"], how="inner")
time_merged = time_merged.merge(boot, on=["dataset"], how="inner")
time_merged["msa_size"] = time_merged["len_x"] * time_merged["num_seq_x"]
time_merged["elapsed_time_ebg_inf"] = time_merged["elapsed_time_ebg"] + time_merged["elapsed_time_inference"]
print(time_merged.shape)


sum_iq = time_merged['elapsed_time_iqtree'].sum()
sum_boot = time_merged['elapsed_time_sbs'].sum()
sum_ebg = time_merged['elapsed_time_ebg'].sum()
sum_rebg_inf = time_merged['elapsed_time_ebg_inf'].sum()

columns = ['Prediction', 'Prediction + Inference','IQTree UFB' , "SBS"]
sums = [sum_ebg,sum_rebg_inf, sum_iq , sum_boot]
plt.bar(columns, sums)
plt.xlabel('Methods')
plt.ylabel('Total Elapsed Time (seconds)')
plt.title('Total Elapsed Time for Different Methods')
plt.tight_layout()
plt.show()

plt.plot(sorted(time_merged['msa_size']), sorted(time_merged['elapsed_time_iqtree']), label='IQTree')
plt.plot(sorted(time_merged['msa_size']), sorted(time_merged['elapsed_time_ebg']), label='Prediction')
plt.plot(sorted(time_merged['msa_size']), sorted(time_merged['elapsed_time_sbs']), label='SBS')
plt.plot(sorted(time_merged['msa_size']), sorted(time_merged['elapsed_time_ebg_inf']), label='Prediction + Inference')
plt.xlabel('Number of sequences * sequence length')
plt.yscale('log')
plt.ylabel('Elapsed Time (seconds)')
plt.legend()
plt.tight_layout()
plt.show()
################## ################## ################## ################## ################## ##################

from numpy import median
from sklearn.metrics import mean_absolute_error, median_absolute_error, accuracy_score, f1_score, roc_auc_score

df_iqtree = pd.read_csv(os.path.join(os.pardir, "tests/final_preds/iq_boots.csv"), usecols=lambda column: column != 'Unnamed: 0')
df_ebg_prediction = pd.read_csv(os.path.join(os.pardir, "tests/final_preds/ebg_prediction_test.csv"))
df_ebg_prediction["dataset"] = df_ebg_prediction["dataset"].str.replace(".csv", "")
df_ebg_prediction["prediction_ebg_tool"] = df_ebg_prediction["prediction_median"]
#df_normal = pd.read_csv(os.path.join(os.pardir, "data/branch_supports.csv"))
#df_raxml = pd.read_csv(os.path.join(os.pardir, "data/raxml_classic_supports.csv"))

#df_merged = df_ebg_prediction.merge(df_normal, on=["branchId", "dataset"], how="inner")
#df_merged = df_merged.merge(df_raxml, on=["dataset", "branchId"], how="inner")
df_merged = df_ebg_prediction.merge(df_iqtree, left_on=["dataset", "branchId"], right_on=["dataset", "branchId_true"], how="inner")
df_merged["pred_error_ebg"] = df_merged["true_support"].values - df_merged["prediction_ebg_tool"]
df_merged["iq_tree_error"] = df_merged["true_support"].values - (df_merged["iq_support"].values)
#df_merged["raxml_error"] = df_merged["support"].values*100 - df_merged["support_raxml_classic"].values

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
print("########################"*5)

within_range_5 = df_merged[(df_merged['true_support'] >= df_merged['prediction_lower5'])]
percentage_within_range = (len(within_range_5) / len(df_merged)) * 100
print("Within range 5: "+str(percentage_within_range))

within_range_10 = df_merged[(df_merged['true_support'] >= df_merged['prediction_lower10'])]
percentage_within_range = (len(within_range_10) / len(df_merged)) * 100
print("Within range 10: "+str(percentage_within_range))

print("########################"*5)

df_merged['ebg_over_80'] = (df_merged['prediction_bs_over_80'] >= 0.5).astype(int)
df_merged['support_over_80'] = (df_merged['true_support'] >= 80).astype(int)
df_merged['iq_support_over_95'] = (df_merged['iq_support'] >= 95).astype(int)
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
