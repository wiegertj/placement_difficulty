import os
from statistics import mean

import pandas as pd
from numpy import median

df = pd.read_csv(os.pardir + "/data/raxml_data_results.csv")

df_pred = df.drop(columns=["prediction_o70","prediction_o80","o70_true","o80_true","prediction_error", "Unnamed: 0_y", "prediction_median","prediction_lower_75","prediction_upper_75","prediction_bs_over_70","prediction_bs_over_80" ,"Unnamed: 0_x"
])
df_pred.to_csv("df_pred.csv")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Assuming you have a DataFrame named df

# Calculate F1 score
f1 = f1_score(df["o70_true"].values, df["prediction_o70"].values)

# Calculate accuracy
accuracy = accuracy_score(df["o70_true"].values, df["prediction_o70"].values)

print("F1 Score:", f1)
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have a DataFrame named df with a "prediction_error" column
# You may need to install matplotlib if you haven't already: pip install matplotlib
print(median(abs(df["prediction_error"])))
# Plot the histogram
plt.hist(df["support"], bins=20, color='blue', edgecolor='black')
plt.title('Prediction Error Histogram')
plt.xlabel('Support')
plt.ylabel('Frequency')

# Show the histogram
plt.show()
