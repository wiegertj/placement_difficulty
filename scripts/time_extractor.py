import os
import pandas as pd
import re

# Directory path where the log files are located
dir_path = "/Users/juliuswiegert/Downloads/iqtree_cpu"

# Initialize an empty list to store the data
data = []

# Regular expression pattern to match the time format (e.g., "16m6.196s" or "0m8.776s")
time_pattern = r'(\d+)m([\d.]+)s'

# Loop through all files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".log"):  # Adjust the file extension as needed
        with open(os.path.join(dir_path, filename), 'r') as file:
            content = file.read()

            # Extract user and sys times using regular expressions
            user_match = re.search(r'user\s+' + time_pattern, content)
            sys_match = re.search(r'sys\s+' + time_pattern, content)

            if user_match and sys_match:
                user_minutes, user_seconds = map(float, user_match.groups())
                sys_minutes, sys_seconds = map(float, sys_match.groups())

                # Convert user and sys times to seconds
                user_time_seconds = user_minutes * 60 + user_seconds
                sys_time_seconds = sys_minutes * 60 + sys_seconds

                # Store the data in a dictionary
                data.append({
                    'dataset': filename.replace(".log", "").replace("cpu_usage_", ""),
                    'user_time_iqtree': user_time_seconds,
                    'sys_time_iqtree': sys_time_seconds,
                    'total_cpu_iqtree': user_time_seconds + sys_time_seconds
                })

# Create a DataFrame from the collected data
df_iqtree = pd.DataFrame(data)

# Print the DataFrame or do further processing

# Directory path where the log files are located
dir_path = "/Users/juliuswiegert/Downloads/ebg_cpu"

# Initialize an empty list to store the data
data = []

# Regular expression pattern to match the time format (e.g., "16m6.196s" or "0m8.776s")
time_pattern = r'(\d+)m([\d.]+)s'

# Loop through all files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".log"):  # Adjust the file extension as needed
        with open(os.path.join(dir_path, filename), 'r') as file:
            content = file.read()

            # Extract user and sys times using regular expressions
            user_match = re.search(r'user\s+' + time_pattern, content)
            sys_match = re.search(r'sys\s+' + time_pattern, content)

            if user_match and sys_match:
                user_minutes, user_seconds = map(float, user_match.groups())
                sys_minutes, sys_seconds = map(float, sys_match.groups())

                # Convert user and sys times to seconds
                user_time_seconds = user_minutes * 60 + user_seconds
                sys_time_seconds = sys_minutes * 60 + sys_seconds

                # Store the data in a dictionary
                data.append({
                    'dataset': filename.replace("_ebg.log", "").replace("cpu_usage_", ""),
                    'user_time_ebg': user_time_seconds,
                    'sys_time_ebg': sys_time_seconds,
                    'total_cpu_ebg': user_time_seconds + sys_time_seconds
                })

# Create a DataFrame from the collected data
df_ebg = pd.DataFrame(data)



# Print the DataFrame or do further processing

# Directory path where the log files are located
dir_path = "/Users/juliuswiegert/Downloads/cpu_usage_inf"

# Initialize an empty list to store the data
data = []

# Regular expression pattern to match the time format (e.g., "16m6.196s" or "0m8.776s")
time_pattern = r'(\d+)m([\d.]+)s'

# Loop through all files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".log"):  # Adjust the file extension as needed
        with open(os.path.join(dir_path, filename), 'r') as file:
            content = file.read()

            # Extract user and sys times using regular expressions
            user_match = re.search(r'user\s+' + time_pattern, content)
            sys_match = re.search(r'sys\s+' + time_pattern, content)

            if user_match and sys_match:
                user_minutes, user_seconds = map(float, user_match.groups())
                sys_minutes, sys_seconds = map(float, sys_match.groups())

                # Convert user and sys times to seconds
                user_time_seconds = user_minutes * 60 + user_seconds
                sys_time_seconds = sys_minutes * 60 + sys_seconds

                # Store the data in a dictionary
                data.append({
                    'dataset': filename.replace("_inf.log", "").replace("cpu_usage_", "").replace("_inf", ""),
                    'user_time_inf': user_time_seconds,
                    'sys_time_inf': sys_time_seconds,
                    'total_cpu_inf': user_time_seconds + sys_time_seconds
                })

# Create a DataFrame from the collected data
df_inf = pd.DataFrame(data)

print(df_inf.shape)
print(df_inf)
print(df_ebg.shape)
print(df_ebg)
print(df_iqtree.shape)
print(df_iqtree)
df_merged = df_iqtree.merge(df_ebg, on=["dataset"], how="inner")
print(df_merged.shape)

df_merged = df_merged.merge(df_inf, on=["dataset"], how="inner")
print(df_merged.shape)

df_merged["sum"] = df_merged["total_cpu_inf"] + df_merged["total_cpu_ebg"]
df_merged["ratio"] = (df_merged["sum"] ) / df_merged["total_cpu_iqtree"]

df_merged = df_merged[df_merged["ratio"] <= 50]
print(df_merged)
df_merged.to_csv("test_times.csv")
print(df_merged[["dataset","total_cpu_ebg","total_cpu_inf", "total_cpu_iqtree","sum", "ratio"]].head(5))
print(df_merged["ratio"].median())
