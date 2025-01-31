#%%

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define method names and federated methods
method_names_list = ["EFF_KAN", "REAL_KAN", "ResNet"]
federated_methods_list = ["FedProx", "FedMedian"]

# Initialize an empty list to store data
data_frames = []

# Directory containing the CSV files
data_dir = "./train_logs"  # Change this to the actual directory where CSV files are stored

# Load CSV files for each method and federated method
for method in method_names_list:
    for federated_method in federated_methods_list:
        file_name = f"{method}_{federated_method}_centralized_metrics.csv"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Method"] = method
            df["Federated_Method"] = federated_method
            data_frames.append(df)
        else:
            print(f"File not found: {file_path}")

# Combine all data into a single DataFrame
df_all = pd.concat(data_frames, ignore_index=True)

# Convert to pivot format for plotting
df_pivot = df_all.pivot_table(index="Round", columns=["Method", "Federated_Method", "Metric"], values="Value")

# Function to plot comparison graphs
def plot_metrics(df, metric_name):
    plt.figure(figsize=(12, 6))
    
    for (method, federated_method, metric), series in df.items():
        if metric == metric_name:
            plt.plot(series.index, series.values, label=f"{method} - {federated_method}")

    plt.xlabel("Round")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name} Across Methods and Federated Strategies")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Precision, Recall, and F1-score
plot_metrics(df_pivot, "Precision")
plot_metrics(df_pivot, "Recall")
plot_metrics(df_pivot, "F1")

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define method name and federated methods for comparison
method_name = "EFF_KAN"  # Change this to the method you want to compare
federated_methods_list = ["FedProx", "FedMedian"]

# Directory containing the CSV files (update path if needed)
data_dir = "./train_logs"

# Initialize an empty list to store data
data_frames = []

# Load CSV files for the chosen method under different federated methods
for federated_method in federated_methods_list:
    file_name = f"{method_name}_{federated_method}_centralized_metrics.csv"
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Federated_Method"] = federated_method
        data_frames.append(df)

# Combine all data into a single DataFrame
df_all = pd.concat(data_frames, ignore_index=True)

# Convert to pivot format for plotting
df_pivot = df_all.pivot_table(index="Round", columns=["Federated_Method", "Metric"], values="Value")

# Function to plot comparison graphs for one method under different federated methods
def plot_metrics(df, metric_name, method_name):
    plt.figure(figsize=(12, 6))
    
    for (federated_method, metric), series in df.items():
        if metric == metric_name:
            plt.plot(series.index, series.values, label=f"{federated_method}")

    plt.xlabel("Round")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name} for {method_name} Under Different Federated Methods")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Precision, Recall, and F1-score for the selected method
plot_metrics(df_pivot, "Precision", method_name)
plot_metrics(df_pivot, "Recall", method_name)
plot_metrics(df_pivot, "F1", method_name)

0102.# %%
