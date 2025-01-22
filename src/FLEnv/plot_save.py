import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot metrics from a CSV file.')
parser.add_argument('filename', type=str, help='Path to the CSV file')
args = parser.parse_args()

# Load the CSV data into a DataFrame
df = pd.read_csv(args.filename)

# Pivot the data for easier plotting
df_pivot = df.pivot(index='Round', columns='Metric', values='Value')

# Plot the data
plt.figure(figsize=(10, 6))
for column in df_pivot.columns:
    plt.plot(df_pivot.index, df_pivot[column], label=column)

# Add labels, title, and legend
plt.xlabel('Round')
plt.ylabel('Value')
plt.title('Metric Values Over Rounds')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file with the same name as the input file
output_filename = args.filename.rsplit('.', 1)[0] + '.png'
plt.savefig(output_filename)
plt.show()
