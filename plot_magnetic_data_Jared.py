import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process file path and identifier.")

# Add the arguments
parser.add_argument("file_path", type=str, help="The path to the data file")
parser.add_argument("identifier", type=str, help="The identifier for the output file name")

# Parse the arguments
args = parser.parse_args()

# Now you can access the file_path and identifier like this:
file_path = args.file_path
identifier = args.identifier

# file_path = "/mnt/c/Users/werty/Downloads/MagneticFieldData/008/Tek001_LightsOff_BreakerOff_ALL.csv" 
top_level_output_dir = "PLOTS_AND_RESULTS"
identifier_directory = f'./{top_level_output_dir}/{identifier}'
if not os.path.isdir(identifier_directory):
    os.makedirs(identifier_directory)

results_path = os.path.join(identifier_directory, "results.txt")
voltage_plot_filename = os.path.join(identifier_directory, "math1_vs_time_plot.png")
b_plot_filename = os.path.join(identifier_directory, "b_vs_time_plot.png")
calculated_norms_plot_filename = os.path.join(identifier_directory, "calculated_norms_plot.png")
voltage_hist_filename = os.path.join(identifier_directory, "VoltageHist.png")
b_hist_filename = os.path.join(identifier_directory, "BFieldHist.png")
print('starting...')
print(f"DATA PATH == {file_path}")

# Load the CSV file
df = pd.read_csv(file_path, skiprows=13)

# Clean the dataframe
df.columns = ['TIME', 'CH1', 'CH2', 'CH3', 'CH4']
# df = df[['TIME', 'MATH1']].dropna()

# Convert the TIME and MATH1 columns to numeric
df['TIME'] = pd.to_numeric(df['TIME'])

# Calculate B using the formula B = MATH1 / 48800
df['MATH1'] = np.sqrt((df['CH1']**2) + (df['CH2']**2) + (df['CH3']**2))
df['B'] = df['MATH1'] / 48800

print('dataframe loaded, plotting...')

def save_plot(x_axis_name, y_axis_name, filename, color):
	# Plot MATH1
	plt.figure(figsize=(10, 6))
	plt.plot(df[x_axis_name], df[y_axis_name], color=color, marker='o', linestyle='None')
	plt.title(f'{y_axis_name} vs {x_axis_name}')
	plt.xlabel(x_axis_name)
	plt.ylabel(y_axis_name)
	plt.grid(True)

	# Save MATH1 plot to a file
	plt.savefig(filename)
	plt.close()

def save_histogram(col_name, filename):
	df[col_name].hist(bins=50)
	plt.title(f'Histogram of {col_name}')
	plt.xlabel(f'{col_name} values')
	plt.ylabel('Frequency')
	plt.savefig(filename)
	plt.close()

save_plot('TIME', 'MATH1', voltage_plot_filename, 'b')
print(f'math1 plot saved to {voltage_plot_filename}')
save_plot('TIME', 'B', b_plot_filename, 'r')
print(f'B plot saved to {b_plot_filename}')
# save_plot('TIME', 'CalcNorm', calculated_norms_plot_filename, 'b')
# print(f'CalcNorm plot saved to {calculated_norms_plot_filename}')
save_histogram('MATH1', voltage_hist_filename)
save_histogram('B', b_hist_filename)



# Calculate the RMS value of B
rms_B = np.sqrt(np.mean(df['B']**2))
rms_V = np.sqrt(np.mean(df['MATH1']**2))

# Open a file to save the output
with open(results_path, 'w') as f:
    # Calculate and write values to the file
    f.write(f"RMS value of voltage norm: {rms_V}\n")
    f.write('voltage norm summary: \n')
    f.write(df["MATH1"].describe().to_string())
    f.write('\n\n')
    f.write(f"RMS value of B: {rms_B}\n")
    f.write('B summary: \n')
    f.write(df["B"].describe().to_string())

# Confirmation message
print(f"Results saved to {results_path}")
