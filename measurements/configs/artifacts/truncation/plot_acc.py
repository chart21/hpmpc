import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
import sys
import os
import random
import warnings
import glob
import argparse

# Your existing base script
wide = False
full = False

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
    "font.size": 16,
    "pgf.rcfonts": False,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "lines.linewidth": 1,
    "lines.markersize": 6
})

# Define marker styles
marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']


def get_label(trunc_approach):
    labels = {
        0: r'$TS_{\{L\}}$',
        1: r'$TS_{\{1\}}$',
        2: r'$TE_{\{0\}}$',
        3: r'$TE_{\{1\}}$',
        4: r'$TS_{\{Mix\}}$',
    }
    return labels.get(trunc_approach, f'Unknown ({trunc_approach})')

# Function to create and save plot
def create_and_save_plot(data, bitlength, trunc_then_mult, trunc_delayed, file_prefix):
    if wide:
        if full:
            fig, ax = plt.subplots(figsize=(6.16, 4.08))
        else:
            fig, ax = plt.subplots(figsize=(3.08, 3.04))
    else:
        if full:
            fig, ax = plt.subplots(figsize=(3.08, 3.08))
        else:
            fig, ax = plt.subplots(figsize=(3.08, 3.08))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['TRUNC_APPROACH'].unique())))
    
    for i, (trunc_approach, group) in enumerate(data.groupby('TRUNC_APPROACH')):
        marker = marker_styles[i % len(marker_styles)]
        ax.plot(group['FRACTIONAL'], group['ACCURACY(%)'], marker=marker, linestyle='-', 
                label=f'TRUNC_APPROACH={trunc_approach}', color=colors[i])
    
    
    ax.set_xlabel('Fractional Bits')
    ax.set_ylabel('Accuracy (\%)')
    
    plt.tight_layout()
    # Convert to integers for filename
    bl_int = int(bitlength)
    ttm_int = int(trunc_then_mult)
    td_int = int(trunc_delayed)
    filename = f'{file_prefix}BL{bl_int}_TTM{ttm_int}_TD{td_int}.pdf'
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# Function to create and save legend
def create_and_save_legend(data, file_prefix):
    fig, ax = plt.subplots(figsize=(6.16, 0.5))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['TRUNC_APPROACH'].unique())))
    legend_elements = [Line2D([0], [0], color=colors[i], marker=marker_styles[i % len(marker_styles)], linestyle='-',
                              label=get_label(trunc_approach))
                       for i, trunc_approach in enumerate(sorted(data['TRUNC_APPROACH'].unique()))]
    
    ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), frameon=False)
    ax.axis('off')
    # Assuming you have a plot already created
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(legend_elements),
               frameon=False, handlelength=1, handleheight=1.5, handletextpad=0.5, columnspacing=1)
    
    plt.tight_layout()
    fig.savefig(f'{file_prefix}legend.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# Function to process a single CSV file
def process_csv_file(csv_file):
    folder_path = os.path.dirname(csv_file)
    csv_filename = os.path.basename(csv_file)
    
    # Extract folder name for file prefix
    folder_name = os.path.basename(os.path.normpath(folder_path))
    file_prefix = f"{folder_name}_{os.path.splitext(csv_filename)[0]}_"

    # Read CSV file
    df = pd.read_csv(csv_file)
    # Replace empty accuracy values with 0
    df['ACCURACY(%)'] = df['ACCURACY(%)'].fillna(0)

    # Create plots for each combination
    for (bitlength, trunc_then_mult, trunc_delayed), group in df.groupby(['BITLENGTH', 'TRUNC_THEN_MULT', 'TRUNC_DELAYED']):
        create_and_save_plot(group, bitlength, trunc_then_mult, trunc_delayed, file_prefix)

    # Create and save legend
    create_and_save_legend(df, file_prefix)

    print(f"Plots and legend have been generated and saved with prefix: {file_prefix}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from CSV files.")
    parser.add_argument("path", help="Path to a CSV file or directory containing CSV files")
    args = parser.parse_args()

    path = os.path.expanduser(args.path)  # Expand ~ to full path

    if os.path.isfile(path):
        if path.endswith('.csv'):
            print(f"Processing file: {path}")
            process_csv_file(path)
        else:
            print(f"Error: {path} is not a CSV file.")
            sys.exit(1)
    elif os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {path}")
            sys.exit(1)
        for csv_file in csv_files:
            if 'figure' not in os.path.basename(csv_file):
                continue
            print(f"Processing {os.path.basename(csv_file)}...")
            process_csv_file(csv_file)
    else:
        print(f"Error: {path} is not a valid file or directory.")
        sys.exit(1)

    print("All processing completed.")

