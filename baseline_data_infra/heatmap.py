import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

############################
# python my_heatmap.py --root_dir ./data --output my_heatmap.png
############################

def collect_tuning_csv_result(root_dir):
    all_mn_sol = []
    root_path = Path(root_dir)
    csv_files = list(root_path.rglob('00_Final.csv'))

    print(f"find {len(csv_files)} 00_Final.csv")

    needed_cols = [' SizeI', ' SizeJ', ' WinnerIdx']

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if all(col in df.columns for col in needed_cols):
                all_mn_sol.append(df[needed_cols])
            else:
                print(f"{csv_path} columns: {df.columns.tolist()} (missing needed columns)")
        except Exception as e:
            print(f"{csv_path} cannot be read: {e}")
    
    if not all_mn_sol:
        raise ValueError("No valid 00_Final.csv files found with required columns!")
    return pd.concat(all_mn_sol, ignore_index=True)

def plot_heatmap(data, output_path = "mn_heatmap.png"):
    """
    m(x axis) n(y axis)  MI parameter(WinnerIdx) plot heatmap
    """
    data.columns = data.columns.str.strip()

    heatmap_data = data.pivot_table(
        index='SizeJ',      # y axis
        columns='SizeI',    # x axis
        values='WinnerIdx', # cell value
        aggfunc='mean'      # if duplicate -> mean 
    )
    heatmap_data = heatmap_data.sort_index(ascending=False)

    vmin, vmax = 0, 127

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="viridis",  
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'WinnerIdx (0~127)'}
    )
    plt.title("WinnerIdx Heatmap")
    plt.xlabel("M")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="tuning results folder")
    parser.add_argument(
        "--tuning_result_folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="solution_heatmap.png",
    )
    args = parser.parse_args()

    df = collect_tuning_csv_result(args.tuning_result_folder)
    print(df.head())
    plot_heatmap(df, output_path=args.output)


if __name__ == "__main__":
    main()