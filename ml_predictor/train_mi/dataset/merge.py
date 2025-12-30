import pandas as pd
import argparse
import os

# python merge.py --file1 G1_step256_shift0.csv --file2 G2_step256_shift128.csv --output G1_G3_merged.csv

def main():
    parser = argparse.ArgumentParser(description='Merge two CSV files with same columns.')
    parser.add_argument('--file1', '-f1', type=str, required=True, help='Path to the first CSV file')
    parser.add_argument('--file2', '-f2', type=str, required=True, help='Path to the second CSV file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    if not os.path.exists(args.file1):
        print(f"Error: File not found: {args.file1}")
        return
    if not os.path.exists(args.file2):
        print(f"Error: File not found: {args.file2}")
        return

    print(f"Reading {args.file1}...")
    df1 = pd.read_csv(args.file1)

    print(f"Reading {args.file2}...")
    df2 = pd.read_csv(args.file2)

    try:
        df2 = df2[df1.columns]
    except KeyError as e:
        print(f"Error: Columns do not match between files. Missing: {e}")
        return

    print("Merging dataframes...")
    merged_df = pd.concat([df1, df2], ignore_index=True)

    print(f"Saving merged data to {args.output}...")
    merged_df.to_csv(args.output, index=False)
    
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    main()