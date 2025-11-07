import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np

############################
# python gflops_data_collect.py --root_dir ./data --output collected_data.csv
############################

MI_PARAMS = [
    [16, 16,16, 1,  1,   1, 1,  4,1],
    [16, 16,16, 1,  1,   2, 1,  4,1],
    [16, 16,16, 1,  1,   3, 1,  4,1],
    [16, 16,16, 1,  1,   4, 1,  4,1],
    [16, 16,16, 1,  1,   1, 1,  2,2],
    [16, 16,16, 1,  1,   2, 1,  2,2],
    [16, 16,16, 1,  1,   3, 1,  2,2],
    [32, 32, 8, 1,  1,   1, 1,  4,1],
    [16, 16,16, 1,  1,   5, 1,  2,2],
    [16, 16,16, 1,  1,   6, 1,  2,2],
    [16, 16,16, 1,  1,   7, 1,  2,2],
    [32, 32, 8, 1,  1,   2, 1,  4,1],
    [16, 16,16, 1,  1,   1, 3,  4,1],
    [16, 16,16, 1,  1,   2, 3,  4,1],
    [16, 16,16, 1,  1,   3, 3,  4,1],
    [16, 16,16, 1,  1,   4, 3,  4,1],
    [16, 16,16, 1,  1,   1, 1,  1,4],
    [16, 16,16, 1,  1,   2, 1,  1,4],
    [16, 16,16, 1,  1,   3, 1,  1,4],
    [32, 32, 8, 1,  1,   1, 1,  2,2],
    [16, 16,16, 1,  1,   5, 1,  1,4],
    [16, 16,16, 1,  1,   6, 1,  1,4],
    [16, 16,16, 1,  1,   7, 1,  1,4],
    [32, 32, 8, 1,  1,   2, 1,  2,2],
    [16, 16,16, 1,  1,   9, 1,  1,4],
    [16, 16,16, 1,  1,  10, 1,  1,4],
    [16, 16,16, 1,  1,  11, 1,  1,4],
    [32, 32, 8, 1,  1,   3, 1,  2,2],
    [16, 16,16, 1,  1,  13, 1,  1,4],
    [16, 16,16, 1,  1,  14, 1,  1,4],
    [16, 16,16, 1,  1,  15, 1,  1,4],
    [32, 32, 8, 1,  1,   2, 2,  4,1],
    [16, 16,16, 1,  1,   1, 5,  4,1],
    [16, 16,16, 1,  1,   2, 5,  4,1],
    [16, 16,16, 1,  1,   3, 5,  4,1],
    [16, 16,16, 1,  1,   4, 5,  4,1],
    [16, 16,16, 1,  1,   1, 3,  2,2],
    [16, 16,16, 1,  1,   2, 3,  2,2],
    [16, 16,16, 1,  1,   3, 3,  2,2],
    [32, 32, 8, 1,  1,   1, 3,  4,1],
    [16, 16,16, 1,  1,   5, 3,  2,2],
    [16, 16,16, 1,  1,   6, 3,  2,2],
    [16, 16,16, 1,  1,   7, 3,  2,2],
    [32, 32, 8, 1,  1,   2, 3,  4,1],
    [16, 16,16, 1,  1,   1, 7,  4,1],
    [16, 16,16, 1,  1,   2, 7,  4,1],
    [16, 16,16, 1,  1,   3, 7,  4,1],
    [16, 16,16, 1,  1,   4, 7,  4,1],
    [16, 16,16, 1,  1,   1, 2,  1,4],
    [32, 32, 8, 1,  1,   1, 1,  1,4],
    [16, 16,16, 1,  1,   3, 2,  1,4],
    [32, 32, 8, 1,  1,   2, 1,  1,4],
    [16, 16,16, 1,  1,   5, 2,  1,4],
    [32, 32, 8, 1,  1,   3, 1,  1,4],
    [16, 16,16, 1,  1,   7, 2,  1,4],
    [32, 32, 8, 1,  1,   2, 2,  2,2],
    [16, 16,16, 1,  1,   9, 2,  1,4],
    [32, 32, 8, 1,  1,   5, 1,  1,4],
    [16, 16,16, 1,  1,  11, 2,  1,4],
    [32, 32, 8, 1,  1,   6, 1,  1,4],
    [16, 16,16, 1,  1,  13, 2,  1,4],
    [32, 32, 8, 1,  1,   7, 1,  1,4],
    [16, 16,16, 1,  1,  15, 2,  1,4],
    [32, 32, 8, 1,  1,   4, 2,  2,2],
    [16, 16,16, 1,  1,   1, 9,  4,1],
    [16, 16,16, 1,  1,   2, 9,  4,1],
    [16, 16,16, 1,  1,   3, 9,  4,1],
    [16, 16,16, 1,  1,   4, 9,  4,1],
    [16, 16,16, 1,  1,   1, 5,  2,2],
    [16, 16,16, 1,  1,   2, 5,  2,2],
    [16, 16,16, 1,  1,   3, 5,  2,2],
    [32, 32, 8, 1,  1,   1, 5,  4,1],
    [16, 16,16, 1,  1,   5, 5,  2,2],
    [16, 16,16, 1,  1,   6, 5,  2,2],
    [16, 16,16, 1,  1,   7, 5,  2,2],
    [32, 32, 8, 1,  1,   2, 5,  4,1],
    [16, 16,16, 1,  1,   1,11,  4,1],
    [16, 16,16, 1,  1,   2,11,  4,1],
    [16, 16,16, 1,  1,   3,11,  4,1],
    [16, 16,16, 1,  1,   4,11,  4,1],
    [16, 16,16, 1,  1,   1, 3,  1,4],
    [16, 16,16, 1,  1,   2, 3,  1,4],
    [16, 16,16, 1,  1,   3, 3,  1,4],
    [32, 32, 8, 1,  1,   1, 3,  2,2],
    [16, 16,16, 1,  1,   5, 3,  1,4],
    [16, 16,16, 1,  1,   6, 3,  1,4],
    [16, 16,16, 1,  1,   7, 3,  1,4],
    [32, 32, 8, 1,  1,   2, 3,  2,2],
    [16, 16,16, 1,  1,   9, 3,  1,4],
    [16, 16,16, 1,  1,  10, 3,  1,4],
    [16, 16,16, 1,  1,  11, 3,  1,4],
    [32, 32, 8, 1,  1,   3, 3,  2,2],
    [16, 16,16, 1,  1,  13, 3,  1,4],
    [16, 16,16, 1,  1,   7, 6,  2,2],
    [16, 16,16, 1,  1,  15, 3,  1,4],
    [32, 32, 8, 1,  1,   2, 6,  4,1],
    [16, 16,16, 1,  1,   1,13,  4,1],
    [16, 16,16, 1,  1,   2,13,  4,1],
    [16, 16,16, 1,  1,   3,13,  4,1],
    [16, 16,16, 1,  1,   4,13,  4,1],
    [16, 16,16, 1,  1,   1, 7,  2,2],
    [16, 16,16, 1,  1,   2, 7,  2,2],
    [16, 16,16, 1,  1,   3, 7,  2,2],
    [32, 32, 8, 1,  1,   1, 7,  4,1],
    [16, 16,16, 1,  1,   5, 7,  2,2],
    [16, 16,16, 1,  1,   3,14,  4,1],
    [16, 16,16, 1,  1,   7, 7,  2,2],
    [32, 32, 8, 1,  1,   2, 7,  4,1],
    [16, 16,16, 1,  1,   1,15,  4,1],
    [16, 16,16, 1,  1,   2,15,  4,1],
    [16, 16,16, 1,  1,   3,15,  4,1],
    [16, 16,16, 1,  1,   4,15,  4,1],
    [16, 16,16, 1,  1,   1, 4,  1,4],
    [32, 32, 8, 1,  1,   1, 2,  1,4],
    [16, 16,16, 1,  1,   3, 4,  1,4],
    [32, 32, 8, 1,  1,   2, 2,  1,4],
    [16, 16,16, 1,  1,   5, 4,  1,4],
    [32, 32, 8, 1,  1,   3, 2,  1,4],
    [16, 16,16, 1,  1,   7, 4,  1,4],
    [32, 32, 8, 1,  1,   2, 4,  2,2],
    [16, 16,16, 1,  1,   9, 4,  1,4],
    [32, 32, 8, 1,  1,   5, 2,  1,4],
    [16, 16,16, 1,  1,  11, 4,  1,4],
    [32, 32, 8, 1,  1,   3, 4,  2,2],
    [16, 16,16, 1,  1,  13, 4,  1,4],
    [32, 32, 8, 1,  1,   7, 2,  1,4],
    [16, 16,16, 1,  1,  15, 4,  1,4],
    [32, 32, 8, 1,  1,   2, 8,  4,1],
]

MI_PARAM_NAMES = ['M', 'N', 'K', 'B', 'MIBlockM', 'WaveTileM', 'WaveTileN', 'WaveM', 'WaveN']

def collect_tuning_data(root_dir, output_file='collected_data.csv'):
    """
    collect all the 00_Final.csv data
    
    every data contains:
    - mi: MI index (0-127)
    - gflops: kernel GFLOPS 
    - SizeI: Problem size m
    - SizeJ: Problem size n
    - SizeL: Problem size k 
    """
    all_data = []
    root_path = Path(root_dir)
    csv_files = list(root_path.rglob('00_Final.csv'))

    print(f"finded {len(csv_files)} num of 00_Final.csv files\n")

    # MI kernel col range in 00_Final.csv from 17 to 17+127 cols (total 128 cols)
    MI_START_COL = 16
    MI_COUNT = 128
    MI_END_COL = MI_START_COL + MI_COUNT  # 16 + 128 = 145

    for csv_path in csv_files:
        try:
            print(f"processing: {csv_path.parent.name}")
            df = pd.read_csv(csv_path)
            
            # get m, n, k
            required_cols = [' SizeI', ' SizeJ', ' SizeL'] # m,n,k
            if not all(col in df.columns for col in required_cols):
                print(f" misssing mnk")
                continue
            
            
            # check mi kernel nums
            if len(df.columns) < MI_END_COL:
                print(f" col out of range ( only {len(df.columns)} cols)")
                continue
            
            # traverse different problem size
            for row_idx, row in df.iterrows():
                size_i = row[' SizeI']
                size_j = row[' SizeJ']
                size_l = row[' SizeL']
                
                if (size_i > 9000 or size_j > 9000):
                    continue
                
                # traverse diff MI ( 17 ~ 144 col)
                for mi_idx in range(MI_COUNT):
                    col_idx = MI_START_COL + mi_idx
                    gflops = row.iloc[col_idx]
                    
                    # skip NaN or 0
                    if pd.isna(gflops) or gflops == 0:
                        continue
                    
                    
                    mi_params = MI_PARAMS[mi_idx]
                    
                    # combine data
                    data_point = {
                        'mi_idx': mi_idx,
                        'M': mi_params[0],
                        'N': mi_params[1],
                        'K': mi_params[2],
                        'B': mi_params[3],
                        'MIBlockM': mi_params[4],
                        'WaveTileM': mi_params[5],
                        'WaveTileN': mi_params[6],
                        'WaveM': mi_params[7],
                        'WaveN': mi_params[8],
                        'gflops': gflops,
                        'm': int(size_i),
                        'n': int(size_j),
                        'k': int(size_l),
                    }                    
                    all_data.append(data_point)
            
            print(f" collected {len(df)} nums of problem sizes")
            
        except Exception as e:
            print(f"  error : {e}")
    
    if not all_data:
        raise ValueError("Fail collecting")
    
    # To DataFrame
    result_df = pd.DataFrame(all_data)
    
    # save as CSV
    result_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Total {len(result_df)} data collected")
    print(f"Save to: {output_file}")
    print(f"{'='*50}")
    print(f"\n Data preview:")
    print(result_df.head(10))
    print(f"\n Stat:")
    print(result_df.describe())
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description='collect tuning data')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root direct')
    parser.add_argument('--output', type=str, default='collected_data.csv',
                        help='output CSV file name')
    
    args = parser.parse_args()
    
    collect_tuning_data(args.root_dir, args.output)


if __name__ == '__main__':
    main()
