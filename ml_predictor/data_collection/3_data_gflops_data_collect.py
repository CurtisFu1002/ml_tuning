import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import yaml

############################
# python 3_data_gflops_data_collect.py --root_dir ./data --yaml ../../tuning_yaml/gemm_fp16_only_mi.yaml --output collected_data.csv
############################

MI_PARAM_NAMES = ['M', 'N', 'K', 'B', 'MIBlockM', 'WaveTileM', 'WaveTileN', 'WaveM', 'WaveN']

def load_mi_params_from_yaml(yaml_path):
    """
    Load MatrixInstruction parameters from the tuning YAML file.
    """
    print(f"Loading MI params from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    try:
        # Traverse YAML structure to find MatrixInstruction
        # Structure: BenchmarkProblems -> list -> list -> dict(ForkParameters) -> list -> dict(MatrixInstruction)
        benchmark_problems = data.get('BenchmarkProblems', [])
        for problem_list in benchmark_problems:
            for item in problem_list:
                if isinstance(item, dict) and 'ForkParameters' in item:
                    fork_params = item['ForkParameters']
                    for param in fork_params:
                        if 'MatrixInstruction' in param:
                            mi_params = param['MatrixInstruction']
                            print(f"\n{'='*60}")
                            print(f"Found {len(mi_params)} MI configurations:")
                            print(f"{'='*60}")
                            print(f"{'idx':>4} | {MI_PARAM_NAMES}")
                            print(f"{'-'*60}")
                            for idx, mi in enumerate(mi_params):
                                print(f"{idx:>4} | {mi}")
                            print(f"{'='*60}\n")
                            return mi_params
    except Exception as e:
        raise ValueError(f"Error parsing YAML structure: {e}")
        
    raise ValueError("Could not find MatrixInstruction in YAML file")



def collect_tuning_data(root_dir, mi_params, output_file='collected_data.csv'):
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
    MI_COUNT = len(mi_params)
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
                    
                    
                    current_mi_params = mi_params[mi_idx]
                    
                    # combine data
                    data_point = {
                        'mi_idx': mi_idx,
                        'M': current_mi_params[0],
                        'N': current_mi_params[1],
                        'K': current_mi_params[2],
                        'B': current_mi_params[3],
                        'MIBlockM': current_mi_params[4],
                        'WaveTileM': current_mi_params[5],
                        'WaveTileN': current_mi_params[6],
                        'WaveM': current_mi_params[7],
                        'WaveN': current_mi_params[8],
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
    parser.add_argument('--yaml', type=str, required=True,
                        help='Path to the tuning yaml file containing MatrixInstruction')
    parser.add_argument('--output', type=str, default='collected_data.csv',
                        help='output CSV file name')
    
    args = parser.parse_args()
    
    mi_params = load_mi_params_from_yaml(args.yaml)
    collect_tuning_data(args.root_dir, mi_params, args.output)


if __name__ == '__main__':
    main()
