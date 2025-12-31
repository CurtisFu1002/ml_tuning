import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import yaml
import warnings

############################
# python 3_data_gflops_data_collect.py --root_dir ./data --yaml ../../tuning_yaml/gemm_fp16_only_mi.yaml --output collected_data.csv
############################

GEMM_RANGE = 8192


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

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No 00_Final.csv files found in {root_dir}")

    print(f"finded {len(csv_files)} num of 00_Final.csv files\n")

    # MI kernel col range in 00_Final.csv from 17 to 17+127 cols (total 128 cols)
    MI_START_COL = 16
    MI_COUNT = len(mi_params)
    MI_END_COL = MI_START_COL + MI_COUNT  # 16 + 128 = 145

    # Counters for warning summary
    skipped_OutOfRange_ProblemSizes = 0
    skipped_nan_or_zero = 0

    for csv_path in csv_files:
        print(f"Processing: {csv_path.parent.name}")
        df = pd.read_csv(csv_path)
        
        # Check required columns (m, n, k) - RAISE ERROR
        required_cols = [' SizeI', ' SizeJ', ' SizeL']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns {missing_cols} in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Check MI kernel column count - RAISE ERROR
        if len(df.columns) < MI_END_COL:
            raise ValueError(
                f"Insufficient columns in {csv_path}: "
                f"expected at least {MI_END_COL} columns, but got {len(df.columns)}. "
                f"Expected MI configurations: {MI_COUNT}"
            )
        
        # Traverse different problem sizes
        for row_idx, row in df.iterrows():
            size_i = row[' SizeI']
            size_j = row[' SizeJ']
            size_l = row[' SizeL']
            
            # Skip large problem sizes - WARNING
            if size_i > GEMM_RANGE or size_j > GEMM_RANGE:
                skipped_OutOfRange_ProblemSizes += 1
                warnings.warn(
                    f"Skipping large problem size: m={size_i}, n={size_j}, k={size_l} "
                    f"(exceeds {GEMM_RANGE} limit) in {csv_path}"
                )
                continue
            
            # Traverse different MI configs
            for mi_idx in range(MI_COUNT):
                col_idx = MI_START_COL + mi_idx
                gflops = row.iloc[col_idx]
                
                # Skip NaN or 0 - WARNING
                if pd.isna(gflops) or gflops == 0:
                    skipped_nan_or_zero += 1
                    warnings.warn(
                        f"Skipping invalid GFLOPS value: {gflops} "
                        f"for mi_idx={mi_idx}, m={size_i}, n={size_j}, k={size_l} "
                        f"in {csv_path}"
                    )
                    continue
                
                current_mi_params = mi_params[mi_idx]
                
                # Combine data
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
        
        print(f" Collected {len(df)} problem sizes")

    # Print warning summary
    if  skipped_OutOfRange_ProblemSizes > 0 or skipped_nan_or_zero > 0:
        print(f"\n{'='*50}")
        print("Warning Summary:")
        print(f"  - Skipped {skipped_OutOfRange_ProblemSizes} rows due to large problem size (>9000)")
        print(f"  - Skipped {skipped_nan_or_zero} entries due to NaN or zero GFLOPS")
        print(f"{'='*50}")
    
    if not all_data:
        raise ValueError("No valid data collected. All rows may have been skipped.")
    
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
