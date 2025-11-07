#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import os

#############################
# python3 problem_csv_gen.py \
#   --M_start 512 --M_end 8192 --M_step 32 \
#   --N_start 512 --N_end 8192 --N_step 32 \
#   --K_list "512,4096,16384" \
#   --B 1 --out_dir "./generated_problem_sizes"
#############################
# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate problem size CSVs for GEMM tuning heatmap")
    parser.add_argument('--M_start', type=int, default=512, help="Starting value for M")
    parser.add_argument('--M_end', type=int, default=8192, help="Ending value for M")
    parser.add_argument('--M_step', type=int, default=32, help="Step size for M")
    parser.add_argument('--N_start', type=int, default=512, help="Starting value for N")
    parser.add_argument('--N_end', type=int, default=8192, help="Ending value for N")
    parser.add_argument('--N_step', type=int, default=32, help="Step size for N")
    parser.add_argument('--K_list', type=str, default="512,4096,16384", help="Comma separated list of K values")
    parser.add_argument('--B', type=int, default=1, help="Batch size (fixed)")
    parser.add_argument('--out_dir', type=str, default='./heatmap_problem_sizes', help="Output directory")
    return parser.parse_args()

# -----------------------------
# Main logic
# -----------------------------
def generate_csv(M_range, N_range, K_list, B, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    total_m = len(M_range)
    total_n = len(N_range)
    print(f"Generating heatmap grid: {total_m} × {total_n} = {total_m * total_n} combinations per K")

    for K in K_list:
        print(f"  -> Generating for K={K}")
        rows = [{'M': m, 'N': n, 'B': B, 'K': K} for m, n in itertools.product(M_range, N_range)]
        df = pd.DataFrame(rows)
        
        out_path = os.path.join(out_dir, f"problem_sizes_heatmap_K{K}.csv")
        df.to_csv(out_path, index=False)
        print(f"     Saved: {out_path} ({len(df)} rows)")

    print(" All CSV files generated successfully.")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    # Convert K_list from string to list of integers
    K_list = list(map(int, args.K_list.split(',')))

    # Define ranges for M and N based on arguments
    M_range = range(args.M_start, args.M_end + 1, args.M_step)
    N_range = range(args.N_start, args.N_end + 1, args.N_step)

    # Generate problem size CSVs
    generate_csv(M_range, N_range, K_list, args.B, args.out_dir)
