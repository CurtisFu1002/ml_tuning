#!/usr/bin/env python3
"""
Batch YAML Generator for Tensile Tuning

Generates multiple tuning YAML files by combining a baseline
template (e.g. gemm_fp16_only_mi.yaml) with problem sizes from a CSV.

Features:
- Supports multi-GPU assignment (--num-devices, --skip-device)
- Optional limit on number of problems to process (--limit)
- Creates batched YAMLs for parallel Tensile tuning

Example:
    python 1_batch_tuning_yaml_generator.py \
        --csv baseline_data_infra/problem_sizes_104CU.csv \
        --template ../tuning_yaml/gemm_fp16_only_mi.yaml \
        --batch-size 200 \
        --out-dir tuning_batches_104CU \
        --num-devices 4 \
        --skip-device 0 \
        --limit 5000
"""
from typing import Optional
import argparse
import math
import os
from pathlib import Path
import pandas as pd
from ruamel.yaml import YAML
from tqdm import tqdm


def generate_batches(csv_path: Path, template_path: Path, batch_size: int, out_dir: Path,
                     num_devices: int = 1, skip_device: Optional[int] = None, limit: Optional[int] = None):
    # === setup ===
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)
    os.makedirs(out_dir, exist_ok=True)

    with open(template_path, "r") as f:
        base_yaml = yaml.load(f)

    df = pd.read_csv(csv_path)
    total = len(df)

    if limit is not None and limit < total:
        df = df.iloc[:limit]
        total = limit
        print(f" Limit set: using first {limit} problem sizes")

    num_batches = math.ceil(total / batch_size)
    print(f" Found {total} problem sizes → generating {num_batches} YAML batches (batch_size={batch_size})")

    # === prepare device list ===
    all_devices = list(range(num_devices))
    if skip_device is not None and skip_device in all_devices:
        all_devices.remove(skip_device)
    if not all_devices:
        raise ValueError("No available devices to assign after skipping.")
    print(f" Available devices for assignment: {all_devices}")

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, total)
        batch_df = df.iloc[start:end]

        # deep copy YAML (avoid shared reference)
        batch_yaml = base_yaml.copy()

        # === assign device ===
        assigned_device = all_devices[batch_idx % len(all_devices)]
        batch_yaml["GlobalParameters"]["Device"] = int(assigned_device)

        # === update problem sizes ===
        from ruamel.yaml.comments import CommentedSeq

        problem_sizes = []
        for m, n, b, k in zip(batch_df.M, batch_df.N, batch_df.B, batch_df.K):
            arr = CommentedSeq([int(m), int(n), int(b), int(k)])
            arr.fa.set_flow_style()   
            problem_sizes.append({"Exact": arr})

        batch_yaml["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"] = problem_sizes

        # === output path ===
        out_path = Path(out_dir) / f"tuning_batch_{batch_idx:04d}_dev{assigned_device}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(batch_yaml, f)

    print(f" Generated {num_batches} YAML files in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate batched tuning YAMLs from problem size CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Input problem size CSV (e.g. problem_sizes_104CU.csv)")
    parser.add_argument("--template", type=Path, required=True, help="Baseline tuning YAML (e.g. gemm_fp16_only_mi.yaml)")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of problem sizes per YAML file")
    parser.add_argument("--out-dir", type=Path, default=Path("tuning_batches"), help="Output folder for generated YAMLs")
    parser.add_argument("--num-devices", type=int, default=1, help="Total number of GPUs available")
    parser.add_argument("--skip-device", type=int, default=None, help="GPU ID to skip (e.g. 0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of problems to process (default: all)")
    args = parser.parse_args()

    generate_batches(args.csv, args.template, args.batch_size, args.out_dir,
                     args.num_devices, args.skip_device, args.limit)


if __name__ == "__main__":
    main()

