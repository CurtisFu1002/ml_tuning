#!/usr/bin/env python3
"""
Analyze speedup achieved by MI configuration optimization
"""

import re
import argparse
from pathlib import Path
import pandas as pd

def parse_timing_summary(summary_file):
    """Parse timing summary file"""
    with open(summary_file, 'r') as f:
        content = f.read()
    
    # Extract individual timings
    timings = []
    pattern = r'(optimal_m\d+_n\d+_b\d+_k\d+_top\d+)\s+(\d+)\s+(SUCCESS|FAILED)'
    
    for match in re.finditer(pattern, content):
        yaml_name, time_str, status = match.groups()
        if status == 'SUCCESS':
            timings.append({
                'yaml': yaml_name,
                'time': int(time_str),
                'status': status
            })
    
    # Extract summary statistics
    total_match = re.search(r'Total execution time\s+:\s+(\d+)s', content)
    avg_match = re.search(r'Average time per YAML\s+:\s+(\d+)s', content)
    
    total_time = int(total_match.group(1)) if total_match else 0
    avg_time = int(avg_match.group(1)) if avg_match else 0
    
    return {
        'timings': timings,
        'total_time': total_time,
        'avg_time': avg_time,
        'count': len(timings)
    }

def analyze_speedup(baseline_file, optimized_file, output_file=None):
    """Analyze speedup between baseline and optimized configurations"""
    
    print("=" * 80)
    print(" MI Configuration Optimization Speedup Analysis")
    print("=" * 80)
    
    # Parse both files
    baseline = parse_timing_summary(baseline_file)
    optimized = parse_timing_summary(optimized_file)
    
    # Calculate speedup
    if baseline['total_time'] > 0:
        speedup_ratio = baseline['total_time'] / optimized['total_time']
        time_saved = baseline['total_time'] - optimized['total_time']
        reduction_pct = (time_saved / baseline['total_time']) * 100
    else:
        speedup_ratio = 0
        time_saved = 0
        reduction_pct = 0
    
    # Print results
    print(f"\nBaseline Configuration:")
    print(f"  Total execution time : {baseline['total_time']}s")
    print(f"  Number of YAMLs      : {baseline['count']}")
    print(f"  Average per YAML     : {baseline['avg_time']}s")
    
    print(f"\nOptimized Configuration:")
    print(f"  Total execution time : {optimized['total_time']}s")
    print(f"  Number of YAMLs      : {optimized['count']}")
    print(f"  Average per YAML     : {optimized['avg_time']}s")
    
    print(f"\nSpeedup Results:")
    print(f"  Speedup ratio        : {speedup_ratio:.2f}x")
    print(f"  Time saved           : {time_saved}s ({time_saved/60:.1f} min)")
    print(f"  Time reduction       : {reduction_pct:.1f}%")
    
    # Per-YAML comparison (if available)
    if baseline['timings'] and optimized['timings']:
        print(f"\nPer-YAML Timing Comparison:")
        print("-" * 80)
        print(f"{'YAML':<50} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for b_timing, o_timing in zip(baseline['timings'], optimized['timings']):
            if b_timing['time'] > 0:
                per_speedup = b_timing['time'] / o_timing['time']
                print(f"{o_timing['yaml']:<50} {b_timing['time']:<12} "
                      f"{o_timing['time']:<12} {per_speedup:.2f}x")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" MI Configuration Optimization Speedup Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Baseline: {baseline_file}\n")
            f.write(f"Optimized: {optimized_file}\n\n")
            
            f.write(f"Baseline Configuration:\n")
            f.write(f"  Total execution time : {baseline['total_time']}s\n")
            f.write(f"  Number of YAMLs      : {baseline['count']}\n")
            f.write(f"  Average per YAML     : {baseline['avg_time']}s\n\n")
            
            f.write(f"Optimized Configuration:\n")
            f.write(f"  Total execution time : {optimized['total_time']}s\n")
            f.write(f"  Number of YAMLs      : {optimized['count']}\n")
            f.write(f"  Average per YAML     : {optimized['avg_time']}s\n\n")
            
            f.write(f"Speedup Results:\n")
            f.write(f"  Speedup ratio        : {speedup_ratio:.2f}x\n")
            f.write(f"  Time saved           : {time_saved}s ({time_saved/60:.1f} min)\n")
            f.write(f"  Time reduction       : {reduction_pct:.1f}%\n")
        
        print(f"\n Analysis saved to: {output_file}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Analyze MI optimization speedup")
    parser.add_argument("--baseline", type=str, required=True,
                       help="Baseline timing summary file")
    parser.add_argument("--optimized", type=str, required=True,
                       help="Optimized timing summary file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output analysis file (optional)")
    
    args = parser.parse_args()
    
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        return
    
    if not Path(args.optimized).exists():
        print(f"Error: Optimized file not found: {args.optimized}")
        return
    
    analyze_speedup(args.baseline, args.optimized, args.output)

if __name__ == "__main__":
    main()