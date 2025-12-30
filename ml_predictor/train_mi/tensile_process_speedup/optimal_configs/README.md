# Optimal MI Configurations

- This directory contains pruned YAML files with only the top-K predicted MI configurations for each problem size. 
- Optimal MI configurations selections vary by problem size, separate YAML files are generated to reduce tuning time (e.g., 128 → 20 configs).
- Detailed metrics and reduction stats are recorded in `optimization_summary_top{K}.txt`.