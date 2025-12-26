# Tensile MI Configuration Optimizer

XGBoost-based optimizer for Tensile Matrix Instruction (MI) configurations. This tool uses machine learning to predict GFLOPS performance and select top-K MI configurations for each problem size, significantly reducing tuning time.

## Features

- **ML-Based Prediction**: Uses XGBoost model to predict GFLOPS for all MI configurations
- **Per-Problem Optimization**: Generates optimized YAML for each problem size independently
- **Configurable Top-K**: Select top N configurations based on predicted performance
- **Format Preservation**: Maintains original YAML structure, comments, and formatting
- **Time Tracking**: Reports prediction time and optimization metrics

## Prerequisites

```bash
# Install required packages
pip install ruamel.yaml xgboost pandas numpy
```

## Quick Start

### 1. Grant Execution Permission

```bash
chmod +x run_optimize.sh
```

### 2. Run Optimization (Default: Top-20)

```bash
./run_optimize.sh
```

### 3. Custom Configuration

```bash
# Specify different Top-K value
python3 predict_mi_and_config_gen.py \
    --model ./model/G3_1500round.xgb \
    --yaml ./speedup_test_logic.yaml \
    --output-dir ./optimal_configs \
    --top-k 15

# Customize all parameters
python3 predict_mi_and_config_gen.py \
    --model ../model/G3_1500round.xgb \
    --yaml ./my_config.yaml \
    --output-dir ./my_output \
    --top-k 10
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `../model/G3_1500round.xgb` | Path to XGBoost model |
| `--yaml` | `./speedup_test_logic.yaml` | Input YAML configuration |
| `--output-dir` | `./optimal_configs` | Output directory for optimized YAMLs |
| `--top-k` | `20` | Number of top MI configurations to keep |

## Workflow

1. **Load Model**: Loads pre-trained XGBoost model for GFLOPS prediction
2. **Extract Configurations**: Parses input YAML to extract MI configs and problem sizes
3. **Predict Performance**: For each problem size, predicts GFLOPS for all 128 MI configurations
4. **Select Top-K**: Ranks MI configs by predicted GFLOPS and selects top K
5. **Generate YAML**: Creates optimized YAML with only top-K MI configs for each problem
6. **Generate Report**: Produces summary report with optimization metrics

## Input YAML Format

Your input YAML should contain:

```yaml
BenchmarkProblems:
  -
    - # ProblemType
      OperationType: GEMM
      DataType: h
      # ... other problem type parameters
    - # BenchmarkProblemSizeGroup
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 16, 1, 1, 1, 1, 4, 1] #0
          - [16, 16, 16, 1, 1, 2, 1, 4, 1] #1
          # ... 128 MI configurations total
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [576, 576, 1, 4096]
          - Exact: [4288, 4288, 1, 4096]
          # ... multiple problem sizes
```

## Output Structure

```
optimal_configs/
├── optimal_m576_n576_b1_k4096_top20.yaml
├── optimal_m4288_n4288_b1_k4096_top20.yaml
├── optimal_m8128_n8128_b1_k4096_top20.yaml
└── optimization_summary_top20.txt
```

### Output YAML Format

Each optimized YAML contains:
- **Same structure** as input YAML
- **Only top-K MI configurations** (e.g., 20 out of 128)
- **Single problem size** per file
- **Preserved comments** and formatting

Example:

```yaml
ForkParameters:
  - MatrixInstruction:
    - [32, 32, 8, 1, 1, 2, 1, 1, 4] #51
    - [16, 16, 16, 1, 1, 1, 5, 2, 2] #68
    - [16, 16, 16, 1, 1, 3, 2, 1, 4] #50
    # ... top 20 configurations
BenchmarkFinalParameters:
  - ProblemSizes:
    - Exact: [576, 576, 1, 4096]  # Single problem size
```

## Optimization Summary Report

The summary report (`optimization_summary_top20.txt`) includes:

- Model and configuration paths
- Total problem sizes processed
- Execution time metrics
- MI configuration reduction rate
- Per-problem details:
  - Original vs optimized MI count
  - Predicted GFLOPS range
  - Top-5 MI indices
  - Prediction time

Example:

```
Problem #0: M=576, N=576, B=1, K=4096
  MI configs: 128 → 20 (reduced 84.4%)
  Predicted GFLOPS: 245.67 (top-1) ~ 198.32 (top-20)
  Top-5 MI indices: [51, 68, 50, 52, 49]
  Prediction time: 0.0234s
```

## Performance Benefits

- **Tuning Time Reduction**: ~84% fewer configurations to test (128 → 20)
- **Fast Prediction**: < 0.1s per problem size
- **Accuracy**: Top-20 captures best configurations with high probability
- **Scalability**: Process multiple problem sizes in parallel

## Model Features

The XGBoost model uses these features for prediction:

| Feature | Description |
|---------|-------------|
| `m`, `n`, `k` | Problem size dimensions |
| `M`, `N`, `K` | MI block dimensions |
| `B` | Batch size |
| `MIBlockM` | MI block M parameter |
| `WaveTileM`, `WaveTileN` | Wave tile sizes |
| `WaveM`, `WaveN` | Wave dimensions |

## Troubleshooting

### Model File Not Found
```bash
Error: Model file not found: ../model/G3_1500round.xgb
```
**Solution**: Check model path and ensure file exists

### YAML Parse Error
```bash
Error: Unable to extract MI configs or Problem Sizes
```
**Solution**: Verify input YAML format matches expected structure

### Import Error
```bash
ModuleNotFoundError: No module named 'ruamel'
```
**Solution**: Install dependencies:
```bash
pip install ruamel.yaml xgboost pandas numpy
```

## Advanced Usage

### Batch Processing Multiple YAMLs

```bash
#!/bin/bash
for yaml_file in configs/*.yaml; do
    python3 predict_mi_and_config_gen.py \
        --model ../model/G3_1500round.xgb \
        --yaml "$yaml_file" \
        --output-dir "./optimal_$(basename $yaml_file .yaml)" \
        --top-k 20
done
```

### Custom Top-K Values

Test different Top-K values to find optimal trade-off:

```bash
for k in 10 15 20 25 30; do
    python3 predict_mi_and_config_gen.py \
        --model ../model/G3_1500round.xgb \
        --yaml ./speedup_test_logic.yaml \
        --output-dir "./optimal_top${k}" \
        --top-k $k
done
```

## Files

- `predict_mi_and_config_gen.py`: Main optimization script
- `run_optimize.sh`: Convenience shell script
- `speedup_test_logic.yaml`: Example input configuration

## License

Internal AMD tool for Tensile optimization workflow.

## Contact

For questions or issues, contact the ML tuning team.