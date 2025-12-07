# GFLOPS Regression Model

A modular XGBoost-based regression framework for predicting GFLOPS (Giga Floating Point Operations Per Second) performance across different matrix multiplication configurations.

## Overview

This project provides a comprehensive pipeline for training and evaluating machine learning models that predict the performance of different kernel configurations for matrix multiplication operations. The model uses XGBoost with custom ranking metrics and supports extensive hyperparameter tuning.

## Project Structure

```
regression_base/
├── __init__.py          # Package initialization
├── args.py              # Command-line argument parsing and configuration
├── data_prepare.py      # Data loading and feature engineering
├── callbacks.py         # XGBoost training callbacks for validation metrics
├── train.py             # Model training logic
├── eval.py              # Model evaluation and visualization
├── utils.py             # Utility functions (logging, file I/O)
└── main.py              # Main entry point
```

## Quick Start


### Setup

1. **Clone or download the repository**
```bash
cd /path/to/ml_tuning/train_mi/regression_base
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "from regression import __version__; print('Regression package version:', __version__)"
```

### Basic Usage

Run with default parameters:
```bash
cd /ml_tuning/train_mi
python -m regression.main
```

### Custom Configuration

Specify custom parameters:
```bash
python -m regression_base.main \
    --csv_path gflops_data_256_and_128.csv \
    --model_path regression_base/models/my_model.xgb \
    --random_state 42 \
    --eta 0.05 \
    --max_depth 16 \
    --exp_name simple_test \
    --log_dir regression_base/logs/my_experiment
```

## Command-Line Arguments

### Data Configuration
- `--csv_path`: Path to input CSV file (default: `gflops_data_256_and_128.csv`)
- `--test_size`: Ratio of test set split (default: `0.1`)
- `--valid_size`: Ratio of validation set split (default: `0.1`)
- `--random_state`: Random seed for reproducibility (default: `42`)

### Feature Engineering
- `--use_standardization`: Apply StandardScaler to features (default: `True`)
- `--no_standardization`: Disable feature standardization
- `--std_problem_size`: Standardize problem size (m, n, k) (default: `False`)
- `--std_wave_params`: Standardize wave parameters (default: `True`)
- `--use_tile_type_encoding`: Use one-hot encoding for tile type (default: `True`)
- `--remove_const_features`: Remove constant features B, MIBlockM (default: `True`)
- `--feature_extension`: Add extended features like gflops_norm (default: `False`)

### XGBoost Hyperparameters
- `--objective`: XGBoost objective function (default: `reg:squarederror`)
- `--eval_metric`: Evaluation metric (default: `rmse`)
- `--eta`: Learning rate (default: `0.05`)
- `--max_depth`: Maximum tree depth (default: `16`)
- `--min_child_weight`: Minimum sum of instance weight in child (default: `5`)
- `--subsample`: Subsample ratio of training instances (default: `0.8`)
- `--colsample_bytree`: Subsample ratio of columns (default: `0.8`)
- `--gamma`: Minimum loss reduction for split (default: `0.1`)
- `--lambda_reg`: L2 regularization term (default: `1.0`)
- `--alpha`: L1 regularization term (default: `0.5`)
- `--booster`: Type of booster: `gbtree`, `dart`, `gblinear` (default: `dart`)
- `--rate_drop`: Dropout rate for DART booster (default: `0.1`)
- `--skip_drop`: Skip dropout probability for DART (default: `0.5`)
- `--tree_method`: Tree construction algorithm (default: `hist`)

### Training Configuration
- `--num_boost_round`: Number of boosting rounds (default: `300`)
- `--early_stopping_rounds`: Early stopping rounds (default: `200`)
- `--verbose_eval`: Print evaluation every N rounds (default: `50`)
- `--callback_period`: Validation callback evaluation period (default: `50`)

### Output Configuration
- `--model_path`: Path to save trained model (default: `gflops_final_full.xgb`)
- `--log_dir`: Directory for log files (default: `logs`)
- `--plot_dir`: Directory for plots (default: `plots`)
- `--plot_data`: Generate evaluation plots (default: `True`)
- `--no_plot`: Disable plot generation
- `--exp_name`: Experiment name for logging (optional)

## Running Experiments

### Example 1: Basic Experiment with Custom Learning Rate

```bash
python -m regression.main \
    --eta 0.1 \
    --exp_name high_lr_experiment
```

### Example 2: Deep Trees with High Regularization

```bash
python -m regression.main \
    --max_depth 20 \
    --lambda_reg 2.0 \
    --alpha 1.0 \
    --exp_name deep_regularized
```

### Example 3: Feature Engineering Experiment

```bash
python -m regression.main \
    --feature_extension \
    --std_problem_size \
    --no_standardization \
    --exp_name feature_eng_test
```

### Example 4: Fast Training with Fewer Rounds

```bash
python -m regression.main \
    --num_boost_round 100 \
    --early_stopping_rounds 50 \
    --eta 0.1 \
    --exp_name fast_test
```

## Evaluation Metrics

The model reports the following metrics:

### Primary Metrics
- **Top-1 Accuracy**: Percentage of times the model correctly predicts the best MI configuration
- **Top-1 Regret**: Performance loss from not selecting the optimal configuration
- **Mean Rank of True Best**: Average rank of the true best configuration in predictions out of 128 MI configurations

| Metric                     | Definition                                                                                 | Formula                                                                                         | Ideal Value | Interpretation |
|----------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------|----------------|
| **Top-1 Accuracy**         | Fraction of problems where the model's #1 predicted config is the true best               | `# correct top-1 problem size count/ total test problems count`                                                              | 100%        | Higher is better |
| **Top-1 Regret**           | Relative performance loss when using the model's top prediction instead of the oracle    | $$\text{Regret@1} = \frac{\text{TrueBest} - \text{PredBest}}{\text{TrueBest}}$$                | 0%          | Lower is better. 0% = perfect prediction |
| **Mean Rank of True Best** | Average rank (1–128) assigned by the model to the true best configuration                | Mean over all problems of (rank of true best; 129 if not in list)                               | 1.0         | Lower is better |


### Prediction Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MRE**: Mean Relative Error (%)

| Metric | Name                        | Formula                                              | Unit     |
|--------|-----------------------------|------------------------------------------------------|----------|
| **MAE**    | Mean Absolute Error         | mean(abs(pred − true))                                  | GFLOPS   |
| **RMSE**   | Root Mean Squared Error     | √mean((pred − true)²)                                | GFLOPS   |
| **MRE**    | Mean Relative Error         | mean(abs(pred − true) / true) × 100                     | %        |

### Top-k Metrics (Relaxed Ranking)

| Metric           | Definition                                                                                           | Ideal Value |
|------------------|------------------------------------------------------------------------------------------------------|-------------|
| **R@k (Recall@k)** | Percentage of problems where the true best config appears in the model’s top-k predictions           | 100%        |
| **Regret@k**       | Average performance loss if you pick the best config from the model’s top-k recommendations          | 0%          |
| **MeanRank@k**     | Average predicted rank of the ground-truth top-k configurations  | As low as possible |

### Real Top-K Error (Critical diagnostic metric)

For each test problem:
- Take the ground-truth top-1, top-2, ..., top-5 configurations to compare with their predictions(the same mi_idx)
- Compute their predicted GFLOPS and report the average relative error (MRE)

Printed as
`RealTopErr: T1: 8.2%  T2: 7.5%  T3: 6.9%  T4: 6.6%  T5: 6.3%`

### Recommended Priority of Metrics

| Priority | Metric                  | Target (strong model) | Practical Meaning |
|---------|-------------------------|------------------------|-------------------|
| ★★★★★   | Top-1 Regret            | ≤ 2.0%                 | Can you trust the #1 recommendation? |
| ★★★★★   | R@10                    | ≥ 95%                  | Is the optimum almost always in top-10? |
| ★★★★☆   | Regret@10               | ≤ 0.8%                 | Performance when doing top-10 + best-of |
| ★★★★☆   | Mean Rank of True Best  | ≤ 5.0                  | Intuitive ranking quality |
| ★★★☆☆   | RealTopErr T1           | ≤ 10%                  | Does the model know what "fast" really means? |
| ★★☆☆☆   | Top-1 Accuracy          | > 40% is already good  | Very hard to get high (128 classes) |


## Output Files

After training, the following files are generated:

### Model Files
- `<model_path>`: Trained XGBoost model (e.g., `gflops_final_full.xgb`)

### Log Files (in `logs/` directory)
- `training_<exp_name>_<timestamp>.log`: Complete training log
- `summary_<exp_name>_<timestamp>.json`: Experiment configuration and results
- `valid_curve_<timestamp>.json`: Validation metrics over training iterations

### Plots (in `plots/` directory)
- `eval_full_v2.png`: Comprehensive evaluation visualization with 6 subplots:
  1. Top-1 Regret Distribution
  2. Predicted vs True GFLOPS
  3. Rank of True Best Configuration
  4. Mean Top-k Regret
  5. Mean Rank of Real Top-k
  6. Relative Prediction Error Distribution

## Shell Script Examples

### Grid Search Over Learning Rates

Create `experiments/grid_search_lr.sh`:
```bash
#!/bin/bash

for lr in 0.01 0.05 0.1 0.2
do
    echo "Training with learning rate: $lr"
    python -m regression.main \
        --eta $lr \
        --exp_name lr_${lr} \
        --model_path models/model_lr_${lr}.xgb
done
```

### Experiment with Different Tree Depths

Create `experiments/depth_search.sh`:
```bash
#!/bin/bash

for depth in 8 12 16 20 24
do
    echo "Training with max_depth: $depth"
    python -m regression.main \
        --max_depth $depth \
        --exp_name depth_${depth} \
        --model_path models/model_depth_${depth}.xgb
done
```

### Compare Different Boosters

Create `experiments/booster_comparison.sh`:
```bash
#!/bin/bash

for booster in gbtree dart
do
    echo "Training with booster: $booster"
    python -m regression.main \
        --booster $booster \
        --exp_name booster_${booster} \
        --model_path models/model_${booster}.xgb
done
```

### Feature Engineering Ablation Study

Create `experiments/feature_ablation.sh`:
```bash
#!/bin/bash

# Baseline
python -m regression.main --exp_name baseline

# With feature extension
python -m regression.main --feature_extension --exp_name with_feature_ext

# With problem size standardization
python -m regression.main --std_problem_size --exp_name with_std_problem

# Without tile type encoding
python -m regression.main \
    --use_tile_type_encoding false \
    --exp_name no_tile_encoding

# All features combined
python -m regression.main \
    --feature_extension \
    --std_problem_size \
    --exp_name all_features
```

Make scripts executable:
```bash
chmod +x experiments/*.sh
```

Run experiments:
```bash
./experiments/grid_search_lr.sh
```

## 🏃 Batch Processing

Run multiple experiments in parallel:
```bash
# Create a simple batch script
cat > run_batch.sh << 'EOF'
#!/bin/bash

python -m regression.main --eta 0.05 --exp_name exp1 &
python -m regression.main --eta 0.1 --exp_name exp2 &
python -m regression.main --max_depth 20 --exp_name exp3 &
wait
echo "All experiments completed!"
EOF

chmod +x run_batch.sh
./run_batch.sh
```

## Tips for Experimentation

1. **Start with small experiments**: Use `--num_boost_round 50` for quick tests
2. **Name your experiments**: Always use `--exp_name` for easy tracking
3. **Monitor validation metrics**: Check the validation curve JSON files
4. **Compare results**: Use the summary JSON files to compare different configurations
5. **Disable plotting for speed**: Use `--no_plot` when running many experiments

## Troubleshooting

### Out of Memory
- Reduce `--max_depth`
- Decrease `--num_boost_round`
- Use `--tree_method hist` (already default)

### Slow Training
- Increase `--eta` (learning rate)
- Reduce `--num_boost_round`
- Decrease `--callback_period` for less frequent validation

### Poor Performance
- Increase `--max_depth`
- Reduce `--eta` and increase `--num_boost_round`
- Enable `--feature_extension`
- Try different `--booster` options

