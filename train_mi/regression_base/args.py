"""
Configuration and argument parsing for GFLOPS regression model.
All hyperparameters and experimental settings are defined here.
"""

import argparse


def get_args():
    """Parse command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description='XGBoost Regression for GFLOPS Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ====================== Data Configuration ======================
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--csv_path', type=str, default='gflops_data_256_and_128.csv',
                           help='Path to input CSV file')
    data_group.add_argument('--test_size', type=float, default=0.1,
                           help='Ratio of test set split')
    data_group.add_argument('--valid_size', type=float, default=0.1,
                           help='Ratio of validation set split from training data')
    data_group.add_argument('--random_state', type=int, default=42,
                           help='Random seed for reproducibility')
    
    # ====================== Feature Engineering ======================
    feature_group = parser.add_argument_group('Feature Engineering')
    feature_group.add_argument('--use_standardization', action='store_true', default=False,
                              help='Apply StandardScaler to features')
    feature_group.add_argument('--no_standardization', dest='use_standardization', 
                              action='store_false',
                              help='Disable feature standardization')
    feature_group.add_argument('--std_problem_size', action='store_true', default=False,
                              help='Standardize problem size (m, n, k)')
    feature_group.add_argument('--std_wave_params', action='store_true', default=False,
                              help='Standardize wave parameters')
    feature_group.add_argument('--use_tile_type_encoding', action='store_true', default=False,
                              help='Use one-hot encoding for tile type (square vs non-square)')
    feature_group.add_argument('--remove_const_features', action='store_true', default=False,
                              help='Remove constant features (B, MIBlockM)')
    feature_group.add_argument('--feature_extension', action='store_true', default=False,
                              help='Add extended features like gflops_norm')
    
    # ====================== XGBoost Hyperparameters ======================
    xgb_group = parser.add_argument_group('XGBoost Hyperparameters')
    xgb_group.add_argument('--objective', type=str, default='reg:squarederror',
                          help='XGBoost objective function')
    xgb_group.add_argument('--eval_metric', type=str, default='rmse',
                          help='Evaluation metric')
    xgb_group.add_argument('--eta', type=float, default=0.05,
                          help='Learning rate')
    xgb_group.add_argument('--max_depth', type=int, default=16,
                          help='Maximum tree depth')
    xgb_group.add_argument('--min_child_weight', type=float, default=5,
                          help='Minimum sum of instance weight in a child')
    xgb_group.add_argument('--subsample', type=float, default=0.8,
                          help='Subsample ratio of training instances')
    xgb_group.add_argument('--colsample_bytree', type=float, default=0.8,
                          help='Subsample ratio of columns when constructing each tree')
    xgb_group.add_argument('--gamma', type=float, default=0.1,
                          help='Minimum loss reduction for split')
    xgb_group.add_argument('--lambda_reg', type=float, default=1.0,
                          help='L2 regularization term on weights')
    xgb_group.add_argument('--alpha', type=float, default=0.5,
                          help='L1 regularization term on weights')
    xgb_group.add_argument('--booster', type=str, default='dart',
                          choices=['gbtree', 'dart', 'gblinear'],
                          help='Type of booster')
    xgb_group.add_argument('--rate_drop', type=float, default=0.1,
                          help='Dropout rate for DART booster')
    xgb_group.add_argument('--skip_drop', type=float, default=0.5,
                          help='Skip dropout probability for DART booster')
    xgb_group.add_argument('--tree_method', type=str, default='hist',
                          choices=['auto', 'exact', 'approx', 'hist'],
                          help='Tree construction algorithm')
    
    # ====================== Training Configuration ======================
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--num_boost_round', type=int, default=300,
                            help='Number of boosting rounds')
    train_group.add_argument('--early_stopping_rounds', type=int, default=200,
                            help='Early stopping rounds')
    train_group.add_argument('--verbose_eval', type=int, default=50,
                            help='Print evaluation every N rounds')
    train_group.add_argument('--callback_period', type=int, default=50,
                            help='Validation callback evaluation period')
    
    # ====================== Output Configuration ======================
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--model_path', type=str, default='gflops_final_full.xgb',
                             help='Path to save trained model')
    output_group.add_argument('--log_dir', type=str, default='logs',
                             help='Directory for log files')
    output_group.add_argument('--plot_dir', type=str, default='plots',
                             help='Directory for plots')
    output_group.add_argument('--plot_data', action='store_true', default=True,
                             help='Generate evaluation plots')
    output_group.add_argument('--no_plot', dest='plot_data', action='store_false',
                             help='Disable plot generation')
    output_group.add_argument('--exp_name', type=str, default='',
                             help='Experiment name for logging (optional)')
    
    args = parser.parse_args()
    return args


def get_xgb_params(args):
    """Convert parsed arguments to XGBoost parameter dictionary."""
    return {
        'objective': args.objective,
        'eval_metric': args.eval_metric,
        'eta': args.eta,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'gamma': args.gamma,
        'lambda': args.lambda_reg,
        'alpha': args.alpha,
        'booster': args.booster,
        'rate_drop': args.rate_drop,
        'skip_drop': args.skip_drop,
        'tree_method': args.tree_method,
        'seed': args.random_state
    }


def print_config(args):
    """Pretty print configuration for logging."""
    print("="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Data: {args.csv_path}")
    print(f"Random State: {args.random_state}")
    print(f"Test/Valid Split: {args.test_size}/{args.valid_size}")
    print(f"\nFeature Engineering:")
    print(f"  - Standardization: {args.use_standardization}")
    print(f"  - Std Problem Size: {args.std_problem_size}")
    print(f"  - Std Wave Params: {args.std_wave_params}")
    print(f"  - Tile Type Encoding: {args.use_tile_type_encoding}")
    print(f"  - Remove Const Features: {args.remove_const_features}")
    print(f"  - Feature Extension: {args.feature_extension}")
    print(f"\nXGBoost Parameters:")
    xgb_params = get_xgb_params(args)
    for key, val in xgb_params.items():
        print(f"  - {key}: {val}")
    print(f"\nTraining:")
    print(f"  - Num Boost Rounds: {args.num_boost_round}")
    print(f"  - Early Stopping: {args.early_stopping_rounds}")
    print(f"\nOutput:")
    print(f"  - Model Path: {args.model_path}")
    print(f"  - Log Dir: {args.log_dir}")
    print(f"  - Plot Dir: {args.plot_dir}")
    if args.exp_name:
        print(f"  - Experiment Name: {args.exp_name}")
    print("="*80)
