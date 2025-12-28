#!/usr/bin/env python3
"""
GFLOPS Ranker v6 - Fixed for Continuous Labels
- Converts continuous GFLOPS to integer relevance ranks for XGBoost ranking
- Within each problem, ranks candidates from 0 (worst) to N-1 (best)
- Uses rank:pairwise objective which learns relative ordering
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import rankdata
import logging
from datetime import datetime
import sys


# ====================== Configuration ======================
CSV_PATH = 'gflops_data_output.csv'
MODEL_PATH = 'gflops_ranker_v6.xgb'
CONFIGS_PATH = 'mi_configs_128.json'
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
RANDOM_STATE = 42
TEST_SIZE = 0.1
TOP_K_VALUES = [1, 3, 5, 10]

# ========== NEW: Train/Test Split Strategy ==========
SPLIT_STRATEGY = 'use_all_train'  # Options: 'standard', 'use_all_train', 'no_test'
# - 'standard': Traditional train/test split (no data leakage)
# - 'use_all_train': Use all data for training, sample TEST_SIZE for evaluation
# - 'no_test': Use all data for training only (no evaluation)


PROBLEM_COLS = ['m', 'n', 'k']
CANDIDATE_COLS = ['M', 'N', 'K', 'B', 'MIBlockM', 'WaveTileM', 'WaveTileN', 'WaveM', 'WaveN']
CONFIG_ID_COL = 'mi_idx'
LABEL_COL = 'gflops'

XGB_PARAMS = {
    'objective': 'rank:pairwise',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.1,
    'eval_metric': 'ndcg',
    'ndcg_exp_gain': False,
    'tree_method': 'hist',
    'seed': RANDOM_STATE
}

# ========================================================
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # timestamp and filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f'training_{timestamp}.log')
    
    # log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # clean handlers
    logger.handlers = []
    
    # file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)
    
    return logger, log_file

def load_and_extract(csv_path):
    """Load CSV and extract unique configurations"""
    df = pd.read_csv(csv_path)
    print(f"Loaded data: {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"GFLOPS range: {df['gflops'].min():.1f} - {df['gflops'].max():.1f}")
    print(f"Problem sizes - m: {df['m'].min()}-{df['m'].max()}, n: {df['n'].min()}-{df['n'].max()}, k: {df['k'].min()}-{df['k'].max()}")
    
    config_cols = [CONFIG_ID_COL] + CANDIDATE_COLS
    unique_configs = df[config_cols].drop_duplicates().sort_values(CONFIG_ID_COL).reset_index(drop=True)
    print(f"Found {len(unique_configs)} unique MI configurations")
    
    unique_configs.to_json(CONFIGS_PATH, orient='records', indent=2)
    return df, unique_configs

def build_features_row(m, n, k, cfg):
    """Build feature vector with ratios and log transforms"""
    features = [float(m), float(n), float(k)]
    
    for col in CANDIDATE_COLS:
        features.append(float(cfg[col]))
    
    eps = 1e-8
    features.extend([
        cfg['M'] / (m + eps),
        cfg['N'] / (n + eps),
        cfg['K'] / (k + eps),
        np.log1p(m), np.log1p(n), np.log1p(k),
        np.log1p(cfg['M']), np.log1p(cfg['N']), np.log1p(cfg['K'])
    ])
    
    return np.array(features, dtype=np.float32)

def prepare_data_with_ranks(df, unique_configs):
    """
    Prepare data with INTEGER relevance ranks for XGBoost ranking.
    For each problem group, converts GFLOPS to ranks: 0 (worst) to N-1 (best).
    This satisfies XGBoost's requirement for integer labels in ranking.
    """
    X_list, y_list, group_list = [], [], []
    y_gflops_list = []  # Keep original GFLOPS for evaluation
    
    print(f"Preparing data with integer relevance ranks...")
    
    problem_groups = df.groupby(PROBLEM_COLS)
    print(f"Found {len(problem_groups)} unique problems")
    
    for (m, n, k), group_df in problem_groups:
        num_candidates = len(group_df)
        group_list.append(num_candidates)
        
        # Convert GFLOPS to integer ranks within this group
        # Higher GFLOPS -> Higher rank (better relevance)
        # rankdata with method='ordinal' ensures unique integer ranks
        gflops_values = group_df[LABEL_COL].values
        relevance_ranks = rankdata(gflops_values, method='ordinal').astype(int) - 1  # 0-indexed
        
        # Build features for each candidate
        for idx, (_, row) in enumerate(group_df.iterrows()):
            mi_idx = row[CONFIG_ID_COL]
            cfg_row = unique_configs[unique_configs[CONFIG_ID_COL] == mi_idx].iloc[0]
            
            X_list.append(build_features_row(m, n, k, cfg_row))
            y_list.append(relevance_ranks[idx])  # Integer rank (0 to N-1)
            y_gflops_list.append(row[LABEL_COL])  # Original GFLOPS for later use
    
    print(f"Prepared {len(X_list)} samples across {len(group_list)} problems")
    print(f"Average candidates per problem: {np.mean(group_list):.1f}")
    print(f"Relevance ranks range: {min(y_list)} - {max(y_list)} (integers)")
    
    # Validation
    if sum(group_list) != len(X_list):
        raise ValueError(f"Group size mismatch: {sum(group_list)} != {len(X_list)}")
    
    X = np.array(X_list, dtype=np.float32)
    # y = np.array(y_list, dtype=np.float32)  # XGBoost can handle float representation of integers
    y = np.array(y_list, dtype=np.int64)
    
    groups = np.array(group_list)
    y_gflops = np.array(y_gflops_list, dtype=np.float32)
    
    print(f"Feature matrix: {X.shape}")
    print(f"Label stats - rank mean: {y.mean():.1f}, original GFLOPS mean: {y_gflops.mean():.1f}")
    
    return X, y, groups, y_gflops

def split_by_problem(df):
    """Split by unique problems (no data leakage)"""
    unique_problems = df[PROBLEM_COLS].drop_duplicates()
    print(f"Total unique problems: {len(unique_problems)}")
    
    train_problems, test_problems = train_test_split(
        unique_problems, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    
    print(f"Train: {len(train_problems)} problems ({len(train_problems)/len(unique_problems)*100:.1f}%)")
    print(f"Test: {len(test_problems)} problems ({len(test_problems)/len(unique_problems)*100:.1f}%)")
    
    train_problems = train_problems.copy()
    test_problems = test_problems.copy()
    train_problems['_split'] = 'train'
    test_problems['_split'] = 'test'
    all_problems = pd.concat([train_problems, test_problems], ignore_index=True)
    
    df_merged = df.merge(all_problems, on=PROBLEM_COLS, how='left')
    train_df = df_merged[df_merged['_split'] == 'train'].drop('_split', axis=1)
    test_df = df_merged[df_merged['_split'] == 'test'].drop('_split', axis=1)
    
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    return train_df, test_df

# def split_by_problem(df):
#     """
#     Use ALL data for training, then sample 10% for testing/evaluation.
#     This maximizes training data while still allowing performance evaluation.
#     """
#     unique_problems = df[PROBLEM_COLS].drop_duplicates()
#     print(f"Total unique problems: {len(unique_problems)}")
    
#     # 從所有問題中隨機抽取10%用於測試評估
#     test_problems = unique_problems.sample(
#         frac=TEST_SIZE,  # 0.1 = 10%
#         random_state=RANDOM_STATE
#     )
    
#     print(f"Train: {len(unique_problems)} problems (100% - using all data)")
#     print(f"Test (for evaluation): {len(test_problems)} problems ({TEST_SIZE*100:.1f}% - sampled from train)")
    
#     # 訓練集 = 全部數據
#     train_df = df.copy()
    
#     # 測試集 = 從全部數據中抽取的子集
#     test_problems = test_problems.copy()
#     test_problems['_test'] = True
#     df_merged = df.merge(test_problems, on=PROBLEM_COLS, how='left')
#     test_df = df_merged[df_merged['_test'] == True].drop('_test', axis=1)
    
#     print(f"Train samples: {len(train_df):,} (100%)")
#     print(f"Test samples: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
#     print(f"  Note: Test set is a subset of training set (for evaluation only)")
    
#     return train_df, test_df



def train_model(X_train, y_train, group_train, X_val=None, y_val=None, group_val=None):
    """
    Train XGBoost ranker with integer relevance labels.
    y_train contains integer ranks (0=worst, N-1=best within each group).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train.tolist())
    
    evals = [(dtrain, 'train')]
    
    if X_val is not None and len(group_val) > 0:
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(group_val.tolist())
        evals.append((dval, 'val'))
        print(f"Validation: {len(X_val)} samples, {len(group_val)} problems")
    
    print("\nTraining XGBoost Ranker with integer relevance labels...")
    print(f"Training: {len(group_train)} problems, {len(X_train)} candidates")
    print(f"Features: {X_train.shape[1]}")
    print(f"Label type: {y_train.dtype}, range: [{y_train.min():.0f}, {y_train.max():.0f}]")
    
    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    print(f"\nTraining complete! Best iteration: {bst.best_iteration}")
    return bst

def evaluate_with_topk(bst, test_df, unique_configs):
    """
    Evaluate with original GFLOPS values (not ranks).
    Model predicts ranking scores, we compare against true best GFLOPS.
    Fixed formatting for float/integer conversion in print statements.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    topk_correct = {k: 0 for k in TOP_K_VALUES}
    total_problems = 0
    regrets = []
    regret_5pct_count = 0
    plot_data = []
    
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*80}")
    
    test_groups = test_df.groupby(PROBLEM_COLS)
    print(f"Testing {len(test_groups)} problems...\n")
    
    for (m, n, k), group in test_groups:
        total_problems += 1
        
        # True best based on GFLOPS
        true_max_gflops = group['gflops'].max()
        true_best_row = group.loc[group['gflops'].idxmax()]
        true_best_mi_idx = int(true_best_row[CONFIG_ID_COL])  # Convert to int
        true_best_config = unique_configs[unique_configs[CONFIG_ID_COL] == true_best_mi_idx].iloc[0]
        true_best_M = int(true_best_config['M'])  # Convert to int
        
        # Predict on all 128 configs
        X_test = np.array([build_features_row(m, n, k, cfg) for _, cfg in unique_configs.iterrows()], dtype=np.float32)
        dtest = xgb.DMatrix(X_test)
        pred_scores = bst.predict(dtest)
        
        # Higher score = better (rank:pairwise learns this from integer ranks)
        sorted_idx = np.argsort(pred_scores)[::-1]
        
        pred_best_idx = sorted_idx[0]
        pred_best_config = unique_configs.iloc[pred_best_idx]
        pred_best_mi_idx = int(pred_best_config[CONFIG_ID_COL])  # Convert to int
        pred_best_M = int(pred_best_config['M'])  # Convert to int
        
        # Get actual GFLOPS for predicted best (from test data if available)
        pred_gflops_row = group[group[CONFIG_ID_COL] == pred_best_mi_idx]
        if len(pred_gflops_row) > 0:
            pred_best_gflops = pred_gflops_row['gflops'].iloc[0]
        else:
            # Config not in test group (shouldn't happen with full 128 configs)
            pred_best_gflops = 0
        
        # Top-k accuracy
        for k_val in TOP_K_VALUES:
            top_k_mi_indices = [int(unique_configs.iloc[i][CONFIG_ID_COL]) for i in sorted_idx[:k_val]]
            if true_best_mi_idx in top_k_mi_indices:
                topk_correct[k_val] += 1
        
        # Regret calculation
        regret = (true_max_gflops - pred_best_gflops) / true_max_gflops if true_max_gflops > 0 else 0
        regrets.append(regret)
        
        if pred_best_gflops >= 0.95 * true_max_gflops:
            regret_5pct_count += 1
        
        plot_data.append({
            'm': m, 'n': n, 'k': k,
            'true_M': true_best_M,
            'pred_M': pred_best_M,
            'true_gflops': true_max_gflops,
            'pred_gflops': pred_best_gflops,
            'regret': regret,
            'true_mi_idx': true_best_mi_idx,
            'pred_mi_idx': pred_best_mi_idx
        })
        
        # Print first 5 problems - FIXED FORMATTING
        if total_problems <= 5:
            print(f"Problem {m:5d}x{n:5d}x{k:5d}")
            print(f"  True:  mi_idx={true_best_mi_idx:3d}, M={true_best_M:3d}, GFLOPS={true_max_gflops:.1f}")
            print(f"  Pred:  mi_idx={pred_best_mi_idx:3d}, M={pred_best_M:3d}, GFLOPS={pred_best_gflops:.1f} (regret {regret:.1%})")
    
    # Results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test problems: {total_problems}")
    
    for k in TOP_K_VALUES:
        acc = topk_correct[k] / total_problems
        print(f"Top-{k:2d} Accuracy: {acc:.3f} ({topk_correct[k]:3d}/{total_problems:3d}) = {acc*100:.1f}%")
    
    print(f"\nPerformance:")
    print(f"  Avg Regret:    {np.mean(regrets):.2%}")
    print(f"  Median Regret: {np.median(regrets):.2%}")
    print(f"  Max Regret:    {np.max(regrets):.2%}")
    print(f"  ≤5% Regret:    {regret_5pct_count}/{total_problems} ({regret_5pct_count/total_problems*100:.1f}%)")
    print(f"  Perfect:       {sum(1 for r in regrets if r < 0.01)}/{total_problems} ({sum(1 for r in regrets if r < 0.01)/total_problems*100:.1f}%)")
    
    # Visualization
    if len(plot_data) > 1:
        df_plot = pd.DataFrame(plot_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GFLOPS Ranker Evaluation', fontsize=16)
        
        # 1. M comparison
        axes[0,0].scatter(df_plot['m'], df_plot['true_M'], label='True Best M', alpha=0.6, s=30, c='blue')
        axes[0,0].scatter(df_plot['m'], df_plot['pred_M'], label='Pred Best M', alpha=0.6, s=30, c='red')
        axes[0,0].set_xscale('log')
        axes[0,0].set_yscale('log')
        axes[0,0].set_xlabel('Problem m')
        axes[0,0].set_ylabel('Block Size M')
        axes[0,0].set_title('True vs Predicted M')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Regret histogram
        axes[0,1].hist(df_plot['regret'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0,1].axvline(0.05, color='red', linestyle='--', lw=2, label='5% threshold')
        axes[0,1].set_xlabel('Regret')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Regret Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. GFLOPS scatter
        axes[1,0].scatter(df_plot['true_gflops'], df_plot['pred_gflops'], alpha=0.6, s=30, c='green')
        min_g = min(df_plot['true_gflops'].min(), df_plot['pred_gflops'].min())
        max_g = max(df_plot['true_gflops'].max(), df_plot['pred_gflops'].max())
        axes[1,0].plot([min_g, max_g], [min_g, max_g], 'k--', lw=2, label='Perfect')
        axes[1,0].set_xlabel('True GFLOPS')
        axes[1,0].set_ylabel('Pred GFLOPS')
        axes[1,0].set_title('GFLOPS Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Top-k curve
        k_vals = sorted(TOP_K_VALUES)
        acc_vals = [topk_correct[k]/total_problems for k in k_vals]
        axes[1,1].plot(k_vals, acc_vals, marker='o', linewidth=2, markersize=8, c='purple')
        axes[1,1].set_xlabel('Top-k')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Top-k Accuracy')
        axes[1,1].axhline(0.95, color='red', linestyle='--', alpha=0.5)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'{PLOT_DIR}/evaluation.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {plot_file}")
        
        # Additional statistics
        print(f"\nDetailed Stats:")
        print(f"  Correlation (true_M, pred_M): {df_plot['true_M'].corr(df_plot['pred_M']):.3f}")
        print(f"  Correlation (true_gflops, pred_gflops): {df_plot['true_gflops'].corr(df_plot['pred_gflops']):.3f}")
        print(f"  Mean Absolute GFLOPS Error: {np.mean(np.abs(df_plot['pred_gflops'] - df_plot['true_gflops'])):.1f}")
        print(f"  Best case regret: {df_plot['regret'].min():.2%}")


# Main
if __name__ == "__main__":
    logger, log_file = setup_logging()
    try:
        print("GFLOPS Ranker with Integer Relevance Labels\n")
        logger.info("="*80)
        logger.info("🚀 GFLOPS Ranker with Integer Relevance Labels")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Configuration:")
        logger.info(f"  CSV: {CSV_PATH}")
        logger.info(f"  Model output: {MODEL_PATH}")
        logger.info(f"  Test size: {TEST_SIZE*100:.1f}%")
        logger.info(f"  Random seed: {RANDOM_STATE}")
        logger.info(f"  XGBoost params: {json.dumps(XGB_PARAMS, indent=2)}")
        
        # Load
        print("Step 1: Loading data...")
        logger.info("\n" + "="*80)
        logger.info("Step 1: Loading data...")
        logger.info("="*80)
        df, unique_configs = load_and_extract(CSV_PATH)
        
        # Split
        print("\nStep 2: Splitting...")
        logger.info("\n" + "="*80)
        logger.info("Step 2: Splitting...")
        logger.info("="*80)
        train_df, test_df = split_by_problem(df)
        
        # Prepare with rank conversion
        print("\nStep 3: Preparing features with integer ranks...")
        logger.info("\n" + "="*80)
        logger.info("Step 3: Preparing features with integer ranks...")
        logger.info("="*80)
        X_train, y_train, group_train, gflops_train = prepare_data_with_ranks(train_df, unique_configs)
        X_test, y_test, group_test, gflops_test = prepare_data_with_ranks(test_df, unique_configs)
        
        # Train
        logger.info("\n" + "="*80)
        logger.info("Step 4: Training...")
        logger.info("="*80)
        print("\nStep 4: Training...")
        bst = train_model(X_train, y_train, group_train, X_test, y_test, group_test)
        
        # Evaluate
        print("\nStep 5: Evaluating...")
        logger.info("\n" + "="*80)
        logger.info("Step 5: Evaluating...")
        logger.info("="*80)
        evaluate_with_topk(bst, test_df, unique_configs)
        
        # Save
        print("\nStep 6: Saving...")
        bst.save_model(MODEL_PATH)
        
        print(f"\n Complete!")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Configs: {CONFIGS_PATH}")
        
        summary_file = os.path.join(LOG_DIR, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'test_size': TEST_SIZE,
                'random_state': RANDOM_STATE,
                'xgb_params': XGB_PARAMS
            },
            'data': {
                'total_samples': len(df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'unique_configs': len(unique_configs)
            },
            'results': results if results else {}
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_file}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ Complete!")
        logger
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
