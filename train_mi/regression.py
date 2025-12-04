import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import sys

# ====================== Configuration ======================
CSV_PATH = 'gflops_data_256_and_128.csv'
MODEL_PATH = 'gflops_final_full.xgb'
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
RANDOM_STATE = 42 # 42
TOP_K_VALUES = [1, 2, 3, 5, 10, 15, 20]

PROBLEM_SIZE_COLS_mnk = ['m', 'n', 'k']
CANDIDATE_COLS = ['M', 'N', 'K', 'B', 'MIBlockM', 'WaveTileM', 'WaveTileN', 'WaveM', 'WaveN']
CONFIG_ID_COL = 'mi_idx'

FEATURE_EXTENSION = False
PLOT_DATA = True

# mse base
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 16,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'lambda': 1.0,
    'alpha': 0.5,
    'booster': 'dart',
    'rate_drop': 0.1,
    'skip_drop': 0.5,
    'tree_method': 'hist',
    'seed': RANDOM_STATE
}

# ====================== Logger ======================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# ====================== Valid Callback======================
class ValidRankingMetrics(xgb.callback.TrainingCallback):
    def __init__(self, valid_df, unique_mi_configs, period=50):
        self.valid_df = valid_df
        self.unique_mi_configs = unique_mi_configs
        self.period = period
        self.history = []

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period != 0 and epoch != 0:
            return False

        top1_correct = 0
        regrets = []
        true_best_ranks = []
        topk_contains = {k: 0 for k in TOP_K_VALUES}
        total_problems = 0

        for (m, n, k), group in self.valid_df.groupby(['m','n','k']):
            total_problems += 1
            
            true_best_gflops = group['gflops'].max()
            true_best_mi = int(group.loc[group['gflops'].idxmax(), 'mi_idx'])

            X_val = np.array([
                [m, n, k] + row[CANDIDATE_COLS].tolist()
                for _, row in self.unique_mi_configs.iterrows()
            ], dtype=np.float32)

            pred_scores = model.predict(xgb.DMatrix(X_val))
            pred_ranked = np.argsort(pred_scores)[::-1]
            pred_mi_list = [int(self.unique_mi_configs.iloc[i]['mi_idx']) for i in pred_ranked]

            # Top-1
            pred_best_mi = pred_mi_list[0]
            pred_best_gflops = group[group['mi_idx'] == pred_best_mi]['gflops'].iloc[0] if pred_best_mi in group['mi_idx'].values else 0
            if pred_best_mi == true_best_mi:
                top1_correct += 1

            # Regret & Rank
            regret = (true_best_gflops - pred_best_gflops) / (true_best_gflops + 1e-8)
            regrets.append(regret)
            true_rank = pred_mi_list.index(true_best_mi) + 1 if true_best_mi in pred_mi_list else 129
            true_best_ranks.append(true_rank)

            # Top-k Recall
            for k in TOP_K_VALUES:
                if true_best_mi in pred_mi_list[:k]:
                    topk_contains[k] += 1
            
            real_top_errors = {k: [] for k in [1,2,3,4,5]}
        
        for (m, n, k), group in self.valid_df.groupby(['m','n','k']):
            # ground truth ranking
            true_sorted = group.sort_values('gflops', ascending=False)
            true_mi_list = true_sorted[CONFIG_ID_COL].astype(int).values.tolist()
            true_gflops_list = true_sorted['gflops'].values.tolist()
            
            # predict all 128 mi in a problem size
            X_val = np.array([
                [m, n, k] + row[CANDIDATE_COLS].tolist()
                for _, row in self.unique_mi_configs.iterrows()
            ], dtype=np.float32)
            pred_scores = model.predict(xgb.DMatrix(X_val))
            mi_to_pred = {int(self.unique_mi_configs.iloc[i]['mi_idx']): pred_scores[i] 
                         for i in range(len(self.unique_mi_configs))}
            
            # Real Top-1~5  M scrivere
            for k in [1,2,3,4,5]:
                if len(true_mi_list) >= k:
                    for i in range(k):
                        mi = true_mi_list[i]
                        true_val = true_gflops_list[i]
                        pred_val = mi_to_pred.get(mi, 0)
                        mre = abs(pred_val - true_val) / true_val if true_val > 0 else 0
                        real_top_errors[k].append(mre)

        print("  RealTopErr →", end="")
        for k in [1,2,3,4,5]:
            if real_top_errors[k]:
                avg_mre = np.mean(real_top_errors[k])
                print(f" T{k}:{avg_mre:5.1%}", end="")
            else:
                print(f" T{k}:  N/A ", end="")
        print("  |", end="")

        top1_acc = top1_correct / total_problems
        mean_regret = np.mean(regrets)
        mean_rank = np.mean(true_best_ranks)

        print(f"\n[Valid @ {epoch:4d}] "
              f"Top-1 Acc: {top1_acc:6.1%} | "
              f"Regret: {mean_regret:6.2%} | "
              f"TrueBest Rank: {mean_rank:5.1f} | "
              f"R@5: {topk_contains[5]/total_problems:6.1%} | "
              f"R@10: {topk_contains[10]/total_problems:6.1%} | "
              f"R@20: {topk_contains[20]/total_problems:6.1%}")

        self.history.append({"epoch": epoch, "top1": top1_acc, "regret": mean_regret, "rank": mean_rank})
        return False

    def after_training(self, model):
        os.makedirs(LOG_DIR, exist_ok=True)
        curve_file = os.path.join(LOG_DIR, f"valid_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(curve_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nValid Curve:{curve_file}")
        return model



# ====================== data preprocessing functions ======================
def load_data():
    df = pd.read_csv(CSV_PATH)
    # unique configs: mi selections
    unique_mi_configs = df[['mi_idx'] + CANDIDATE_COLS].drop_duplicates().sort_values('mi_idx').reset_index(drop=True)
    print(f"Loaded {len(df):,} samples, {len(unique_mi_configs)} mi configs")
    return df, unique_mi_configs

def split_data(df):
    """
    Spliting data by a problem size (m,n,k)  and each problem size contains 128 mi config
    """
    
    # unique problem size m,n,k
    problems = df[PROBLEM_SIZE_COLS_mnk].drop_duplicates()
    train_prob, test_prob = train_test_split(problems, test_size=0.1, random_state=RANDOM_STATE)
    
    train_df = df.merge(train_prob, on=PROBLEM_SIZE_COLS_mnk, how='inner')
    test_df  = df.merge(test_prob,  on=PROBLEM_SIZE_COLS_mnk, how='inner')

    # train : valid = 9 : 1
    valid_prob = train_prob.sample(frac=0.1, random_state=RANDOM_STATE + 1)
    valid_df = train_df.merge(valid_prob, on=PROBLEM_SIZE_COLS_mnk, how='inner')
    train_df = train_df.drop(valid_df.index)

    print(f"Train: {len(train_df)//128} probs, Valid: {len(valid_df)//128}, Test: {len(test_df)//128}")
    return train_df, valid_df, test_df

def prepare_data(df, unique_mi_configs, extra_feature_cols=None):
    config_map = {cfg['mi_idx']: cfg for _, cfg in unique_mi_configs.iterrows()}
    
    X, y = [], []
    for _, row in df.iterrows():
        mi = row['mi_idx']
        if mi in config_map:
            cfg = config_map[mi]
            # get the (m, n, k) of this row
            features = [row['m'], row['n'], row['k']]
            # append the extended mi config ['M', 'N', 'K', 'B', 'MIBlockM', 'WaveTileM', 'WaveTileN', 'WaveM', 'WaveN']
            for c in CANDIDATE_COLS:
                features.append(cfg[c])
                
            if extra_feature_cols is not None:
                for ext_col in extra_feature_cols:
                    features.append(row[ext_col])
            
            X.append(features)
            y.append(row['gflops'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def feature_extension(df):
    """
    Add an col in df:
    1. with in one problem size 'gflops_norm' = (curr_gflops - min_gflops) / (max_gflops - min_gflops)
    2. Add other features...
    """
    
    # for every problem (m,n,k) calculate min_g、max_g
    stats = df.groupby(['m', 'n', 'k'])['gflops'].agg(
        min_g='min',
        max_g='max'
    ).reset_index()
    # merge the max min of each problem size min_g / max_g
    df = df.merge(stats, on=['m', 'n', 'k'], how='left')
    df['gflops_norm'] = (df['gflops'] - df['min_g']) / (df['max_g'] - df['min_g'] + 1e-8)
    df = df.drop(columns=['min_g', 'max_g'])

    return df

def plot_eval(results_df,
              rank_of_true_best_list,
              topk_regret,
              real_topk_mean_rank,
              mre_list,
              plot_dir):
    
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Top-1 Regret distribution
    axes[0,0].hist(results_df['regret'], bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[0,0].axvline(0.05, color='red', linestyle='--')
    axes[0,0].set_title('Top-1 Regret Distribution')
    axes[0,0].grid(alpha=0.3)

    # Pred vs True GFLOPS
    axes[0,1].scatter(results_df['true_gflops'], results_df['pred_gflops'], alpha=0.6)
    maxv = max(results_df['true_gflops'].max(), results_df['pred_gflops'].max()) * 1.05
    axes[0,1].plot([0, maxv], [0, maxv], 'k--')
    axes[0,1].set_title('Predicted vs True Best GFLOPS')
    axes[0,1].grid(alpha=0.3)

    # Rank of True Best
    axes[0,2].hist(rank_of_true_best_list, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,2].set_title('Rank of True Best')
    axes[0,2].grid(alpha=0.3)

    # Mean Top-k Regret
    axes[1,0].bar(TOP_K_VALUES, [np.mean(topk_regret[k]) for k in TOP_K_VALUES])
    axes[1,0].set_title('Mean Top-k Regret')
    axes[1,0].grid(alpha=0.3)

    # Mean Rank of Real Top-k
    axes[1,1].bar(TOP_K_VALUES, [np.mean(real_topk_mean_rank[k]) for k in TOP_K_VALUES], color='orange')
    axes[1,1].set_title('Mean Rank of Real Top-k')
    axes[1,1].grid(alpha=0.3)

    # Relative Prediction Error distribution
    axes[1,2].hist(mre_list, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,2].set_title('Relative Prediction Error Distribution')
    axes[1,2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'eval_full_v2.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nsave plot {plot_path}")


def evaluate(bst, test_df, unique_mi_configs, plot=True):
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    topk_correct = {k: 0 for k in TOP_K_VALUES}
    topk_regret = {k: [] for k in TOP_K_VALUES}
    real_topk_mean_rank = {k: [] for k in TOP_K_VALUES}
    total = 0
    regrets = []
    results = []
    rank_of_true_best_list = []
    error_lists = {'mae': [], 'rmse': [], 'mre': []}
    
    n_configs = len(unique_mi_configs)
    print(f"\nEvaluating on {len(test_df.groupby(PROBLEM_SIZE_COLS_mnk))} test problems, each has up to {n_configs} configs...")
    
    for (m, n, k), group in test_df.groupby(PROBLEM_SIZE_COLS_mnk):
        total += 1
        group = group.reset_index(drop=True)
        
        # Ground truth
        actual_gflops = group['gflops'].values
        true_best_gflops = actual_gflops.max()
        
        # ranking (local)
        true_sorted_local_idx = np.argsort(actual_gflops)[::-1]
        true_mi_indices = group.iloc[true_sorted_local_idx][CONFIG_ID_COL].astype(int).values
        
        true_best_local_idx = true_sorted_local_idx[0]
        true_mi_idx = int(group.iloc[true_best_local_idx][CONFIG_ID_COL])
        
        mi_to_actual = dict(zip(group[CONFIG_ID_COL].astype(int), group['gflops']))
        
        # predict all 128
        X_test = []
        cfg_to_global_idx = {}
        for idx, cfg in unique_mi_configs.iterrows():
            mi_idx = int(cfg[CONFIG_ID_COL])
            features = [m, n, k] + [cfg[col] for col in CANDIDATE_COLS]
            X_test.append(features)
            cfg_to_global_idx[mi_idx] = idx
        X_test = np.array(X_test, dtype=np.float32)
        dtest = xgb.DMatrix(X_test)
        pred_scores = bst.predict(dtest)
        
        pred_sorted_global_idx = np.argsort(pred_scores)[::-1]
        pred_mi_indices = [int(unique_mi_configs.iloc[i][CONFIG_ID_COL]) for i in pred_sorted_global_idx]
        
        # errors on measured configs
        for global_idx in range(n_configs):
            mi_idx = int(unique_mi_configs.iloc[global_idx][CONFIG_ID_COL])
            if mi_idx in mi_to_actual:
                true_val = mi_to_actual[mi_idx]
                pred_val = pred_scores[global_idx]
                abs_err = abs(pred_val - true_val)
                rel_err = abs_err / true_val if true_val > 0 else 0
                error_lists['mae'].append(abs_err)
                error_lists['rmse'].append(abs_err ** 2)
                error_lists['mre'].append(rel_err)
        
        # Top-1 regret
        pred_best_mi = pred_mi_indices[0]
        pred_gflops = mi_to_actual.get(pred_best_mi, 0)
        regret = (true_best_gflops - pred_gflops) / true_best_gflops if true_best_gflops > 0 else 0
        regrets.append(regret)
        
        # rank of true best
        if true_mi_idx in cfg_to_global_idx:
            true_global_idx = cfg_to_global_idx[true_mi_idx]
            true_best_rank = np.where(pred_sorted_global_idx == true_global_idx)[0][0] + 1
        else:
            true_best_rank = n_configs + 1
        rank_of_true_best_list.append(true_best_rank)
        
        # Top-k accuracy
        for kk in TOP_K_VALUES:
            if true_mi_idx in pred_mi_indices[:kk]:
                topk_correct[kk] += 1
        
        # Top-k regret
        for kk in TOP_K_VALUES:
            topk_mis = pred_mi_indices[:kk]
            actual_in_topk = [mi_to_actual.get(mi, 0) for mi in topk_mis]
            best_in_topk = max(actual_in_topk)
            tk_regret = max(0, (true_best_gflops - best_in_topk) / true_best_gflops)
            topk_regret[kk].append(tk_regret)
        
        # Real top-k mean predicted rank
        for kk in TOP_K_VALUES:
            real_topk_mi = true_mi_indices[:kk]
            ranks = []
            for mi in real_topk_mi:
                if mi in pred_mi_indices:
                    rank = pred_mi_indices.index(mi) + 1
                else:
                    rank = n_configs + 1
                ranks.append(rank)
            real_topk_mean_rank[kk].append(np.mean(ranks))
        
        results.append({
            'm': m, 'n': n, 'k': k,
            'true_gflops': true_best_gflops,
            'pred_gflops': pred_gflops,
            'regret': regret,
            'true_best_rank': true_best_rank
        })
        
        if total <= 3:
            print(f"  {m}x{n}x{k}: true={true_best_gflops:.0f}, pred={pred_gflops:.0f}, regret={regret:.1%}, rank={true_best_rank}")
    
    # aggregate
    mae = np.mean(error_lists['mae'])
    rmse = np.sqrt(np.mean(error_lists['rmse']))
    mre = np.mean(error_lists['mre'])
    
    # metrics & plot_data
    metrics = {
        'topk_accuracy': {k: topk_correct[k]/total for k in TOP_K_VALUES},
        'regret_mean': float(np.mean(regrets)),
        'true_best_rank_mean': float(np.mean(rank_of_true_best_list)),
        'prediction_errors': {'mae': float(mae), 'rmse': float(rmse), 'mre': float(mre)},
        'topk_regret_mean': {k: float(np.mean(topk_regret[k])) for k in TOP_K_VALUES},
        'real_topk_mean_rank': {k: float(np.mean(real_topk_mean_rank[k])) for k in TOP_K_VALUES},
    }
    
    if plot:
        plot_eval(
            results_df=pd.DataFrame(results),
            rank_of_true_best_list=rank_of_true_best_list,
            topk_regret=topk_regret,
            real_topk_mean_rank=real_topk_mean_rank,
            mre_list=error_lists['mre'],
            plot_dir=PLOT_DIR,
        )
    
    return metrics


# ====================== Main ======================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f"training_FULL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_file)
    print("="*80)

    df, unique_mi_configs = load_data()
    
    extra_features = None
    if FEATURE_EXTENSION:
        df = feature_extension(df)
        # extend extra features to the list for future
        extra_features = ['gflops_norm']

    train_df, valid_df, test_df = split_data(df)

    X_train, y_train = prepare_data(train_df, unique_mi_configs, extra_features)
    X_valid, y_valid = prepare_data(valid_df, unique_mi_configs, extra_features)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain,'train'), (dvalid,'valid')],
        early_stopping_rounds=200,
        callbacks=[ValidRankingMetrics(valid_df, unique_mi_configs, period=50)],
        # callbacks=[ValidRankingMetrics(test_df, unique_mi_configs, period=50)],
        verbose_eval=50
    )

    print("\nstart evaluation")
    results = evaluate(bst, test_df, unique_mi_configs, PLOT_DATA)
    # results = evaluate(bst, valid_df, unique_mi_configs)

    
    
    bst.save_model(MODEL_PATH)
    summary_file = os.path.join(LOG_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump({"params": XGB_PARAMS, "results": results}, f, indent=2)

    print(f"\n{MODEL_PATH} summary_file:{summary_file}   log_file:{log_file}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__