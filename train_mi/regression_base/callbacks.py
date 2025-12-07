"""
XGBoost callback functions for validation and ranking metrics.
Monitors model performance during training with custom metrics.
"""

import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime


# Top-k values for recall metrics
TOP_K_VALUES = [1, 2, 3, 5, 10, 15, 20]


class ValidRankingMetrics(xgb.callback.TrainingCallback):
    """
    Validation callback that computes ranking metrics during training.

    This is the new version that uses pre-prepared features for efficiency.
    Compatible with the "prepare-first-then-split" pipeline to avoid data leakage.
    """

    def __init__(
        self, X_valid_prepared, valid_df, unique_mi_configs, period=50, log_dir="logs"
    ):
        """
        Args:
            X_valid_prepared: Pre-built feature matrix for validation set
            valid_df: Original validation DataFrame (for ground truth and grouping)
            unique_mi_configs: DataFrame with all 128 MI configurations
            period: Evaluate every N iterations
            log_dir: Directory to save validation curves
        """
        self.X_valid = X_valid_prepared
        self.valid_df = valid_df
        self.unique_mi_configs = unique_mi_configs
        self.period = period
        self.log_dir = log_dir
        self.history = []
        self.current_pos = 0  # Pointer for slicing X_valid

    def before_iteration(self, model, epoch, evals_log):
        """Reset position pointer at the start of each iteration"""
        self.current_pos = 0
        return False

    def after_iteration(self, model, epoch, evals_log):
        """Compute and log ranking metrics after each training iteration"""
        if epoch % self.period != 0 and epoch != 0:
            return False

        # Metrics accumulators
        top1_correct = 0
        regrets = []
        true_best_ranks = []
        topk_contains = {k: 0 for k in TOP_K_VALUES}
        total_problems = 0
        real_top_errors = {k: [] for k in [1, 2, 3, 4, 5]}

        n_configs = len(self.unique_mi_configs)

        # Evaluate each problem size separately
        for (m, n, k), group in self.valid_df.groupby(["m", "n", "k"]):
            total_problems += 1

            # Get predictions for all 128 configs (use pre-built features!)
            X_block = self.X_valid[self.current_pos : self.current_pos + n_configs]
            pred_scores = model.predict(xgb.DMatrix(X_block))
            self.current_pos += n_configs

            # Ground truth
            true_best_gflops = group["gflops"].max()
            true_best_mi = int(group.loc[group["gflops"].idxmax(), "mi_idx"])

            # Predicted ranking
            pred_ranked = np.argsort(pred_scores)[::-1]
            pred_mi_list = [
                int(self.unique_mi_configs.iloc[i]["mi_idx"]) for i in pred_ranked
            ]

            # Top-1 accuracy
            if pred_mi_list[0] == true_best_mi:
                top1_correct += 1

            # Top-1 regret
            pred_best_mi = pred_mi_list[0]
            pred_best_gflops = (
                group[group["mi_idx"] == pred_best_mi]["gflops"].iloc[0]
                if pred_best_mi in group["mi_idx"].values
                else 0
            )
            regret = (true_best_gflops - pred_best_gflops) / (true_best_gflops + 1e-8)
            regrets.append(regret)

            # Rank of true best configuration
            true_rank = (
                pred_mi_list.index(true_best_mi) + 1
                if true_best_mi in pred_mi_list
                else 129
            )
            true_best_ranks.append(true_rank)

            # Top-k recall
            for k in TOP_K_VALUES:
                if true_best_mi in pred_mi_list[:k]:
                    topk_contains[k] += 1

            # Real Top-k prediction error
            true_sorted = group.sort_values("gflops", ascending=False)
            true_mi_list = true_sorted["mi_idx"].astype(int).values.tolist()
            true_gflops_list = true_sorted["gflops"].values.tolist()

            mi_to_pred = {
                int(self.unique_mi_configs.iloc[i]["mi_idx"]): pred_scores[i]
                for i in range(len(self.unique_mi_configs))
            }

            for k in [1, 2, 3, 4, 5]:
                if len(true_mi_list) >= k:
                    for i in range(k):
                        mi = true_mi_list[i]
                        true_val = true_gflops_list[i]
                        pred_val = mi_to_pred.get(mi, 0)
                        mre = abs(pred_val - true_val) / true_val if true_val > 0 else 0
                        real_top_errors[k].append(mre)

        # Print Real Top-k errors
        print("  RealTopErr: ", end="")
        for k in [1, 2, 3, 4, 5]:
            if real_top_errors[k]:
                avg_mre = np.mean(real_top_errors[k])
                print(f" T{k}:{avg_mre:5.1%}", end="")
            else:
                print(f" T{k}:  N/A ", end="")
        print("  |", end="")

        # Aggregate metrics
        top1_acc = top1_correct / total_problems
        mean_regret = np.mean(regrets)
        mean_rank = np.mean(true_best_ranks)

        # Print summary
        print(
            f"\n[Valid @ {epoch:4d}] "
            f"Top-1 Acc: {top1_acc:6.1%} | "
            f"Regret: {mean_regret:6.2%} | "
            f"TrueBest Rank: {mean_rank:5.1f} | "
            f"R@5: {topk_contains[5]/total_problems:6.1%} | "
            f"R@10: {topk_contains[10]/total_problems:6.1%} | "
            f"R@20: {topk_contains[20]/total_problems:6.1%}"
        )

        # Save to history
        self.history.append(
            {
                "epoch": epoch,
                "top1_acc": top1_acc,
                "regret": mean_regret,
                "mean_rank": mean_rank,
            }
        )

        return False

    def after_training(self, model):
        """Save validation curve to JSON file after training completes."""
        os.makedirs(self.log_dir, exist_ok=True)
        curve_file = os.path.join(
            self.log_dir, f"valid_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(curve_file, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nValidation curve saved to: {curve_file}")
        return model