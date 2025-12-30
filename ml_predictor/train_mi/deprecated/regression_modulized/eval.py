"""
Model evaluation and visualization module.
Computes comprehensive metrics and generates evaluation plots.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os


# Top-k values for evaluation
TOP_K_VALUES = [1, 2, 3, 5, 10, 15, 20]


def evaluate_model(
    bst, X_test_prepared, test_df, unique_mi_configs, plot=True, plot_dir="plots"
):
    """
    Comprehensive evaluation of trained model on test set.

    This function:
    - Uses pre-prepared features (consistent with training)
    - Computes ranking metrics (Top-k accuracy, regret, etc.)
    - Calculates prediction errors (MAE, RMSE, MRE)
    - Generates visualization plots

    Args:
        bst: Trained XGBoost model
        X_test_prepared: Pre-built feature matrix for test set
        test_df: Original test DataFrame (for ground truth)
        unique_mi_configs: DataFrame with all 128 MI configurations
        plot: Whether to generate evaluation plots
        plot_dir: Directory to save plots

    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    os.makedirs(plot_dir, exist_ok=True)

    n_configs = len(unique_mi_configs)
    total_problems = len(test_df.groupby(["m", "n", "k"]))

    # Metrics containers
    regrets = []
    rank_of_true_best_list = []
    topk_recall = {k: 0 for k in TOP_K_VALUES}
    topk_regret = {k: [] for k in TOP_K_VALUES}
    real_topk_mean_rank = {k: [] for k in TOP_K_VALUES}
    error_mae, error_rmse, error_mre = [], [], []
    results_for_plot = []

    print(
        f"\nEvaluating on {total_problems} test problems "
        f"(each with up to {n_configs} configs)..."
    )

    # Process each problem size
    test_start_idx = 0

    for (m, n, k), group in test_df.groupby(["m", "n", "k"]):
        group = group.reset_index(drop=True)

        # Get predictions for all 128 configs using pre-built features
        X_block = X_test_prepared[test_start_idx : test_start_idx + n_configs]
        pred_scores = bst.predict(xgb.DMatrix(X_block))
        test_start_idx += n_configs

        # Ground truth
        true_best_gflops = group["gflops"].max()
        true_best_mi_idx = int(group.loc[group["gflops"].idxmax(), "mi_idx"])
        mi_to_gflops = dict(zip(group["mi_idx"].astype(int), group["gflops"]))

        # Predicted ranking
        pred_ranked_idx = np.argsort(pred_scores)[::-1]
        pred_mi_list = [
            int(unique_mi_configs.iloc[i]["mi_idx"]) for i in pred_ranked_idx
        ]

        # Top-1 prediction
        pred_best_mi = pred_mi_list[0]
        pred_best_gflops = mi_to_gflops.get(pred_best_mi, 0.0)
        regret = (true_best_gflops - pred_best_gflops) / (true_best_gflops + 1e-8)
        regrets.append(regret)

        # Rank of true best configuration
        true_best_rank = (
            pred_mi_list.index(true_best_mi_idx) + 1
            if true_best_mi_idx in pred_mi_list
            else 129
        )
        rank_of_true_best_list.append(true_best_rank)

        # Top-k recall (how often true best is in predicted top-k)
        for k in TOP_K_VALUES:
            if true_best_mi_idx in pred_mi_list[:k]:
                topk_recall[k] += 1

        # Top-k regret
        for k in TOP_K_VALUES:
            best_in_topk = max(mi_to_gflops.get(mi, 0.0) for mi in pred_mi_list[:k])
            tk_regret = max(
                0.0, (true_best_gflops - best_in_topk) / (true_best_gflops + 1e-8)
            )
            topk_regret[k].append(tk_regret)

        # Real Top-k Mean Rank
        # (where are the actual top-1,2,...,k configs ranked by model?)
        true_sorted = group.sort_values("gflops", ascending=False)
        true_topk_mi = true_sorted["mi_idx"].astype(int).values[: max(TOP_K_VALUES)]
        for k in TOP_K_VALUES:
            ranks = []
            for mi in true_topk_mi[:k]:
                rank = pred_mi_list.index(mi) + 1 if mi in pred_mi_list else 129
                ranks.append(rank)
            real_topk_mean_rank[k].append(np.mean(ranks))

        # Prediction error on measured configs
        for i, mi_idx in enumerate(unique_mi_configs["mi_idx"]):
            mi_idx = int(mi_idx)
            if mi_idx in mi_to_gflops:
                true_val = mi_to_gflops[mi_idx]
                pred_val = pred_scores[i]
                abs_err = abs(pred_val - true_val)
                rel_err = abs_err / (true_val + 1e-8)
                error_mae.append(abs_err)
                error_rmse.append(abs_err**2)
                error_mre.append(rel_err)

        # Save for plotting
        results_for_plot.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "true_gflops": true_best_gflops,
                "pred_gflops": pred_best_gflops,
                "regret": regret,
                "true_best_rank": true_best_rank,
            }
        )

        # Print first few examples
        if len(results_for_plot) <= 3:
            print(
                f"  {m:5}×{n:5}×{k:5} | True={true_best_gflops:6.0f} → "
                f"Pred={pred_best_gflops:6.0f} | Regret={regret:.2%} | "
                f"Rank={true_best_rank:3d}"
            )

    # Aggregate metrics
    mae = np.mean(error_mae)
    rmse = np.sqrt(np.mean(error_rmse))
    mre = np.mean(error_mre) * 100

    metrics = {
        "Top1_Accuracy": sum(1 for r in regrets if r <= 0.0) / total_problems,
        "Top1_Regret": np.mean(regrets),
        "Mean_Rank_of_TrueBest": np.mean(rank_of_true_best_list),
        "MAE": mae,
        "RMSE": rmse,
        "MRE(%)": mre,
        "TopK_Recall": {
            f"R@{k}": topk_recall[k] / total_problems for k in TOP_K_VALUES
        },
        "TopK_Regret": {f"Regret@{k}": np.mean(topk_regret[k]) for k in TOP_K_VALUES},
        "RealTopK_MeanRank": {
            f"MeanRank@{k}": np.mean(real_topk_mean_rank[k]) for k in TOP_K_VALUES
        },
    }

    # Print summary
    print_evaluation_summary(metrics)

    # Generate plots
    if plot:
        plot_evaluation(
            results_df=pd.DataFrame(results_for_plot),
            rank_of_true_best_list=rank_of_true_best_list,
            topk_regret=topk_regret,
            real_topk_mean_rank=real_topk_mean_rank,
            mre_list=error_mre,
            plot_dir=plot_dir,
        )

    return metrics


def print_evaluation_summary(metrics):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Top-1 Accuracy       : {metrics['Top1_Accuracy']:.2%}")
    print(f"Top-1 Regret         : {metrics['Top1_Regret']:.2%}")
    print(f"Mean Rank (True Best): {metrics['Mean_Rank_of_TrueBest']:.2f}")
    print(
        f"Prediction Error     : MAE={metrics['MAE']:.1f} | "
        f"RMSE={metrics['RMSE']:.1f} | MRE={metrics['MRE(%)']:.2f}%"
    )
    print("\nTop-k Performance:")
    for k in [1, 5, 10, 20]:
        print(
            f"  R@{k:2d} = {metrics['TopK_Recall'][f'R@{k}']:.2%} | "
            f"Regret@{k:2d} = {metrics['TopK_Regret'][f'Regret@{k}']:.2%} | "
            f"MeanRank@{k:2d} = {metrics['RealTopK_MeanRank'][f'MeanRank@{k}']:.2f}"
        )
    print("=" * 80)


def plot_evaluation(
    results_df,
    rank_of_true_best_list,
    topk_regret,
    real_topk_mean_rank,
    mre_list,
    plot_dir,
):
    """
    Generate comprehensive evaluation plots.

    Creates a 2x3 subplot figure with:
    1. Top-1 Regret Distribution
    2. Predicted vs True GFLOPS
    3. Rank of True Best Distribution
    4. Mean Top-k Regret
    5. Mean Rank of Real Top-k
    6. Relative Prediction Error Distribution

    Args:
        results_df: DataFrame with per-problem results
        rank_of_true_best_list: List of true best ranks
        topk_regret: Dict of top-k regret lists
        real_topk_mean_rank: Dict of real top-k mean ranks
        mre_list: List of mean relative errors
        plot_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Top-1 Regret Distribution
    axes[0, 0].hist(
        results_df["regret"], bins=50, alpha=0.7, color="salmon", edgecolor="black"
    )
    axes[0, 0].axvline(0.05, color="red", linestyle="--", label="5% threshold")
    axes[0, 0].set_title("Top-1 Regret Distribution", fontsize=14)
    axes[0, 0].set_xlabel("Regret")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Predicted vs True GFLOPS
    axes[0, 1].scatter(
        results_df["true_gflops"], results_df["pred_gflops"], alpha=0.6, s=20
    )
    maxv = max(results_df["true_gflops"].max(), results_df["pred_gflops"].max()) * 1.05
    axes[0, 1].plot([0, maxv], [0, maxv], "k--", label="Perfect prediction")
    axes[0, 1].set_title("Predicted vs True Best GFLOPS", fontsize=14)
    axes[0, 1].set_xlabel("True GFLOPS")
    axes[0, 1].set_ylabel("Predicted GFLOPS")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Rank of True Best
    axes[0, 2].hist(
        rank_of_true_best_list, bins=50, alpha=0.7, color="lightblue", edgecolor="black"
    )
    axes[0, 2].set_title("Rank of True Best Configuration", fontsize=14)
    axes[0, 2].set_xlabel("Rank")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].grid(alpha=0.3)

    # 4. Mean Top-k Regret
    axes[1, 0].bar(
        TOP_K_VALUES,
        [np.mean(topk_regret[k]) for k in TOP_K_VALUES],
        color="steelblue",
        edgecolor="black",
    )
    axes[1, 0].set_title("Mean Top-k Regret", fontsize=14)
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Mean Regret")
    axes[1, 0].set_xticks(TOP_K_VALUES)
    axes[1, 0].grid(alpha=0.3)

    # 5. Mean Rank of Real Top-k
    axes[1, 1].bar(
        TOP_K_VALUES,
        [np.mean(real_topk_mean_rank[k]) for k in TOP_K_VALUES],
        color="orange",
        edgecolor="black",
    )
    axes[1, 1].set_title("Mean Rank of Real Top-k Configs", fontsize=14)
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Mean Rank")
    axes[1, 1].set_xticks(TOP_K_VALUES)
    axes[1, 1].grid(alpha=0.3)

    # 6. Relative Prediction Error Distribution
    axes[1, 2].hist(mre_list, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[1, 2].set_title("Relative Prediction Error Distribution", fontsize=14)
    axes[1, 2].set_xlabel("MRE")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_dir, "eval_full_v2.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nEvaluation plots saved to: {plot_path}")
    plt.close()
