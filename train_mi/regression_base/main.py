"""
Main entry point for GFLOPS regression model training and evaluation.
Orchestrates the complete training pipeline from data loading to evaluation.
"""

import sys

from regression_base.args import get_args, print_config, get_xgb_params
from regression_base.data_prepare import load_data, prepare_full_dataset
from regression_base.train import train_model, save_model
from regression_base.eval import evaluate_model
from regression_base.utils import setup_logging, save_summary, ensure_directories


def main():
    """Main training and evaluation pipeline."""

    # Parse arguments
    args = get_args()

    # Setup directories
    ensure_directories(args.log_dir, args.plot_dir)

    # Setup logging
    logger, log_file = setup_logging(args.log_dir, args.exp_name)
    sys.stdout = logger

    print("=" * 80)
    print("GFLOPS REGRESSION MODEL - TRAINING PIPELINE")
    print("=" * 80)

    # Print configuration
    print_config(args)

    # ====================== Step 1: Load Data ======================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    df, unique_mi_configs = load_data(args.csv_path)

    # ====================== Step 2: Prepare Data ======================
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING DATA")
    print("=" * 80)

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train_df,
        valid_df,
        test_df,
    ) = prepare_full_dataset(df, unique_mi_configs, args)

    # ====================== Step 3: Train Model ======================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODEL")
    print("=" * 80)

    bst = train_model(
        X_train, y_train, X_valid, y_valid, valid_df, unique_mi_configs, args,
        train_df=train_df
    )

    # ====================== Step 4: Evaluate Model ======================
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING MODEL")
    print("=" * 80)

    results = evaluate_model(
        bst,
        X_test,
        test_df,
        unique_mi_configs,
        plot=args.plot_data,
        plot_dir=args.plot_dir,
    )

    # ====================== Step 5: Save Results ======================
    print("\n" + "=" * 80)
    print("STEP 5: SAVING RESULTS")
    print("=" * 80)

    # Save model
    save_model(bst, args.model_path)

    # Save summary
    xgb_params = get_xgb_params(args)
    summary_file = save_summary(results, xgb_params, args, args.log_dir, args.exp_name)

    # ====================== Final Summary ======================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Model saved to:   {args.model_path}")
    print(f"Summary saved to: {summary_file}")
    print(f"Log saved to:     {log_file}")
    if args.plot_data:
        print(f"Plots saved to:   {args.plot_dir}/")
    print("=" * 80)

    # Close logger
    logger.close()
    sys.stdout = sys.__stdout__

    print(f"\n✓ Training completed! Check {log_file} for details.")


if __name__ == "__main__":
    main()
