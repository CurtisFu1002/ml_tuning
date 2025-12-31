"""
Model training module.
Handles XGBoost model training with custom callbacks.
"""

import xgboost as xgb
from .callbacks import ValidRankingMetrics
from .objective import compute_topk_weights, compute_topk_weights_v2, print_weight_statistics

def train_model(X_train, y_train, X_valid, y_valid, valid_df, 
                unique_mi_configs, args, train_df=None):
    """
    Train XGBoost model with validation callbacks.
    
    Args:
        X_train, y_train: Training features and targets
        X_valid, y_valid: Validation features and targets
        valid_df: Original validation DataFrame for callbacks
        unique_mi_configs: DataFrame with all MI configurations
        args: Parsed arguments containing training configuration
        
    Returns:
        bst: Trained XGBoost Booster model
    """
    from .args import get_xgb_params
    
    # Sample weights
    sample_weights = None
    if args.use_sample_weights and train_df is not None:
        from .objective import compute_topk_weights, compute_topk_weights_v2, print_weight_statistics
        
        if args.weight_scheme in ['stepped', 'smooth']:
            sample_weights = compute_topk_weights_v2(
                df=train_df,
                top_k=args.weight_top_k,
                scheme=args.weight_scheme,
                top1_multiplier=args.weight_top1,
                topk_multiplier=args.weight_topk,
                base_weight=args.weight_base
            )
        else:
            sample_weights = compute_topk_weights(
                df=train_df,
                top_k=args.weight_top_k,
                weight_decay=args.weight_scheme,
                base_weight=args.weight_base,
                top1_weight=args.weight_top1
            )
        
        # Print weight statistics
        print_weight_statistics(sample_weights, train_df, args.weight_top_k)

    # Create DMatrix with sample weights if provided
    dtrain = xgb.DMatrix(X_train, label=y_train)
    if sample_weights is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        print(f"Sample weights: ENABLED (top-{args.weight_top_k}, scheme={args.weight_scheme})")
    # Create DMatrix objects
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # Get XGBoost parameters from args
    xgb_params = get_xgb_params(args)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_valid):,}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Boost rounds: {args.num_boost_round}")
    print(f"Early stopping: {args.early_stopping_rounds}")
    print("="*80 + "\n")
    
    # Initialize validation callback
    ranking_callback = ValidRankingMetrics(
        X_valid_prepared=X_valid,
        valid_df=valid_df,
        unique_mi_configs=unique_mi_configs,
        period=args.callback_period,
        log_dir=args.log_dir
    )
    
    train_kwargs = {
        'params': xgb_params,
        'dtrain': dtrain,
        'num_boost_round': args.num_boost_round,
        'evals': [(dtrain, 'train'), (dvalid, 'valid')],
        'early_stopping_rounds': args.early_stopping_rounds,
        'callbacks': [ranking_callback],
        'verbose_eval': args.verbose_eval
    }
    
    # use custom objective
    if args.use_custom_objective:
        from .objective import create_weighted_objective, weighted_rmse_metric
        
        custom_obj = create_weighted_objective(delta=args.huber_delta)
        train_kwargs['obj'] = custom_obj
        # Optional: use custom metric
        # train_kwargs['custom_metric'] = weighted_rmse_metric
        
        print("Using CUSTOM weighted objective function")
        if args.huber_delta:
            print(f"  Loss type: Huber (delta={args.huber_delta})")
        else:
            print(f"  Loss type: Weighted MSE")
    
    # Train model
    bst = xgb.train(**train_kwargs)


    # Train model
    # bst = xgb.train(
    #     params=xgb_params,
    #     dtrain=dtrain,
    #     num_boost_round=args.num_boost_round,
    #     evals=[(dtrain, 'train'), (dvalid, 'valid')],
    #     early_stopping_rounds=args.early_stopping_rounds,
    #     callbacks=[ranking_callback],
    #     verbose_eval=args.verbose_eval
    # )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best iteration: {bst.best_iteration}")
    print(f"Best score: {bst.best_score}")
    print("="*80 + "\n")
    
    return bst


def save_model(bst, model_path):
    """
    Save trained model to file.
    
    Args:
        bst: Trained XGBoost model
        model_path: Path to save the model
    """
    bst.save_model(model_path)
    print(f"Model saved to: {model_path}")