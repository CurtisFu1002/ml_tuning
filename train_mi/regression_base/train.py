"""
Model training module.
Handles XGBoost model training with custom callbacks.
"""

import xgboost as xgb
from .callbacks import ValidRankingMetrics


def train_model(X_train, y_train, X_valid, y_valid, valid_df, 
                unique_mi_configs, args):
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
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
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
    
    # Train model
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=args.early_stopping_rounds,
        callbacks=[ranking_callback],
        verbose_eval=args.verbose_eval
    )
    
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