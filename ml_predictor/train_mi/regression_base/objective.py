"""
Custom objective functions for XGBoost.
Implements ranking-aware weighted loss to emphasize top-k configurations.
"""

import numpy as np


def compute_topk_weights(df, top_k=5, weight_decay='linear', base_weight=1.0, top1_weight=10.0):
    """
    Compute sample weights to emphasize top-k configurations.
    
    Weighting Strategy:
    - Top-1: Maximum weight (top1_weight)
    - Top-2 to Top-k: Decaying weights
    - Top-(k+1) to 128: Base weight (base_weight)
    
    Args:
        df: DataFrame, must contain 'm', 'n', 'k', 'gflops', 'mi_idx' columns
        top_k: Number of top configurations to emphasize
        weight_decay: Weight decay scheme ('linear', 'exponential', 'harmonic')
        base_weight: Base weight for non-top-k configurations
        top1_weight: Maximum weight for Top-1 configuration
        
    Returns:
        weights: numpy array of weights with same length as df
    """
    weights = np.ones(len(df)) * base_weight
    
    # Calculate ranking and weights for each problem size
    for (m, n, k), group in df.groupby(['m', 'n', 'k']):
        idx = group.index.tolist()
        gflops = group['gflops'].values
        
        # Calculate ranking (1-based, 1 = highest GFLOPS)
        ranks = (-gflops).argsort().argsort() + 1
        
        # Assign weights based on ranking
        group_weights = np.ones(len(group)) * base_weight
        
        for i, rank in enumerate(ranks):
            if rank == 1:
                # Top-1 gets maximum weight
                group_weights[i] = top1_weight
            elif rank <= top_k:
                # Top-2 to Top-k: decaying weights
                if weight_decay == 'linear':
                    # Linear decay: top1_weight -> base_weight
                    decay = (top1_weight - base_weight) * (top_k - rank) / (top_k - 1)
                    group_weights[i] = base_weight + decay
                elif weight_decay == 'exponential':
                    # Exponential decay
                    decay_rate = np.log(top1_weight / base_weight) / (top_k - 1)
                    group_weights[i] = top1_weight * np.exp(-decay_rate * (rank - 1))
                elif weight_decay == 'harmonic':
                    # Harmonic decay: 1/rank style
                    group_weights[i] = top1_weight / rank
                else:
                    # Default: linear
                    decay = (top1_weight - base_weight) * (top_k - rank) / (top_k - 1)
                    group_weights[i] = base_weight + decay
            # else: keep base_weight
        
        # Write weights back
        for local_i, global_i in enumerate(idx):
            weights[global_i] = group_weights[local_i]
    
    return weights


def compute_topk_weights_v2(df, top_k=5, scheme='stepped', 
                            top1_multiplier=10.0, topk_multiplier=3.0, base_weight=1.0):
    """
    Alternative weight calculation method: stepped weights.
    
    Args:
        df: DataFrame
        top_k: Number of top configurations to focus on
        scheme: 'stepped' | 'smooth'
        top1_multiplier: Weight multiplier for top-1
        topk_multiplier: Weight multiplier for top-2 to top-k
        base_weight: Base weight for remaining configurations
        
    Returns:
        weights: numpy array
    """
    weights = np.ones(len(df)) * base_weight
    
    for (m, n, k), group in df.groupby(['m', 'n', 'k']):
        idx = group.index.tolist()
        gflops = group['gflops'].values
        ranks = (-gflops).argsort().argsort() + 1
        
        group_weights = np.ones(len(group)) * base_weight
        
        if scheme == 'stepped':
            # Stepped: Top-1 highest, Top-2~k medium, others lowest
            for i, rank in enumerate(ranks):
                if rank == 1:
                    group_weights[i] = top1_multiplier
                elif rank <= top_k:
                    group_weights[i] = topk_multiplier
        
        elif scheme == 'smooth':
            # Smooth decay: using softmax-like weights
            max_gflops = gflops.max()
            min_gflops = gflops.min()
            if max_gflops > min_gflops:
                normalized = (gflops - min_gflops) / (max_gflops - min_gflops)
                # Temperature parameter controls distribution steepness
                temperature = 0.3
                exp_weights = np.exp(normalized / temperature)
                # Scale to [base_weight, top1_multiplier] range
                scaled = base_weight + (top1_multiplier - base_weight) * (exp_weights - exp_weights.min()) / (exp_weights.max() - exp_weights.min() + 1e-8)
                group_weights = scaled
        
        for local_i, global_i in enumerate(idx):
            weights[global_i] = group_weights[local_i]
    
    return weights


def weighted_mse_objective(predt, dtrain):
    """
    加權 MSE objective function。
    
    XGBoost 自定義 objective 需要返回 gradient 和 hessian。
    
    Loss: L = sum(w_i * (y_i - pred_i)^2)
    Gradient: dL/d(pred_i) = -2 * w_i * (y_i - pred_i)
    Hessian: d²L/d(pred_i)² = 2 * w_i
    
    Args:
        predt: 模型預測值
        dtrain: DMatrix，包含 label 和 weight
        
    Returns:
        grad: gradient array
        hess: hessian array
    """
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    
    if weights is None or len(weights) == 0:
        weights = np.ones_like(y)
    
    residual = y - predt
    grad = -2.0 * weights * residual
    hess = 2.0 * weights
    
    return grad, hess


def weighted_huber_objective(predt, dtrain, delta=1.0):
    """
    Weighted Huber loss objective (more robust to outliers).
    
    Huber Loss:
        L = 0.5 * (y - pred)^2           if |y - pred| <= delta
        L = delta * |y - pred| - 0.5 * delta^2   otherwise
    """
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    
    if weights is None or len(weights) == 0:
        weights = np.ones_like(y)
    
    residual = y - predt
    abs_residual = np.abs(residual)
    
    # Gradient
    grad = np.where(
        abs_residual <= delta,
        -weights * residual,  # MSE gradient
        -weights * delta * np.sign(residual)  # MAE gradient
    )
    
    # Hessian
    hess = np.where(
        abs_residual <= delta,
        weights,  # MSE hessian
        weights * 0.01  # Small hessian to avoid numerical issues
    )
    
    return grad, hess


def create_weighted_objective(delta=None):
    """
    Factory function: create weighted objective.
    
    Args:
        delta: If provided, use Huber loss; otherwise use MSE
        
    Returns:
        objective function
    """
    if delta is not None:
        def obj(predt, dtrain):
            return weighted_huber_objective(predt, dtrain, delta=delta)
        return obj
    else:
        return weighted_mse_objective


def weighted_rmse_metric(predt, dtrain):
    """
    Weighted RMSE evaluation metric.
    
    Args:
        predt: Predictions
        dtrain: DMatrix
        
    Returns:
        metric_name: str
        metric_value: float
    """
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    
    if weights is None or len(weights) == 0:
        weights = np.ones_like(y)
    
    weighted_mse = np.sum(weights * (y - predt) ** 2) / np.sum(weights)
    weighted_rmse = np.sqrt(weighted_mse)
    
    return 'weighted_rmse', weighted_rmse


def print_weight_statistics(weights, df, top_k=5):
    """
    Print weight statistics for debugging.
    """
    print("\n" + "="*60)
    print("SAMPLE WEIGHT STATISTICS")
    print("="*60)
    print(f"Total samples: {len(weights):,}")
    print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"Weight mean: {weights.mean():.2f}")
    print(f"Weight sum: {weights.sum():.2f}")
    
    # Analyze top-k vs others
    topk_mask = np.zeros(len(weights), dtype=bool)
    for (m, n, k), group in df.groupby(['m', 'n', 'k']):
        idx = group.index.tolist()
        gflops = group['gflops'].values
        ranks = (-gflops).argsort().argsort() + 1
        for local_i, global_i in enumerate(idx):
            if ranks[local_i] <= top_k:
                topk_mask[global_i] = True
    
    print(f"\nTop-{top_k} samples: {topk_mask.sum():,} ({topk_mask.sum()/len(weights)*100:.1f}%)")
    print(f"  Weight mean: {weights[topk_mask].mean():.2f}")
    print(f"Other samples: {(~topk_mask).sum():,} ({(~topk_mask).sum()/len(weights)*100:.1f}%)")
    print(f"  Weight mean: {weights[~topk_mask].mean():.2f}")
    print(f"\nEffective weight ratio (top-k / others): {weights[topk_mask].mean() / weights[~topk_mask].mean():.2f}x")
    print("="*60)