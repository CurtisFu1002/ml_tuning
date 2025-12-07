"""
Data loading, preprocessing, and feature engineering module.
Handles CSV loading, train/test splitting, and feature construction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Column definitions
PROBLEM_SIZE_COLS_mnk = ["m", "n", "k"]
MI_TUNING_PARAMETERS_COL = [
    "M",
    "N",
    "K",
    "B",
    "MIBlockM",
    "WaveTileM",
    "WaveTileN",
    "WaveM",
    "WaveN",
]
CONFIG_ID_COL = "mi_idx"

# Global scalers (initialized during first data preparation)
scaler_problem = None
scaler_wave = None


def load_data(csv_path):
    """
    Load data from CSV file.

    Args:
        csv_path: Path to the CSV file containing GFLOPS data

    Returns:
        df: Full DataFrame with all samples
        unique_mi_configs: DataFrame with unique MI configurations (128 configs)
    """
    df = pd.read_csv(csv_path)

    # Extract unique MI configurations
    unique_mi_configs = (
        df[["mi_idx"] + MI_TUNING_PARAMETERS_COL]
        .drop_duplicates()
        .sort_values("mi_idx")
        .reset_index(drop=True)
    )

    print(f"Loaded {len(df):,} samples, {len(unique_mi_configs)} MI configs")
    return df, unique_mi_configs


def split_data(df, test_size=0.1, valid_size=0.1, random_state=42):
    """
    Split data by problem size (m,n,k) to avoid data leakage.
    Each problem size contains 128 MI configurations.

    Args:
        df: Full DataFrame
        test_size: Ratio of test set
        valid_size: Ratio of validation set (from remaining training data)
        random_state: Random seed for reproducibility

    Returns:
        train_prob: Training problem sizes
        valid_prob: Validation problem sizes
        test_prob: Test problem sizes
    """
    # Get unique problem sizes
    problems = df[PROBLEM_SIZE_COLS_mnk].drop_duplicates().reset_index(drop=True)

    # First split: train+valid vs test
    train_valid_prob, test_prob = train_test_split(
        problems, test_size=test_size, random_state=random_state
    )

    # Second split: train vs valid
    # train_prob, valid_prob = train_test_split(
    #     train_valid_prob, test_size=valid_size, random_state=random_state + 1
    # )
    # another way to split
    valid_prob = train_valid_prob.sample(
        frac=valid_size,
        random_state=random_state + 1
    )
    train_prob = train_valid_prob.drop(valid_prob.index).reset_index(drop=True)
    valid_prob = valid_prob.reset_index(drop=True)

    print(f"Problem size split:")
    print(f"  Train: {len(train_prob)} problems")
    print(f"  Valid: {len(valid_prob)} problems")
    print(f"  Test:  {len(test_prob)} problems")

    return train_prob, valid_prob, test_prob


def feature_extension(df):
    """
    Add extended features to the DataFrame.
    Currently adds normalized GFLOPS within each problem size.

    Args:
        df: Input DataFrame

    Returns:
        df: DataFrame with additional features
    """
    # Calculate min/max GFLOPS for each problem size (m,n,k)
    stats = (
        df.groupby(["m", "n", "k"])["gflops"]
        .agg(min_g="min", max_g="max")
        .reset_index()
    )

    # Merge statistics back to original DataFrame
    df = df.merge(stats, on=["m", "n", "k"], how="left")

    # Create normalized GFLOPS feature
    df["gflops_norm"] = (df["gflops"] - df["min_g"]) / (
        df["max_g"] - df["min_g"] + 1e-8
    )
    df = df.drop(columns=["min_g", "max_g"])

    return df


def prepare_data(df, unique_mi_configs, args, extra_feature_cols=None):
    """
    Build feature matrix from raw data.
    Applies standardization and feature engineering based on configuration.

    This function is called ONCE on the full dataset to ensure consistency.
    Scalers are fitted on the first call and reused for all subsequent data.

    Args:
        df: DataFrame with samples
        unique_mi_configs: DataFrame with all MI configurations
        args: Parsed arguments containing feature engineering settings
        extra_feature_cols: Optional list of additional feature column names

    Returns:
        X: Feature matrix (numpy array)
        y: Target values (GFLOPS, numpy array)
    """
    global scaler_problem, scaler_wave

    # Create a mapping from mi_idx to configuration row
    config_map = {row["mi_idx"]: row for _, row in unique_mi_configs.iterrows()}

    X, y = [], []
    problem_sizes = [] if args.std_problem_size else None
    wave_params = [] if args.std_wave_params else None

    # Build features for each sample
    for _, row in df.iterrows():
        mi_idx = row["mi_idx"]
        if mi_idx not in config_map:
            continue
        cfg = config_map[mi_idx]

        feats = []

        if args.use_standardization:
            # 1. Problem size features (m, n, k)
            m, n, k = row["m"], row["n"], row["k"]
            if args.std_problem_size:
                problem_sizes.append([m, n, k])
                if scaler_problem is not None:  # Apply scaling if scaler exists
                    m, n, k = scaler_problem.transform([[m, n, k]])[0]
            feats.extend([m, n, k])

            # 2. Tile type encoding (square vs non-square)
            if args.use_tile_type_encoding:
                tile_type = 0 if (cfg["M"] == cfg["N"] == cfg["K"]) else 1
                feats.append(tile_type)
            else:
                feats.extend([cfg["M"], cfg["N"], cfg["K"]])

            # 3. Constant features (optionally removed)
            if not args.remove_const_features:
                feats.extend([cfg["B"], cfg["MIBlockM"]])

            # 4. Wave parameters (WaveTileM, WaveTileN, WaveM, WaveN)
            wave = [cfg["WaveTileM"], cfg["WaveTileN"], cfg["WaveM"], cfg["WaveN"]]
            if args.std_wave_params:
                wave_params.append(wave)
                if scaler_wave is not None:  # Apply scaling if scaler exists
                    wave = scaler_wave.transform([wave])[0]
            feats.extend(wave)
        else:
            # Use raw features without standardization
            feats = [
                row["m"],
                row["n"],
                row["k"],
                cfg["M"],
                cfg["N"],
                cfg["K"],
                cfg["B"],
                cfg["MIBlockM"],
                cfg["WaveTileM"],
                cfg["WaveTileN"],
                cfg["WaveM"],
                cfg["WaveN"],
            ]

        # Add extra features if specified
        if extra_feature_cols:
            feats.extend([row[col] for col in extra_feature_cols])

        X.append(feats)
        y.append(row["gflops"])

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Fit scalers on full data (only happens once)
    if args.use_standardization:
        if args.std_problem_size and scaler_problem is None and problem_sizes:
            scaler_problem = StandardScaler().fit(problem_sizes)
            print(f"[Scaler] Problem size fitted on {len(problem_sizes)} samples")
        if args.std_wave_params and scaler_wave is None and wave_params:
            scaler_wave = StandardScaler().fit(wave_params)
            print(f"[Scaler] Wave params fitted on {len(wave_params)} samples")

    return X, y


def prepare_full_dataset(df, unique_mi_configs, args):
    """
    Prepare the complete dataset with proper splitting.
    Builds features ONCE for all data, then splits to avoid data leakage.

    Args:
        df: Full DataFrame
        unique_mi_configs: DataFrame with all MI configurations
        args: Parsed arguments

    Returns:
        X_train, y_train: Training features and targets
        X_valid, y_valid: Validation features and targets
        X_test, y_test: Test features and targets
        train_df, valid_df, test_df: Original DataFrames for each split
    """
    # Optional feature extension
    extra_features = None
    if args.feature_extension:
        print("Applying feature extension...")
        df = feature_extension(df)
        extra_features = ["gflops_norm"]

    # Build FULL feature matrix ONCE (key to avoiding data leakage!)
    print("\nBuilding full feature matrix for all data points...")
    X_full, y_full = prepare_data(df, unique_mi_configs, args, extra_features)

    print(f"Final feature dimension: {X_full.shape[1]}")
    print(f"Total samples: {len(X_full):,}")

    # Split by problem size (no data leakage)
    train_prob, valid_prob, test_prob = split_data(
        df,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.random_state,
    )

    # Get indices for each split
    train_idx = df.merge(train_prob, on=PROBLEM_SIZE_COLS_mnk).index
    valid_idx = df.merge(valid_prob, on=PROBLEM_SIZE_COLS_mnk).index
    test_idx = df.merge(test_prob, on=PROBLEM_SIZE_COLS_mnk).index

    # Slice the pre-computed features
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_valid, y_valid = X_full[valid_idx], y_full[valid_idx]
    X_test, y_test = X_full[test_idx], y_full[test_idx]

    # Also keep original DataFrames for evaluation (needed for groupby operations)
    train_df = df.loc[train_idx].reset_index(drop=True)
    valid_df = df.loc[valid_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    print(f"\nSplit completed:")
    print(f"  Train: {len(train_prob)} problems → {len(X_train):,} samples")
    print(f"  Valid: {len(valid_prob)} problems → {len(X_valid):,} samples")
    print(f"  Test:  {len(test_prob)} problems → {len(X_test):,} samples")

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        train_df,
        valid_df,
        test_df,
    )
