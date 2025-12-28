"""
Utility functions and classes.
Includes logging, file I/O, and other helper functions.
"""

import sys
import os
import json
from datetime import datetime


class Logger:
    """
    Dual-output logger that writes to both console and file.
    Useful for capturing all output during training.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Path to log file
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush both output streams."""
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file."""
        self.log.close()


def setup_logging(log_dir, exp_name=""):
    """
    Setup logging infrastructure.

    Args:
        log_dir: Directory for log files
        exp_name: Optional experiment name for log file

    Returns:
        logger: Logger object
        log_file: Path to log file
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_name:
        log_filename = f"training_{exp_name}_{timestamp}.log"
    else:
        log_filename = f"training_{timestamp}.log"

    log_file = os.path.join(log_dir, log_filename)
    logger = Logger(log_file)

    return logger, log_file


def save_summary(results, xgb_params, args, log_dir, exp_name=""):
    """
    Save experiment summary to JSON file.

    Args:
        results: Evaluation metrics dictionary
        xgb_params: XGBoost parameters dictionary
        args: Parsed arguments
        log_dir: Directory for log files
        exp_name: Optional experiment name

    Returns:
        summary_file: Path to summary JSON file
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_name:
        summary_filename = f"summary_{exp_name}_{timestamp}.json"
    else:
        summary_filename = f"summary_{timestamp}.json"

    summary_file = os.path.join(log_dir, summary_filename)

    # Prepare summary data
    summary = {
        "timestamp": timestamp,
        "experiment_name": exp_name,
        "config": {
            "csv_path": args.csv_path,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "valid_size": args.valid_size,
            "feature_engineering": {
                "use_standardization": args.use_standardization,
                "std_problem_size": args.std_problem_size,
                "std_wave_params": args.std_wave_params,
                "use_tile_type_encoding": args.use_tile_type_encoding,
                "remove_const_features": args.remove_const_features,
                "feature_extension": args.feature_extension,
            },
            "training": {
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
            },
        },
        "xgb_params": xgb_params,
        "results": results,
    }

    # Save to JSON
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    return summary_file


def ensure_directories(*dirs):
    """
    Create directories if they don't exist.

    Args:
        *dirs: Variable number of directory paths
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
