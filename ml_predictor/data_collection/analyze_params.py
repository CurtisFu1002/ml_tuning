import yaml
import pandas as pd
from collections import defaultdict, Counter

def normalize_value(val):
    if isinstance(val, (list, dict)):
        return str(val)
    return val

def summarize_outliers(yaml_path, ignore_keys=None):
    # ignore the keys that are not relevant for comparison
    default_ignore = {
        "BaseName", "KernelNameMin", "SolutionNameMin",
        "SolutionIndex", "SolutionName", "CustomKernelName"
    }

    if ignore_keys is None:
        ignore_keys = default_ignore
    else:
        ignore_keys = set(ignore_keys) | default_ignore

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Count total kernels with "Kernel": True
    all_kernels = [k for k in data if isinstance(k, dict) and k.get("Kernel") is True]
    total_kernels = len(all_kernels)

    param_values = defaultdict(list)
    for idx, kernel in enumerate(all_kernels):
        for key, value in kernel.items():
            if key in ignore_keys:
                continue
            param_values[key].append((idx, normalize_value(value)))

    summary_records = []
    fixed_records = []

    for key, values in param_values.items():
        all_values = [v for _, v in values]
        counter = Counter(all_values)

        if len(counter) == 1:
            # fixed parameter
            val = all_values[0]
            fixed_records.append({
                "Parameter": key,
                "FixedValue": val,
                "Count": f"{len(all_values)}/{total_kernels}"
            })
            continue
        
        # the most common value 
        majority_val, majority_count = counter.most_common(1)[0]

        # outliers
        outliers = []
        for val, count in counter.items():
            if val != majority_val:
                indices = [idx for idx, v in values if v == val]
                outliers.append(f"value={val} @ kernels={';'.join(map(str, indices))}")

        summary_records.append({
            "Parameter": key,
            "MajorityValue": majority_val,
            "MajorityCount": f"{majority_count}/{total_kernels}",
            "Outliers": " | ".join(outliers)
        })

    summary_df = pd.DataFrame(summary_records)
    fixed_df = pd.DataFrame(fixed_records)

    return summary_df, fixed_df


if __name__ == "__main__":
    yaml_path = "00_Final_v1.yaml"
    summary_df, fixed_df = summarize_outliers(yaml_path)

    summary_df.to_csv("param_outlier_summary.csv", index=False)
    fixed_df.to_csv("param_fixed_summary.csv", index=False)

    print("Completed Analyze:")
    print(" - param_outlier_summary.csv")
    print(" - param_fixed_summary.csv")






