import argparse
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

MT0 = 256
MT1 = 256
CU_COUNT = 104
M_N_B_K_INDEX_IN_LOGIC_YAML = 7


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logic",
        type=Path,
        required=True,
        help="Folder containing logic yaml files",
    )
    return parser.parse_args()


def get_logic_yamls(logic_yaml_dir: Path):
    logic_yaml_dir = logic_yaml_dir.resolve()
    return [logic_yaml_path for logic_yaml_path in logic_yaml_dir.rglob("*.yaml")]


def get_m_n_b_k(logic_yaml_path: Path):
    problem_sizes: Set[Tuple[int, int, int, int]] = set()
    with open(logic_yaml_path, "r") as f:
        data = yaml.safe_load(f)
        assert data[M_N_B_K_INDEX_IN_LOGIC_YAML - 1] == [2, 3, 0, 1]
        for m_n_b_k, _ in data[M_N_B_K_INDEX_IN_LOGIC_YAML]:
            problem_sizes.add(tuple(m_n_b_k[:4]))
    return problem_sizes


def filter_out_problem_sizes(problem_sizes: Set[Tuple[int, int, int, int]]):
    filtered_problem_sizes = {
        problem_size
        for problem_size in problem_sizes
        if problem_size[0] * problem_size[1] <= MT0 * MT1 * CU_COUNT * 10
    }
    print(
        f"Number of problem sizes before/after filter: {len(problem_sizes)}/{len(filtered_problem_sizes)}"
    )
    return filtered_problem_sizes


def parse_problem_size(problem_sizes: Set[Tuple[int, int, int, int]]):
    m_coords, n_coords, b_coords, k_coords = map(list, zip(*problem_sizes))

    m_sse = summarize_dimension(m_coords)
    n_sse = summarize_dimension(n_coords)
    b_sse = summarize_dimension(b_coords)
    k_sse = summarize_dimension(k_coords)
    print(f"m_sse: {m_sse}")
    print(f"n_sse: {n_sse}")
    print(f"b_sse: {b_sse}")
    print(f"k_sse: {k_sse}")

    print(
        f"smallest: [{min(m_coords)}, {min(n_coords)}, {min(b_coords)}, {min(k_coords)}]"
    )
    print(
        f"largest: [{max(m_coords)}, {max(n_coords)}, {max(b_coords)}, {max(k_coords)}]"
    )
    # Plot the points on a 2D grid
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(m_coords, n_coords, c=k_coords, cmap="viridis", s=5)
    plt.title("2D Grid Map of Points")
    plt.xlabel("m", fontsize=14)
    plt.ylabel("n", fontsize=14)

    x_ticks = np.arange(0, max(m_coords) + 1, 500)
    y_ticks = np.arange(0, max(n_coords) + 1, 500)
    x_tick_labels = [str(x) if i % 2 == 0 else "" for i, x in enumerate(x_ticks)]
    y_tick_labels = [str(y) if i % 2 == 0 else "" for i, y in enumerate(y_ticks)]
    plt.xticks(x_ticks, x_tick_labels, fontsize=10, rotation=90)
    plt.yticks(y_ticks, y_tick_labels, fontsize=10, rotation=0)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(y_ticks, fontsize=10)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Add colorbar to show k values
    cbar = plt.colorbar(scatter)
    cbar.set_label("k value", fontsize=12)

    # plt.tight_layout()
    plt.savefig("problem_sizes.jpg", format="jpg")


def summarize_dimension(values):
    """
    Given a list of integers, return:
    - [start, step, end] if it's an arithmetic sequence
    - [value] if all values are the same
    - sorted list if irregular (optional)
    """
    unique_vals = sorted(set(values))
    if len(unique_vals) == 1:
        return [unique_vals[0]]
    elif len(unique_vals) == 2:
        # Not enough points to confirm a pattern
        return unique_vals

    # Try to detect step
    diffs = [b - a for a, b in zip(unique_vals, unique_vals[1:])]
    step = diffs[0]

    if all(d == step for d in diffs):
        return [unique_vals[0], step, unique_vals[-1]]
    else:
        return unique_vals  # Irregular


def main():
    args = arg_parse()
    logic_yaml_paths = get_logic_yamls(args.logic)

    problem_sizes: Set[Tuple[int, int, int, int]] = set()

    for logic_yaml_path in tqdm(logic_yaml_paths):
        problem_sizes |= filter_out_problem_sizes(get_m_n_b_k(logic_yaml_path))

    parse_problem_size(problem_sizes)


if __name__ == "__main__":
    main()
