import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any
from typing_extensions import Annotated

import numpy as np
import pandas as pd
import typer
import yaml
from Tensile import Tensile

from config_generator.llm import get_llm_response, get_llm_response_parsed, call_llm
from config_generator.prompts.gpu_spec import GPU_SPEC_INFO
from config_generator.prompts.template import (
    get_user_prompt_v1,
    get_user_prompt_v2,
    create_prompts,
)
from config_generator.utils import extract_yaml, write_output_files

app = typer.Typer()


def _generate_helper(
    config_text: str,
    gpu_spec: str,
    model_name: str,
    logic_text: str | None = None,
) -> tuple[str, str, str]:
    """
    Generate config/logic YAML using LLM.

    Returns:
        tuple of (prompt, response, yaml_content)
    """
    if logic_text is None:
        # v1: generate config file
        prompt = get_user_prompt_v1(config_text, gpu_spec)
        response: dict[str, Any] = get_llm_response_parsed(model_name, prompt)
        output_yaml = yaml.safe_dump(response, sort_keys=False)
        return prompt, str(response), output_yaml
    else:
        # v2: generate logic file
        prompt = get_user_prompt_v2(config_text, gpu_spec, logic_text)
        response: str = get_llm_response(model_name, prompt)
        output_yaml = extract_yaml(response)
        return prompt, response, output_yaml


def _compare_csv_files(file1: Path, file2: Path, column_name: str) -> bool:
    """Compare two CSV files based on a specific column."""
    with file1.open("r") as f1, file2.open("r") as f2:
        reader1 = csv.DictReader(f1)
        reader2 = csv.DictReader(f2)

        values1 = [row[column_name].strip() for row in reader1]
        values2 = [row[column_name].strip() for row in reader2]

        valid = values1 == values2

        if not valid:
            print(values1)
            print(values2)

        return valid


@app.command()
def generate(
    config_yaml: Annotated[
        Path,
        typer.Argument(
            help="the kernel parameter config file to use (YAML format)",
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Argument(
            help="the file path to write the model response to (Markdown or YAML)",
        ),
    ] = None,
    logic_yaml: Annotated[
        Path | None,
        typer.Option(
            help="the logic file example to guide LLM output format",
        ),
    ] = None,
    model_name: Annotated[str, typer.Option("--model")] = "gpt-oss:120b",
    gpu_name: Annotated[str, typer.Option("--gpu")] = "MI210",
    verbose: bool = False,
) -> None:
    """
    Generate optimized kernel configuration using LLM.

    Uses a large language model to analyze and optimize GEMM kernel parameters
    for specific GPU architectures. The LLM generates recommendations based on
    hardware specifications and best practices.

    Two modes of operation:

    - Config mode (no --logic-yaml): Generates optimized kernel config from input

    - Logic mode (with --logic-yaml): Generates logic file using example format


    Outputs (when output_file is not stdout):

        - <output_file>.yaml: Generated optimized configuration

        - <output_file>.md: Analysis including prompt, reasoning, and recommendations
    """

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logging.info(f"Config file: {config_yaml}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Logic file: {logic_yaml}")

    # Read input files
    config_text = Path(config_yaml).read_text()
    logic_text = Path(logic_yaml).read_text() if logic_yaml else None
    gpu_spec = GPU_SPEC_INFO[gpu_name]

    # Generate output
    prompt, response, yaml_content = _generate_helper(
        config_text, gpu_spec, model_name, logic_text
    )

    # Handle stdout
    if output_file in ["stdout", "-", "", None]:
        print(f"Response: {response}\n")
        print(f"Type: {type(response)}\n")
        return

    # Write output files
    output_path = Path(output_file)
    write_output_files(
        output_path, prompt, response, yaml_content, is_logic=logic_text is not None
    )


def _tensile_full_help(value: bool) -> None:
    """Show full Tensile help if requested."""
    if value:
        Tensile.Tensile(["--help"])
        raise typer.Exit()


@app.command()
def tensile(
    config_file: Annotated[
        Path,
        typer.Argument(help="Benchmark config.yaml file"),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(help="Path to conduct benchmark and write output files"),
    ],
    prebuilt_client: Annotated[
        Path,
        typer.Option(
            help="Specify the full path to a pre-built tensilelite-client executable",
        ),
    ] = "/mnt/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client",
    full_help: Annotated[
        bool,
        typer.Option(
            "--full-help",
            is_eager=True,
            callback=_tensile_full_help,
            help="Show full help message from underlying Tensile script",
        ),
    ] = False,
) -> None:
    """
    Run Tensile benchmark with the specified configuration.

    This command is a wrapper around the Tensile tuning framework that benchmarks
    GEMM kernel configurations on AMD GPUs. It measures actual performance metrics
    to validate optimization strategies.

    The benchmark tests various kernel parameters defined in the config file and
    generates performance data, assembly code, and optimized kernel implementations.


    Outputs (written to <output_path>):

        - Benchmark results and performance metrics

        - Generated kernel assembly code

        - Optimized kernel library files

        - Detailed timing and profiling data


    Note:

        Requires a properly configured AMD ROCm environment and compatible GPU.
        Benchmark execution may take significant time depending on config complexity.
    """
    Tensile.Tensile(
        [
            f"--prebuilt-client={prebuilt_client}",
            config_file,
            output_path,
        ]
    )


@app.command()
def autotune(
    config_yaml: Annotated[
        Path,
        typer.Argument(
            help="Path to the kernel parameter config YAML to use",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to conduct benchmark and write LLM-generated config YAML and Tensile-generated output files",
        ),
    ],
    model_name: Annotated[str, typer.Option("--model")] = "gpt-oss:120b",
    gpu_name: Annotated[str, typer.Option("--gpu")] = "MI210",
    prebuilt_client: Annotated[
        Path,
        typer.Option(
            help="Specify the full path to a pre-built tensilelite-client executable",
        ),
    ] = "/mnt/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client",
    validate: Annotated[
        Path | None,
        typer.Option(
            "--validate",
            help="Specify the path to the output of Tensile-only tuning, which is used to check the correctness of LLM-integrated tuning",
        ),
    ] = None,
) -> None:
    """
    Generate optimized config using LLM and run Tensile benchmark automatically.


    This command combines the 'generate' and 'tensile' commands into a single workflow:

    1. Uses LLM to generate an optimized kernel config based on GPU specifications

    2. Writes the generated config and analysis to the output directory

    3. Automatically runs Tensile benchmark using the generated config


    Outputs:

        - <output_dir>/modified.yaml: LLM-generated optimized config

        - <output_dir>/modified.md: Analysis and reasoning from LLM

        - Tensile benchmark results in the output directory
    """
    # Read input files
    config_file = Path(config_yaml).absolute()
    config_text = config_file.read_text()
    gpu_spec = GPU_SPEC_INFO[gpu_name]

    # Generate output
    t = time.time()
    prompt, response, yaml_content = _generate_helper(
        config_text, gpu_spec, model_name, logic_text=None
    )
    time_generation = time.time() - t

    # Write output files
    output_path = Path(output_dir).absolute()
    generated_file = output_path / "modified.yaml"
    write_output_files(generated_file, prompt, response, yaml_content)

    # Run Tensile with generated config
    t = time.time()
    Tensile.Tensile(
        [
            f"--prebuilt-client={prebuilt_client}",
            str(generated_file),
            str(generated_file.parent),
        ]
    )
    time_benchmark = time.time() - t

    # Optional validation step
    if validate:
        validate_path = Path(validate).absolute()
        t = time.time()
        Tensile.Tensile(
            [
                f"--prebuilt-client={prebuilt_client}",
                str(config_file),
                str(validate_path),
            ]
        )
        time_baseline = time.time() - t

        valid = _compare_csv_files(
            validate_path
            / "2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_Bias_UserArgs_00_CSVWinner.csv",
            output_path
            / "2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_Bias_UserArgs_00_CSVWinner.csv",
            column_name=" WinnerName",
        )

        print("[Validation Result]")
        if valid:
            print("Success: LLM-integrated tuning matches Tensile-only tuning.")
        else:
            print(
                "Failed: Results differ between LLM-integrated and Tensile-only tuning."
            )

    print("\n[Timing Summary (Optimized)]")
    print(f"LLM generation time: {time_generation:.2f} seconds")
    print(f"Tensile tuning time: {time_benchmark:.2f} seconds")
    print(f"Total time: {time_generation + time_benchmark:.2f} seconds")
    if validate:
        print("\n[Timing Summary (Baseline)]")
        print(f"Tensile tuning time: {time_baseline:.2f} seconds")


@app.command()
def evaluate(
    config_yaml: Annotated[
        Path,
        typer.Argument(
            help="Path to the kernel parameter config YAML to use",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to conduct benchmark and write LLM-generated config YAML and Tensile-generated output files",
        ),
    ],
    num_runs: Annotated[
        int,
        typer.Option(
            "--num-runs",
            help="Number of times to run the evaluation",
        ),
    ] = 1,
    version: Annotated[str, typer.Option(help="Version of prompt strategies")] = "v1_2",
    model_name: Annotated[str, typer.Option("--model")] = "gpt-oss:120b",
    gpu_name: Annotated[str, typer.Option("--gpu")] = "MI210",
    prebuilt_client: Annotated[
        Path,
        typer.Option(
            help="Specify the full path to a pre-built tensilelite-client executable",
        ),
    ] = "/mnt/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client",
) -> None:
    """
    Evaluate LLM-guided kernel tuning with quantitative metrics.

    This command runs a baseline Tensile benchmark followed by multiple LLM-guided
    tuning iterations. It computes the following evaluation metrics for each run:

    Metrics:

        - Tuning Time Reduction (TR): Net time savings compared to baseline,
          calculated as (T_baseline - (T_llm + T_tensile)) / T_baseline

        - Performance Retention (PR): Ratio of optimized to baseline GFLOPS,
          calculated as max_perf_optimized / max_perf_baseline

        - Winner Consistency (WC): Whether the LLM-optimized config produces
          the same winning kernel as the baseline


    Outputs:

        - <output_dir>/baseline/: Baseline Tensile benchmark results

        - <output_dir>/run_<n>/: Results for each LLM-guided run

        - <output_dir>/evaluation_summary.csv: Aggregated metrics across all runs
    """

    # Read input files
    config_file = Path(config_yaml).absolute()
    config_text = config_file.read_text()
    gpu_spec = GPU_SPEC_INFO[gpu_name]

    # Run baseline Tensile benchmark
    t = time.time()
    Tensile.Tensile(
        [
            f"--prebuilt-client={prebuilt_client}",
            str(config_file),
            str(output_dir / "baseline"),
        ]
    )
    time_baseline = time.time() - t
    shutil.copy2(config_file, output_dir / "baseline" / config_file.name)

    FILENAME = "2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_Bias_UserArgs_00_CSVWinner.csv"
    df_baseline = pd.read_csv(output_dir / "baseline" / FILENAME)
    perf_baseline: np.float64 = df_baseline[" WinnerGFlops"].values[0]
    winner_baseline: str = df_baseline[" WinnerName"].values[0]

    df = pd.DataFrame(
        columns=[
            "Run",
            "Baseline Time (s)",
            "LLM Time (s)",
            "Tensile Time (s)",
            "Tuning Time Reduction",
            "Performance Retention",
            "Winner Matched",
        ]
    )

    # Run multiple evaluation runs
    for run_idx in range(num_runs):
        print(f"\n[Run {run_idx + 1}/{num_runs}]")

        # Generate output
        t = time.time()
        prompts = create_prompts(version, config_text, gpu_spec)
        response = call_llm(model_name, prompts)
        yaml_content = yaml.safe_dump(response, sort_keys=False)
        time_llm = time.time() - t
        prompt = prompts[-1]["content"]

        # Write output files
        output_path = (output_dir / f"run_{run_idx + 1}").absolute()
        generated_file = output_path / "modified.yaml"
        write_output_files(generated_file, prompt, str(response), yaml_content)

        # Run Tensile with generated config
        t = time.time()
        Tensile.Tensile(
            [
                f"--prebuilt-client={prebuilt_client}",
                str(generated_file),
                str(generated_file.parent),
            ]
        )
        time_tensile = time.time() - t

        # Calculate tuning time reduction
        tuning_time_reduction: float = (
            time_baseline - (time_llm + time_tensile)
        ) / time_baseline

        # Read benchmark results from CSV
        df_optimized = pd.read_csv(output_path / FILENAME)
        perf_optimized: np.float64 = df_optimized[" WinnerGFlops"].values[0]
        winner_optimized: str = df_optimized[" WinnerName"].values[0]

        # Calculate performance retention
        performance_retension: float = (
            perf_optimized / perf_baseline if perf_baseline != 0 else 0
        )

        # Check if winners matched
        winner_matched: bool = winner_optimized == winner_baseline

        df.loc[run_idx] = [
            run_idx + 1,
            round(time_baseline, 2),
            round(time_llm, 2),
            round(time_tensile, 2),
            round(tuning_time_reduction, 4),
            round(performance_retension, 4),
            winner_matched,
        ]
        winner_consistency = df["Winner Matched"].mean()

        print("\n[Evaluation Summary]")
        print(df.to_string(index=False))  # type: ignore
        print(f"Winner Consistency = {winner_consistency}")

        df.to_csv(output_dir / "evaluation_summary.csv", index=False)
        df.to_markdown(output_dir / "evaluation_summary.md", index=False)
        md_file = Path(output_dir / "evaluation_summary.md")
        md_file.write_text(
            "# Evaluation Summary\n\n"
            + md_file.read_text()
            + f"\n\nWinner Consistency = {winner_consistency}\n"
        )


if __name__ == "__main__":
    app()
