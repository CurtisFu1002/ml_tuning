import logging
from pathlib import Path
from typing import Any
from typing_extensions import Annotated

import typer
import yaml
from Tensile import Tensile

from config_generator.llm import get_llm_response, get_llm_response_parsed
from config_generator.prompts.gpu_spec import GPU_SPEC_INFO
from config_generator.prompts.template import get_user_prompt_v1, get_user_prompt_v2
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


@app.command()
def generate(
    config_yaml: Annotated[
        str,
        typer.Argument(
            help="the kernel parameter config file to use (YAML format)",
        ),
    ],
    output_file: Annotated[
        str,
        typer.Argument(
            help="the file path to write the model response to (Markdown is recommended)",
        ),
    ] = "stdout",
    logic_yaml: Annotated[
        str | None,
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
        str,
        typer.Argument(help="Benchmark config.yaml file"),
    ],
    output_path: Annotated[
        str,
        typer.Argument(help="Path to conduct benchmark and write output files"),
    ],
    prebuilt_client: Annotated[
        str,
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
        str,
        typer.Argument(
            help="Path to the kernel parameter config YAML to use",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Argument(
            help="Path to conduct benchmark and write LLM-generated config YAML and Tensile-generated output files",
        ),
    ],
    model_name: Annotated[str, typer.Option("--model")] = "gpt-oss:120b",
    gpu_name: Annotated[str, typer.Option("--gpu")] = "MI210",
    prebuilt_client: Annotated[
        str,
        typer.Option(
            help="Specify the full path to a pre-built tensilelite-client executable",
        ),
    ] = "/mnt/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client",
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
    config_text = Path(config_yaml).read_text()
    gpu_spec = GPU_SPEC_INFO[gpu_name]

    # Generate output
    prompt, response, yaml_content = _generate_helper(
        config_text, gpu_spec, model_name, logic_text=None
    )

    # Write output files
    generated_file = Path(output_dir).absolute() / "modified.yaml"
    write_output_files(generated_file, prompt, response, yaml_content)

    # Run Tensile with generated config
    Tensile.Tensile(
        [
            f"--prebuilt-client={prebuilt_client}",
            str(generated_file),
            str(generated_file.parent),
        ]
    )


if __name__ == "__main__":
    app()
