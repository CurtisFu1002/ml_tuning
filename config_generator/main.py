import argparse
import logging
import re
from pathlib import Path
import subprocess
from typing import Any
from typing_extensions import Annotated

from ollama import chat
import typer
import yaml

from config_generator.prompts.gpu_spec import GPU_SPEC_INFO
from config_generator.prompts.format import ConfigYaml


app = typer.Typer()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune a GEMM kernel using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:120b",
        help="specify the model to use from https://ollama.com/search",
    )
    parser.add_argument(
        "--config-file",
        "-i",
        type=str,
        required=True,
        help="the kernel parameter config file to use",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default="stdout",
        help="the file path to write the model response to (Markdown is recommended)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print debug information",
    )
    return parser.parse_args()


def get_llm_response(model_name: str, prompt: str) -> str:
    logging.info(f"Using model: {model_name}")
    resp = chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return resp.message.content


def get_llm_response_parsed(model_name: str, prompt: str) -> dict[str, Any]:
    logging.info(f"Using model: {model_name}")
    resp = chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        format=ConfigYaml.model_json_schema(),
        options={"temperature": 0},
    )
    return ConfigYaml.model_validate_json(resp.message.content).model_dump()


def get_user_prompt_v1(config: str, gpu_spec: str) -> str:
    prompt = f"""You are a performance engineer at AMD working on optimizing the GEMM kernels for hipBLASLt in ROCm libraries.

You are trying to tune a GEMM kernel using Tensile tuning framework on AMD's MI210 (gfx90a) GPU.

The input config yaml is:

```python
{config}
```

and the GPU spec is:

```python
{gpu_spec}
```

Please tell me how to modify the config to the better performance with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format

Delete the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!
"""
    return prompt


def get_user_prompt_v2(config: str, gpu_spec: str, logic: str) -> str:
    return f"""You are a performance engineer at AMD working on optimizing the GEMM kernels for hipBLASLt in ROCm libraries.

You are trying to tune a GEMM kernel using Tensile tuning framework on AMD's MI210 (gfx90a) GPU.

The input config yaml is:

```yaml
{config}
```

and the GPU spec is:

```python
{gpu_spec}
```

The output format is called 'logic file', and here is an example:

```yaml
{logic}
```

Please provide an optimal logic file which will potentially deliver the best performance on the given GPU along with the following analysis:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Optimal logic file in YAML format surrounded by triple backticks (```yaml)

Note that all the fields in the output logic yaml (in the example) should be remained in output, even if it is not changed!
"""


def extract_yaml(text: str) -> str | None:
    content = re.search(r"```yaml(.*?)```", text, re.DOTALL)
    if content:
        return content.group(1).strip()
    else:
        return None


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
    model: str = "gpt-oss:120b",
    gpu: str = "MI210",
    verbose: bool = False,
) -> None:
    """Generate a logic yaml from a kernel config for tuning a GEMM kernel using Tensile"""

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logging.info(f"Config file: {config_yaml}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Logic file: {logic_yaml}")

    config = Path(config_yaml).read_text()

    if logic_yaml is None:
        prompt = get_user_prompt_v1(config, f"{GPU_SPEC_INFO[gpu]}")
        res: dict[str, Any] = get_llm_response_parsed(model, prompt)
        output_yaml = yaml.safe_dump(res, sort_keys=False)
    else:
        logic = Path(logic_yaml).read_text()
        prompt = get_user_prompt_v2(config, f"{GPU_SPEC_INFO[gpu]}", logic)
        res: str = get_llm_response(model, prompt)
        output_yaml = extract_yaml(res)

    if output_file in ["stdout", "-", "", None]:
        print(res)
        print(type(res))
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path.with_suffix(".md"), "w") as f:
        f.write(f"## Prompt\n\n{prompt}\n")
        if logic_yaml is None:
            f.write(f"## Response\n\n```yaml\n{output_yaml}\n```\n")
        else:
            f.write(f"## Response\n\n{res}\n")

    with open(output_path.with_suffix(".yaml"), "w") as f:
        f.write(output_yaml)


@app.command()
def tensile(
    args: Annotated[
        list[str],
        typer.Argument(
            help="arguments passed to Tensile.sh (e.g. config_file output_path)"
        ),
    ],
) -> None:
    """Run Tensile tuning application with given config file"""

    cmd = ["/mnt/rocm-libraries/projects/hipblaslt/build-tensile/Tensile.sh"] + args

    logging.info(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Tensile output: {res.stdout}")


if __name__ == "__main__":
    app()
