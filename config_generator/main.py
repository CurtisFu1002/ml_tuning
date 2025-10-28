import argparse
import logging
import re
from pathlib import Path
import subprocess
from typing_extensions import Annotated

from ollama import chat
import typer
import yaml

from config_generator.prompts.gpu_spec import GPU_SPEC_INFO


app = typer.Typer()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune a GEMM kernel using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
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


def get_user_prompt_v1(config: str, gpu_spec: str) -> str:
    prompt = f"""
You are a performance engineer at AMD working on optimizing the GEMM kernels for hipBLASLt in ROCm libraries.

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
4. Modified config in YAML format surrounded by triple backticks (```yaml)
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
    config_file: str,
    logic_file: str,
    output_file: str,
    model: str = "llama3.1:8b",
    verbose: bool = False,
):
    """Generate a logic yaml from a kernel config for tuning a GEMM kernel using Tensile"""

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logging.info(f"Config file: {config_file}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Logic file: {logic_file}")

    with open(Path(config_file), "r") as f:
        config = f.read()

    with open(Path(logic_file), "r") as f:
        logic = f.read()

    # prompt = get_user_prompt_v1(config, f"{GPU_SPEC_INFO['MI210']}")
    prompt = get_user_prompt_v2(config, f"{GPU_SPEC_INFO['MI210']}", logic)
    res = get_llm_response(model, prompt)

    if output_file not in ["stdout", "-", "", None]:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"## Prompt\n{prompt}\n")
            f.write(f"## Response\n\n{res}\n")
    else:
        print(res)

    logic_yaml = extract_yaml(res)
    with open(Path(f"{output_file}.yaml"), "w") as f:
        f.write(logic_yaml)
    print(logic_yaml)


@app.command()
def tensile(
    args: Annotated[
        list[str],
        typer.Argument(
            help="arguments passed to Tensile.sh (e.g. config_file output_path)"
        ),
    ],
):
    """Run Tensile tuning application with given config file"""

    cmd = ["Tensile.sh"] + args

    logging.info(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Tensile output: {res.stdout}")


if __name__ == "__main__":
    app()
