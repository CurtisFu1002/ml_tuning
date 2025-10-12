import argparse
import logging
from pathlib import Path

import yaml
from ollama import chat

from prompts.gpu_spec import GPU_SPEC_INFO


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


def get_user_prompt(config: str, gpu_spec: str) -> str:
    prompt = f"""
    You are a performance engineer at AMD working on optimizing the GEMM kernels for hipBLASLt in ROCm libraries.

    You are trying to tune a GEMM kernel using Tensile tuning framework on AMD's MI210 (gfx90a) GPU.

    The input config yaml is:

    ```yaml
    {config}
    ```

    and the GPU spec is:

    ```json
    {gpu_spec}
    ```

    Please tell me how to modify the config to the better performance.
    """
    return prompt


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info(args)

    with open(Path(args.config_file), "r") as f:
        config = yaml.safe_load(f)

    prompt = get_user_prompt(config, f"{GPU_SPEC_INFO['MI210']}")
    res = get_llm_response(args.model, prompt)

    if args.output_file not in ["stdout", "-", "", None]:
        with open(Path(args.output_file), "w") as f:
            f.write(res)
    else:
        print(res)


if __name__ == "__main__":
    main()
