import argparse
from pathlib import Path

from ollama import chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-coder-v2:16b",
        help="model name or path to model",
    )
    parser.add_argument(
        "--input", type=str, default="code/gemm.cpp", help="input code file"
    )
    parser.add_argument(
        "--meta", type=str, default="meta-prompts/tiling.md", help="meta prompt file"
    )
    parser.add_argument(
        "--iter", type=int, default=1, help="number of iterations to run"
    )
    parser.add_argument(
        "--output", type=str, default="generated", help="output directory"
    )
    return parser.parse_args()


def get_user_prompt(input_code: str, meta_prompt: str) -> str:
    return f"""
You are an expert performance optimization software engineer.
Given the following GEMM implementation and optimization technique, write a new optimized version of the GEMM implementation that uses the optimization technique.

Optimization technique:

{meta_prompt}

GEMM implementation:

```cpp
{input_code}
```

Return ONLY the C++/HIP code for each variants, NO Markdown formatting. Each variant is seperated by a line of '---' characters.
"""


def get_llm_response(model: str, input_code: str, meta_prompt: str) -> str:
    user_prompt = get_user_prompt(input_code, meta_prompt)
    resp = chat(model=model, messages=[{"role": "user", "content": user_prompt}])
    resp_list = resp.message.content.split("---")
    return resp_list


def main():
    args = parse_args()
    input_code = Path(args.input).read_text()
    meta_prompt = Path(args.meta).read_text()

    unvisited = [input_code]
    visited = set()

    for i in range(args.iter):
        code = unvisited.pop(0)
        visited.add(code)

        resp_list = get_llm_response(args.model, code, meta_prompt)
        unvisited.extend(resp_list)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, code in enumerate(resp_list):
            code = code.replace("```cpp", "").replace("```", "").strip()
            if code:
                output_path = f"{out_dir}/{Path(args.input).stem}-{i}-{idx}.cpp"
                print(f"Writing to {output_path}")
                print(code, file=open(output_path, "w"))


if __name__ == "__main__":
    main()
