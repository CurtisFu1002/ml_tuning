import argparse
from pathlib import Path
import subprocess

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
        "--output",
        type=str,
        default="code/generated",
        help="output directory for generated code",
    )
    parser.add_argument(
        "--build", type=str, default="code/build", help="build directory for make"
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

Return ONLY the C++/HIP code for each variant, NO Markdown formatting.
Remember to add the necessary header `#include "gemm.hpp"`!
Each variant is separated by a line of '---' characters.
"""


def get_llm_response(model: str, input_code: str, meta_prompt: str) -> list[str]:
    user_prompt = get_user_prompt(input_code, meta_prompt)
    resp = chat(model=model, messages=[{"role": "user", "content": user_prompt}])
    resp_list = resp.message.content.split("---")
    return resp_list


def run_benchmark(kernel_source: Path, build_dir: Path) -> subprocess.CompletedProcess:
    topdir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", f"KERNEL={kernel_source}", f"BUILD_DIR={build_dir}", "bench"],
        cwd=f"{topdir}/code",
        check=True,
        capture_output=True,
        text=True,
    )
    return result


def main():
    args = parse_args()
    input_code = Path(args.input).absolute()
    meta_prompt = Path(args.meta).absolute()
    build_dir = Path(args.build).absolute()
    out_dir = Path(args.output).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(input_code, build_dir)
    print(result.stdout)

    unvisited = [input_code.read_text()]
    visited = set()

    for i in range(args.iter):
        code = unvisited.pop(0)
        visited.add(code)

        resp_list = get_llm_response(args.model, code, meta_prompt.read_text())
        unvisited.extend(resp_list)

        for idx, code in enumerate(resp_list):
            code = code.replace("```cpp", "").replace("```", "").strip()
            if not code:
                continue

            output_path = f"{out_dir}/{Path(args.input).stem}-{i}-{idx}.cpp"
            print(f"Writing to {output_path}")
            print(code, file=open(output_path, "w"))

            result = run_benchmark(output_path, build_dir)
            print(result.stdout)


if __name__ == "__main__":
    main()
