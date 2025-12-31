"""Utility functions for config generation."""

import re
from pathlib import Path

import yaml


def extract_yaml(text: str) -> str | None:
    """Extract YAML content from markdown code blocks."""
    content = re.search(r"```yaml(.*?)```", text, re.DOTALL)
    return content.group(1).strip() if content else None


def write_output_files(
    output_path: Path,
    prompt: str,
    response: str,
    yaml_content: str,
    is_logic: bool = False,
) -> None:
    """Write markdown and YAML output files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write markdown file
    with open(output_path.with_suffix(".md"), "w") as f:
        f.write(f"## Prompt\n\n{prompt}\n")
        f.write(f"## Response\n")
        if is_logic:
            f.write(f"\n{response}\n")
        else:
            text = yaml.safe_load(yaml_content)
            f.write(
                f"\nNumber of `MatrixInstruction` generated: {len(text['BenchmarkProblems'][0][1]['ForkParameters'][0]['MatrixInstruction'])}\n"
            )
            f.write(f"\n```yaml\n{yaml_content}\n```\n")

    # Write YAML file
    with open(output_path.with_suffix(".yaml"), "w") as f:
        f.write(yaml_content)
