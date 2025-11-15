"""Prompt templates for LLM interactions."""


def get_user_prompt_v1(config: str, gpu_spec: str) -> str:
    """Generate prompt for config optimization without logic file."""
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

Please tell me how to modify the config to the better performance with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format

Delete the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!
"""


def get_user_prompt_v2(config: str, gpu_spec: str, logic: str) -> str:
    """Generate prompt for logic file generation."""
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
