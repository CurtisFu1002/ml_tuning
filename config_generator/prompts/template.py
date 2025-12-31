"""Prompt templates for LLM interactions."""

from config_generator.prompts.message import Message


ROLE = "You are a performance engineer at AMD working on optimizing the GEMM kernels for hipBLASLt in ROCm libraries."

TASK_V1 = "You are trying to tune a GEMM kernel using Tensile tuning framework on AMD's MI210 (gfx90a) GPU."

CONTEXT_V1 = """The input config yaml is:
```yaml
{config}
```

and the GPU spec is:

```python
{gpu_spec}
```"""

ZERO_SHOT_COT_TRIGGER_1 = "Let's think step by step"
ZERO_SHOT_COT_TRIGGER_2 = "Let's work this out in a step by step way to be sure we have the best performance"

FEW_SHOT_COT_EXAMPLES = """For the follow problem sizes, the corresponding winner matrix instructions are:
- Example1
    - ProblemSize: [1472, 576, 1, 4096]
    - MatrixInstruction: [16, 16, 16, 1, 1, 1, 1, 4, 1]
- Example2
    - ProblemSize: [1472, 832, 1, 4096]
    - MatrixInstruction: [16, 16, 16, 1, 1, 3, 1, 4, 1]
- Example3
    - ProblemSize: [1472, 1856, 1, 4096]
    - MatrixInstruction: [16, 16, 16, 1, 1, 7, 1, 2, 2]

Please find the best matrix instruction for the targeting problem size.
"""

def get_user_prompt_v1(config: str, gpu_spec: str) -> str:
    """Generate prompt for config optimization without logic file."""
    return f"""{ROLE}

{TASK_V1}

{CONTEXT_V1.format(config=config, gpu_spec=gpu_spec)}

Please tell me how to modify the config to the better performance with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format

Delete the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!
"""


def get_user_prompt_v1_1(config: str, gpu_spec: str) -> str:
    """Generate prompt for config optimization without logic file."""
    return f"""{ROLE}

{TASK_V1}

{CONTEXT_V1.format(config=config, gpu_spec=gpu_spec)}

{ZERO_SHOT_COT_TRIGGER_1}. Please tell me how to modify the config to the better performance with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format

Delete the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!
"""


def get_user_prompt_v1_2(config: str, gpu_spec: str) -> str:
    """Generate prompt for config optimization without logic file."""
    return f"""{ROLE}

{TASK_V1}

{CONTEXT_V1.format(config=config, gpu_spec=gpu_spec)}

Modify the config to the better performance. {ZERO_SHOT_COT_TRIGGER_2} with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format by deleting the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!
"""


def get_user_prompt_v1_3(config: str, gpu_spec: str) -> str:
    """Generate prompt for config optimization without logic file."""
    return f"""{ROLE}

{TASK_V1}

{CONTEXT_V1.format(config=config, gpu_spec=gpu_spec)}

{ZERO_SHOT_COT_TRIGGER_1}. Please tell me how to modify the config to the better performance with the following structure:

1. Initial observations
2. Potential optimization
3. Expected performance improvement
4. Modified config in YAML format

Delete the MatrixInstruction candidate options that are not valid or sub-optimal for the given GPU, and provide the modified config in YAML format.

Note that the `TestParameters` and `GlobalParameters` should not be changed in the output config!

{FEW_SHOT_COT_EXAMPLES}
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

Note that all the fields in the output logic yaml (in the example) should remain in output, even if it is not changed!
"""


def create_prompts(
    version: str, config: str, gpu_spec: str, logic_text: str = ""
) -> list[Message]:
    print(f"prompt version: {version}")
    match version:
        case "v1":
            user_prompt = get_user_prompt_v1(config, gpu_spec)
        case "v1_1":
            user_prompt = get_user_prompt_v1_1(config, gpu_spec)
        case "v1_2":
            user_prompt = get_user_prompt_v1_2(config, gpu_spec)
        case "v1_3":
            user_prompt = get_user_prompt_v1_3(config, gpu_spec)
        case "v2":
            user_prompt = get_user_prompt_v2(config, gpu_spec, logic_text)
        case _:
            raise ValueError(f"Prompt version {version} is not supported.")
    return [Message(role="user", content=user_prompt)]
