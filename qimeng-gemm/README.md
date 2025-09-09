# Experiments for QiMeng-GEMM

## Environment Setup

Install Ollama in Docker container.

```
curl -fsSL https://ollama.com/install.sh | sh
```

Serve Ollama in the background (e.g. `tmux`)

```
ollama serve
```

Download model in another session, take DeepSeek-Coder-V2-13B for example.

```
ollama pull deepseek-coder-v2:16b
```

Read/write files

```
ollama run deepseek-coder-v2:16b "Summarize the content of this file in 10 words." < input.md > output.md
```

## Meta-Prompts Template

The meta-prompt offers templates for various general optimization techniques and platform-specific optimization details. It composed of three parts:

1. platform-agnostic description
2. platform-specific hints
3. instantiation method

### Platform-agnostic Description

General optimization techniques:

- tiling
- reordering
- layout
- vectorization
- pipeline

For each optimization primitive, the corresponding meta-prompts are pre-defined by human experts to decrease the reasoning complexity.

### Platform-specific Hints

Platform-specific hints include:

- natural language description
- code skeleton (few-shot)

tailored for specific hardware characteristics to generate optimized GEMM code across diverse hardware platforms.

### Instantiation Method

The instantiation describes the process from the general meta-prompts to the prompts tailored for specific platform.

## Auto-Tuning

The auto-tuning algorithm in the paper is shown below:

```
Input: a set of meta-prompts p_A, beam width k, max number of iterations T

Output: high-performance GEMM code

subroutine generate_gemm_kernel {
    initialize naive GEMM implementation c0 as the root of tree C <- {c0}

    for t < T {
        for each code c in C {
            LLM suggests optimization primitive candidates

            for each candidate mp {
                instance mp with platform-specific hint
                c^t = LLM(c^{t-1}, mp)
                add c^t to C_new with its performance
            }
        }
        prune C_new to retain only top k performance
        C <- C_new
    }
    return the code in C with highest performance
}
```
