# Evaluation of Generalization Across Problem Sizes

To evaluate the robustness of our LLM-guided tuning framework, we initially conducted experiments on a relatively constrained problem size, specifically `[256, 128, 1,4096]`. To test the model's ability to generalize, we employed a **few-shot prompting strategy** by incorporating three significantly larger and more complex problem sizes into the context:

- `[1472, 576, 1, 4096]`
- `[1472, 832, 1, 4096]`
- `[1472, 1856, 1, 4096]`

By providing these representative examples, the **winner consistency** — defined as the LLM's ability to correctly identify the optimal configuration — improved to **70%**. While this preliminary result suggests that in-context learning effectively helps the model capture the performance characteristics of the GPU kernels, it also highlights a potential limitation in **out-of-distribution generalization**. The current consistency gains may fluctuate when the model is tasked with predicting performance for problem sizes that differ drastically in aspect ratio or tiling behavior from the few-shot examples provided.

## Future Work

Moving forward, we plan to extend our evaluation to a more diverse set of problem sizes to ensure the model's predictive reliability. By broadening the test coverage, we aim to assess whether the current winner consistency can be maintained or further optimized through more sophisticated prompt engineering or fine-tuning strategies.
