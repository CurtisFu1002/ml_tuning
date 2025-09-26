# Meta-Prompt for Tiling on AMD GPU

## Component 1: platform-agnostic description

Tiling the matrix to small pieces.

## Component 2: platform-specific hints

### Specific tiling method on AMD GPU

Wavefront Tiling: Wavefront is a collection of threads in compute units (CU) that executes in lock step. Typically, a wavefront contains 64 threads. There are a shared memory shared by all threads in a CU. Add wavefront-level tiling to optimize the GEMM kernel.

### Specific tiling code on AMD GPU

```cpp
int wid = tid / 64;

for (int wm = 0; ...)
    for (int wn = 0; ...)
        for (int i, ...)
            for (int j, ...)
                for (int k; ...)
```
