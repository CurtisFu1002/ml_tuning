#include <hip/hip_runtime.h>

#include "gemm.hpp"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ void gemmKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void gemm(const float* A, const float* B, float* C, int m, int n, int k) {
    float *d_A, *d_B, *d_C;

    HIP_ASSERT(hipMalloc((void**)&d_A, m * k * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_B, k * n * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_C, m * n * sizeof(float)));

    HIP_ASSERT(hipMemcpy(d_A, A, m * k * sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_B, B, k * n * sizeof(float), hipMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    gemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

    HIP_ASSERT(hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost));

    HIP_ASSERT(hipFree(d_A));
    HIP_ASSERT(hipFree(d_A));
    HIP_ASSERT(hipFree(d_A));
}
