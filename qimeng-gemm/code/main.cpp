#include <cmath>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#include "gemm.hpp"

#define M 1024
#define N 1024
#define K 1024

void gemmCpu(const float* A, const float* B, float* C, int m, int n, int k) {
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            float psum = 0.0f;
            for (uint32_t ki = 0; ki < k; ki++) {
                psum += A[mi * k + ki] * B[ki * n + ni];
            }
            C[mi * m + ni] = psum;
        }
    }
}

uint32_t compareMatrix1d(float* mat1, float* mat2, uint32_t size, float tolerance = 1e-4) {
    uint32_t error_count = 0;
    for (uint32_t i = 0; i < size; i++) {
        if (std::fabs(mat1[i] - mat2[i]) > tolerance) error_count++;
    }
    return error_count;
}

void printMatrix1d(float* a, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        std::cout << a[i] << "  ";
    }
    std::cout << std::endl;
}

uint32_t testGemm(const uint32_t m, const uint32_t n, const uint32_t k) {
    float *in1 = new float[m * k];
    float *in2 = new float[k * n];
    float *out_gpu = new float[m * n];
    float *out_cpu = new float[m * n];

    for (uint32_t i = 0; i < m * k; i++) {
        in1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (uint32_t i = 0; i < k * n; i++) {
        in2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    gemm(in1, in2, out_gpu, m, n, k);
    gemmCpu(in1, in2, out_cpu, m, n, k);
    uint32_t error_count = compareMatrix1d(out_cpu, out_gpu, m * n);

    if (error_count > 0) {
        printMatrix1d(out_cpu, m * n < 10 ? m * n : 10);
        printMatrix1d(out_gpu, m * n < 10 ? m * n : 10);
    }

    delete[] in1;
    delete[] in2;
    delete[] out_gpu;
    delete[] out_cpu;
    return error_count;
}

float benchGemm(const uint32_t m, const uint32_t n, const uint32_t k, uint32_t iterations = 10) {
    hipEvent_t start, end;
    hipEventCreate(&start);
    hipEventCreate(&end);
    hipStream_t stream;
    hipStreamCreate(&stream);
    float elapsed_ms = 0.0f;

    float *in1 = new float[m * k];
    float *in2 = new float[k * n];
    float *out_gpu = new float[m * n];
    float *out_cpu = new float[m * n];

    for (uint32_t i = 0; i < m * k; i++) {
        in1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (uint32_t i = 0; i < k * n; i++) {
        in2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Warm-up
    gemm(in1, in2, out_gpu, m, n, k);

    hipEventRecord(start, stream);
    for (uint32_t i = 0; i < iterations; ++i) {
        gemm(in1, in2, out_gpu, m, n, k);
    }
    hipEventRecord(end, stream);
    hipEventSynchronize(end);
    hipEventElapsedTime(&elapsed_ms, start, end);

    hipEventDestroy(start);
    hipEventDestroy(end);
    hipStreamDestroy(stream);

    gemmCpu(in1, in2, out_cpu, m, n, k);
    uint32_t error_count = compareMatrix1d(out_cpu, out_gpu, m * n);
    std::cout << "Error count: " << error_count << std::endl;

    delete[] in1;
    delete[] in2;
    delete[] out_gpu;
    delete[] out_cpu;

    // Return average time per iteration
    return elapsed_ms / iterations;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test|bench>" << std::endl;
        exit(EXIT_FAILURE);
    }

    const std::string mode = argv[1];
    if (mode == "test") {
        uint32_t error_count = testGemm(M, N, K);
        if (error_count == 0) {
            std::cout << "Test passed!" << std::endl;
        } else {
            std::cout << "Test failed! Error count: " << error_count << std::endl;
        }
    } else if (mode == "bench") {
        float elapsed_ms = benchGemm(M, N, K);
        std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
