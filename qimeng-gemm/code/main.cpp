#include <iostream>

#include "gemm.hpp"

#define M 1024
#define N 1024
#define K 1024

int main() {
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    gemm(A, B, C, M, N, K);

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
