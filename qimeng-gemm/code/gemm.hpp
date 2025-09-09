#ifndef GEMM_H
#define GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

void gemm(const float* A, const float* B, float* C, int m, int n, int k);

#ifdef __cplusplus
}
#endif

#endif  // GEMM_H
