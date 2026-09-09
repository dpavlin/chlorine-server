#pragma once
#include <cstdint>

namespace chlorine {

// Dynamic activation quantization: BF16 -> INT4 with FP16 scale per 256 elements
// d_in: [M, K] BF16 (uint16_t)
// d_out_q: [M, K / 2] packed INT4
// d_out_s: [M, K / 256] FP16 scales
void actq(const uint16_t* d_in, int M, int K,
          uint8_t* d_out_q, uint16_t* d_out_s, void* stream = nullptr);

// Hardware INT4 WMMA GEMM on AMD Strix Halo gfx1151:
// Out[M, N] = A[M, K] x W[N, K]^T
// A_q: [M, K/2] INT4 packed
// A_s: [M, K/256] FP16 scales
// W_q: [N, K/2] INT4 packed
// W_s: [N, K/256] FP16 scales
// Out: [M, N] BF16
void gemm_i4(const uint8_t* A_q, const uint16_t* A_s,
             const uint8_t* W_q, const uint16_t* W_s,
             uint16_t* Out, int M, int N, int K, void* stream = nullptr);

// Fused activation quantization + INT4 WMMA GEMM:
// Automatically quantizes d_A using workspace buffers and computes GEMM
void gemm_i4_fused(const uint16_t* d_A,
                   const uint8_t* W_q, const uint16_t* W_s,
                   uint16_t* d_Out, int M, int N, int K,
                   uint8_t* d_Aq_ws, uint16_t* d_As_ws,
                   void* stream = nullptr);

} // namespace chlorine
