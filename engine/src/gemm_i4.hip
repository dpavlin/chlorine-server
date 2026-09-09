// gemm_i4.hip — Hardware INT4 WMMA GEMM on AMD Strix Halo gfx1151
// Clean-room implementation using native HIP intrinsics
#include "gemm_i4.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdint.h>

namespace chlorine {

typedef int __attribute__((ext_vector_type(2))) int2_v;
typedef int __attribute__((ext_vector_type(8))) int8_v;

__device__ inline float bf16f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    return __uint_as_float(u);
}

__device__ inline uint16_t f32bf(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t lo = u & 0xFFFFu;
    return (uint16_t)((u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u))));
}

// -----------------------------------------------------------------------------
// k_actq: Dynamic activation quantization from BF16 into signed INT4 [-7..7]
// with 1 FP16 scale per 256 elements along the K dimension.
// Grid: ((K + 255) / 256, M), Block: 256 threads
// -----------------------------------------------------------------------------
__global__ void k_actq(const uint16_t* __restrict__ X, int rows, int cols,
                       uint8_t* __restrict__ Out_quant, uint16_t* __restrict__ Out_scales) {
    int row = blockIdx.y;
    int col_blk = blockIdx.x * 256;
    if (row >= rows || col_blk >= cols) return;

    int tid = threadIdx.x;
    int c = col_blk + tid;

    float val = (c < cols) ? bf16f(X[(long)row * cols + c]) : 0.f;
    float absv = fabsf(val);

    // Reduction across 256 threads in shared memory to find group max
    __shared__ float s_max[256];
    s_max[tid] = absv;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    float grp_max = s_max[0];
    float scale = grp_max / 7.0f;
    float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;

    // Write scale for group
    if (tid == 0) {
        int scale_idx = row * (cols / 256) + blockIdx.x;
        __half h_scale = __float2half(scale);
        Out_scales[scale_idx] = *(reinterpret_cast<uint16_t*>(&h_scale));
    }

    // Quantize element to [-7..7]
    int q = 0;
    if (scale > 1e-8f) {
        q = __float2int_rn(val * inv_scale);
        if (q > 7) q = 7;
        if (q < -7) q = -7;
    }
    uint8_t nibble = (uint8_t)(q & 0x0F);

    // Pack pairs of adjacent nibbles (tid 2k with tid 2k+1)
    __shared__ uint8_t s_nib[256];
    s_nib[tid] = nibble;
    __syncthreads();

    if ((tid & 1) == 0 && (c < cols)) {
        uint8_t lo = s_nib[tid];
        uint8_t hi = (tid + 1 < 256) ? s_nib[tid + 1] : 0;
        Out_quant[(long)row * (cols / 2) + (c / 2)] = (hi << 4) | (lo & 0x0F);
    }
}

// -----------------------------------------------------------------------------
// k_gemm_i4_core: Hardware INT4 WMMA GEMM
// Block tile: 128 x 128 (8 waves of 32 threads = 256 threads)
// Wave tile: 32 x 64 (4 waves in M, 2 waves in N)
// LDS staging: 128 rows x 32 elements (16 bytes) per slice
// -----------------------------------------------------------------------------
__global__ void k_gemm_i4_core(const uint8_t* __restrict__ A_q, const uint16_t* __restrict__ A_s,
                               const uint8_t* __restrict__ W_q, const uint16_t* __restrict__ W_s,
                               uint16_t* __restrict__ Out, int M, int N, int K) {
    int bx = blockIdx.x * 128;
    int by = blockIdx.y * 128;

    int tid = threadIdx.x;
    int wave_id = tid / 32;
    int lane_id = tid % 32;

    int wave_m = (wave_id % 4) * 32;
    int wave_n = (wave_id / 4) * 64;

    // FP32 accumulators: 2 tiles in M x 4 tiles in N x 8 elements per lane
    float acc_f32[2][4][8];
    for (int tm = 0; tm < 2; tm++)
        for (int tn = 0; tn < 4; tn++)
            for (int e = 0; e < 8; e++)
                acc_f32[tm][tn][e] = 0.f;

    // gfx1151 WMMA lane-to-row mapping inside 16x16 tile:
    // Rows 0..7 map to lanes 0, 2, 4, ..., 14 (lane = 2 * r)
    // Rows 8..15 map to lanes 17, 19, 21, ..., 31 (lane = 17 + 2 * (r - 8))
    int a_r_sub = -1;
    if (lane_id < 16 && (lane_id % 2 == 0)) a_r_sub = lane_id / 2;
    else if (lane_id >= 16 && (lane_id % 2 == 1)) a_r_sub = 8 + (lane_id - 17) / 2;

    // Matrix B / W: lane_id % 16 directly maps to tile column (0..15)
    int b_c_sub = lane_id % 16;

    // LDS staging: 128 rows x 16 bytes (32 INT4 elements), 8 bytes padding
    __shared__ uint8_t s_A[128][24];
    __shared__ uint8_t s_W[128][24];

    // Loop over K in groups of 256 (matching FP16 scale block size)
    for (int kg = 0; kg < K; kg += 256) {
        int8_v acc_i32[2][4];
        for (int tm = 0; tm < 2; tm++)
            for (int tn = 0; tn < 4; tn++)
                acc_i32[tm][tn] = (int8_v){0, 0, 0, 0, 0, 0, 0, 0};

        // 8 slices of 32 elements per 256-element group
        for (int k_slice = 0; k_slice < 256; k_slice += 32) {
            int k = kg + k_slice;

            // Collaborative 128-bit vector loads into LDS
            if (tid < 128) {
                int r = by + tid;
                if (r < M && (k + 32) <= K) {
                    *reinterpret_cast<int4*>(&s_A[tid][0]) =
                        *reinterpret_cast<const int4*>(A_q + (long)r * (K / 2) + k / 2);
                } else {
                    *reinterpret_cast<int4*>(&s_A[tid][0]) = (int4){0, 0, 0, 0};
                }
            } else {
                int col = bx + (tid - 128);
                if (col < N && (k + 32) <= K) {
                    *reinterpret_cast<int4*>(&s_W[tid - 128][0]) =
                        *reinterpret_cast<const int4*>(W_q + (long)col * (K / 2) + k / 2);
                } else {
                    *reinterpret_cast<int4*>(&s_W[tid - 128][0]) = (int4){0, 0, 0, 0};
                }
            }
            __syncthreads();

            // Execute 2 x 4 = 8 WMMA operations for two 16-element sub-steps
            #pragma unroll
            for (int sub_k = 0; sub_k < 2; sub_k++) {
                int byte_off = sub_k * 8;

                int2_v a_frag[2];
                for (int tm = 0; tm < 2; tm++) {
                    a_frag[tm] = (int2_v){0, 0};
                    if (a_r_sub >= 0) {
                        int r = wave_m + tm * 16 + a_r_sub;
                        a_frag[tm] = *reinterpret_cast<const int2_v*>(&s_A[r][byte_off]);
                    }
                }

                int2_v w_frag[4];
                for (int tn = 0; tn < 4; tn++) {
                    int c = wave_n + tn * 16 + b_c_sub;
                    w_frag[tn] = *reinterpret_cast<const int2_v*>(&s_W[c][byte_off]);
                }

                for (int tm = 0; tm < 2; tm++) {
                    for (int tn = 0; tn < 4; tn++) {
                        acc_i32[tm][tn] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                            true, a_frag[tm], true, w_frag[tn], acc_i32[tm][tn], false);
                    }
                }
            }
            __syncthreads();
        }

        // Apply FP16 group scales and accumulate into FP32 registers
        int g_idx = kg / 256;
        for (int tm = 0; tm < 2; tm++) {
            for (int tn = 0; tn < 4; tn++) {
                int col = bx + wave_n + tn * 16 + b_c_sub;
                float w_scale = 0.f;
                if (col < N) {
                    __half hw = *reinterpret_cast<const __half*>(W_s + (long)col * (K / 256) + g_idx);
                    w_scale = __half2float(hw);
                }

                int r_base = (lane_id >= 16) ? 8 : 0;
                for (int e = 0; e < 8; e++) {
                    int r = by + wave_m + tm * 16 + r_base + e;
                    float a_scale = 0.f;
                    if (r < M) {
                        __half ha = *reinterpret_cast<const __half*>(A_s + (long)r * (K / 256) + g_idx);
                        a_scale = __half2float(ha);
                    }
                    acc_f32[tm][tn][e] += (float)acc_i32[tm][tn][e] * (a_scale * w_scale);
                }
            }
        }
    }

    // Epilogue: store FP32 accumulators rounded to BF16
    for (int tm = 0; tm < 2; tm++) {
        for (int tn = 0; tn < 4; tn++) {
            int col = bx + wave_n + tn * 16 + b_c_sub;
            int r_base = (lane_id >= 16) ? 8 : 0;
            for (int e = 0; e < 8; e++) {
                int r = by + wave_m + tm * 16 + r_base + e;
                if (r < M && col < N) {
                    Out[(long)r * N + col] = f32bf(acc_f32[tm][tn][e]);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host entry points
// -----------------------------------------------------------------------------

void actq(const uint16_t* d_in, int M, int K,
          uint8_t* d_out_q, uint16_t* d_out_s, void* stream) {
    dim3 grid((K + 255) / 256, M);
    dim3 block(256);
    hipStream_t s = static_cast<hipStream_t>(stream);
    k_actq<<<grid, block, 0, s>>>(d_in, M, K, d_out_q, d_out_s);
}

void gemm_i4(const uint8_t* A_q, const uint16_t* A_s,
             const uint8_t* W_q, const uint16_t* W_s,
             uint16_t* Out, int M, int N, int K, void* stream) {
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dim3 block(256);
    hipStream_t s = static_cast<hipStream_t>(stream);
    k_gemm_i4_core<<<grid, block, 0, s>>>(A_q, A_s, W_q, W_s, Out, M, N, K);
}

void gemm_i4_fused(const uint16_t* d_A,
                   const uint8_t* W_q, const uint16_t* W_s,
                   uint16_t* d_Out, int M, int N, int K,
                   uint8_t* d_Aq_ws, uint16_t* d_As_ws,
                   void* stream) {
    actq(d_A, M, K, d_Aq_ws, d_As_ws, stream);
    gemm_i4(d_Aq_ws, d_As_ws, W_q, W_s, d_Out, M, N, K, stream);
}

} // namespace chlorine
