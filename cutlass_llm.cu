#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h> // For mma.sync intrinsics
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip> // For std::setprecision

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Configuration (deduced from PTX analysis) ---
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int MMA_M = 8; // Using standard Ampere m8n8k4 for bf16
constexpr int MMA_N = 8;
constexpr int MMA_K = 4; // Using standard Ampere m8n8k4 for bf16
constexpr int NumStages = 2;

// Try to match PTX block dims if possible - need 256 threads
constexpr int BLOCK_DIM_X = 128; // Threads along warp N dimension?
constexpr int BLOCK_DIM_Y = 2;   // Threads along warp M dimension?
constexpr int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y; // 256

// --- Derived Constants ---
constexpr int WARPS_PER_BLOCK_M = BLOCK_M / WARP_M; // 128 / 32 = 4
constexpr int WARPS_PER_BLOCK_N = BLOCK_N / WARP_N; // 128 / 64 = 2
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 4 * 2 = 8
constexpr int THREADS_PER_WARP = 32;

// Accumulator tile size per thread (per MMA op)
constexpr int ACCUM_M_PER_MMA = MMA_M;
constexpr int ACCUM_N_PER_MMA = MMA_N;

// How many MMA ops vertically/horizontally per thread in a warp tile?
// This depends on how threads map to the MMA fragments. WMMA API abstracts this,
// but for manual accumulator storage, we need the effective tile per thread.
// Assuming a thread contributes to multiple 8x8 MMAs within the 32x64 warp tile.
// E.g., Each thread handles 4x8 = 32 elements.
constexpr int ACCUM_M_PER_THREAD = WARP_M / MMA_M; // 32 / 8 = 4 (across 4 8x8 blocks vertically)
constexpr int ACCUM_N_PER_THREAD = WARP_N / MMA_N; // 64 / 8 = 8 (across 8 8x8 blocks horizontally)
constexpr int ACCUM_ELEMS_PER_THREAD = ACCUM_M_PER_THREAD * ACCUM_N_PER_THREAD; // 4 * 8 = 32

// Shared memory padding (bytes) - minimal padding for alignment
constexpr int SMEM_ALIGNMENT_BYTES = 16; // Align buffers to 16 bytes
constexpr int SMEM_A_PADDING = (SMEM_ALIGNMENT_BYTES / sizeof(__nv_bfloat16)) - 1; // ~7 elements
constexpr int SMEM_B_PADDING = (SMEM_ALIGNMENT_BYTES / sizeof(__nv_bfloat16)) - 1; // ~7 elements

// Shared memory tile dimensions
constexpr int SMEM_M = BLOCK_M;
constexpr int SMEM_N = BLOCK_N;
constexpr int SMEM_K = BLOCK_K;

// Size in elements
constexpr size_t SMEM_A_TILE_SIZE = SMEM_M * SMEM_K; // 128 * 8 = 1024
constexpr size_t SMEM_B_TILE_SIZE = SMEM_K * SMEM_N; // 8 * 128 = 1024 (Stored KxN)

// Total shared memory per buffer (including padding, aligned)
constexpr size_t SMEM_A_BUFFER_SIZE_ELEM = ((SMEM_A_TILE_SIZE * sizeof(__nv_bfloat16) + SMEM_ALIGNMENT_BYTES - 1) / SMEM_ALIGNMENT_BYTES) * SMEM_ALIGNMENT_BYTES / sizeof(__nv_bfloat16);
constexpr size_t SMEM_B_BUFFER_SIZE_ELEM = ((SMEM_B_TILE_SIZE * sizeof(__nv_bfloat16) + SMEM_ALIGNMENT_BYTES - 1) / SMEM_ALIGNMENT_BYTES) * SMEM_ALIGNMENT_BYTES / sizeof(__nv_bfloat16);


// --- Kernel Parameters (Reverse Engineered) ---
struct GemmParams {
    int M;
    int N;
    int K;
    // PTX only showed M, N, K counts, swizzle param not directly used here
    // int swizzle_log_tile;

    const __nv_bfloat16 *ptr_A;
    long long lda;       // Leading dimension (stride between rows for RowMajor)
    long long stride_a;  // Stride between batches (if any)

    const __nv_bfloat16 *ptr_B;
    long long ldb;       // Leading dimension (stride between rows for ColMajor - K elements)
    long long stride_b;

    const __nv_bfloat16 *ptr_C;
    long long ldc;
    long long stride_c;

     __nv_bfloat16 *ptr_D;
    long long ldd;
    long long stride_d;

    // Pass alpha/beta by pointer as suggested by PTX load pattern
    const __nv_bfloat16* ptr_alpha;
    const __nv_bfloat16* ptr_beta;
};

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
pure_cuda_gemm_bf16(GemmParams params)
{
    // Shared memory layout: | A[0] | A[1] | B[0] | B[1] |
    extern __shared__ char s_mem_raw[];
    __nv_bfloat16* s_A_raw = reinterpret_cast<__nv_bfloat16*>(s_mem_raw);
    __nv_bfloat16* s_B_raw = s_A_raw + NumStages * SMEM_A_BUFFER_SIZE_ELEM;

    // --- Thread Indexing ---
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = thread_id / THREADS_PER_WARP;       // 0..7
    const int lane_id = thread_id % THREADS_PER_WARP;       // 0..31

    // Map block index to M and N coordinates
    // Assumes gridDim.x corresponds to M-tiles, gridDim.y to N-tiles
    int blockIdx_m = blockIdx.x;
    int blockIdx_n = blockIdx.y;
    int batch_idx = blockIdx.z;

    // Calculate start offsets for A, B, C, D in global memory for this block
    // A is RowMajor: M rows, K cols
    const __nv_bfloat16 *g_A = params.ptr_A + batch_idx * params.stride_a
                             + blockIdx_m * BLOCK_M * params.lda; // Row offset
    // B is ColMajor: K rows, N cols
    const __nv_bfloat16 *g_B = params.ptr_B + batch_idx * params.stride_b
                             + blockIdx_n * BLOCK_N;             // Col offset
    // D is RowMajor: M rows, N cols
          __nv_bfloat16 *g_D = params.ptr_D + batch_idx * params.stride_d
                             + blockIdx_m * BLOCK_M * params.ldd // Row offset
                             + blockIdx_n * BLOCK_N;             // Col offset
    // C is RowMajor: M rows, N cols
    const __nv_bfloat16 *g_C = params.ptr_C + batch_idx * params.stride_c
                             + blockIdx_m * BLOCK_M * params.ldc // Row offset
                             + blockIdx_n * BLOCK_N;             // Col offset

    // --- Accumulator Initialization ---
    // WMMA fragments for accumulator tiles (bf16 accumulates, as seen in PTX)
    using MmaFragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, __nv_bfloat16>;
    MmaFragC accum_frags[ACCUM_M_PER_THREAD][ACCUM_N_PER_THREAD]; // 4x8 array of 8x8 fragments per thread

    #pragma unroll
    for (int m = 0; m < ACCUM_M_PER_THREAD; ++m) {
        #pragma unroll
        for (int n = 0; n < ACCUM_N_PER_THREAD; ++n) {
            nvcuda::wmma::fill_fragment(accum_frags[m][n], __float2bfloat16(0.0f));
        }
    }

    // --- Pipelined Main Loop ---
    int k_tiles = (params.K + BLOCK_K - 1) / BLOCK_K;
    int k_tile_iter = 0;

    // --- Stage 1: Load first tile into SMEM buffer 0 ---
    {
        int k_start = 0;
        __nv_bfloat16* s_A_write_ptr = s_A_raw; // Buffer 0
        __nv_bfloat16* s_B_write_ptr = s_B_raw; // Buffer 0

        // Parallel load using all threads in the block
        int load_a_elems_total = SMEM_M * SMEM_K;
        int load_b_elems_total = SMEM_K * SMEM_N; // B is KxN in SMEM

        // Load A (MxK tile)
        for (int i = thread_id; i < load_a_elems_total; i += THREADS_PER_BLOCK) {
            int m = i / SMEM_K; // row in tile
            int k = i % SMEM_K; // col in tile (k-dim)
            int g_m = blockIdx_m * BLOCK_M + m;
            int g_k = k_start + k;
            if (g_m < params.M && g_k < params.K) {
                s_A_write_ptr[m * SMEM_K + k] = g_A[m * params.lda + g_k];
            } else {
                s_A_write_ptr[m * SMEM_K + k] = __float2bfloat16(0.0f); // Padding
            }
        }

        // Load B (KxN tile into KxN SMEM layout)
        for (int i = thread_id; i < load_b_elems_total; i += THREADS_PER_BLOCK) {
            int k = i / SMEM_N; // row in tile (k-dim)
            int n = i % SMEM_N; // col in tile
            int g_k = k_start + k;
            int g_n = blockIdx_n * BLOCK_N + n;
            if (g_k < params.K && g_n < params.N) {
                // Global B is ColMajor (K rows, N cols), read g_B[g_k, g_n]
                s_B_write_ptr[k * SMEM_N + n] = g_B[g_k * params.ldb + g_n];
            } else {
                s_B_write_ptr[k * SMEM_N + n] = __float2bfloat16(0.0f); // Padding
            }
        }
    }
    __syncthreads(); // Ensure first tile is loaded

    // --- Main K loop ---
    for (k_tile_iter = 0; k_tile_iter < k_tiles - 1; ++k_tile_iter)
    {
        // Set read pointers to current buffer, write pointers to next buffer
        int read_stage = k_tile_iter % NumStages;
        int write_stage = (k_tile_iter + 1) % NumStages;
        const __nv_bfloat16* s_A_read_ptr = s_A_raw + read_stage * SMEM_A_BUFFER_SIZE_ELEM;
        const __nv_bfloat16* s_B_read_ptr = s_B_raw + read_stage * SMEM_B_BUFFER_SIZE_ELEM;
              __nv_bfloat16* s_A_write_ptr = s_A_raw + write_stage * SMEM_A_BUFFER_SIZE_ELEM;
              __nv_bfloat16* s_B_write_ptr = s_B_raw + write_stage * SMEM_B_BUFFER_SIZE_ELEM;

        // --- Stage 1: Load k+1 tile ---
        {
            int k_start_next = (k_tile_iter + 1) * BLOCK_K;
            const __nv_bfloat16* g_A_next = g_A + k_start_next; // Advance A by K columns
            const __nv_bfloat16* g_B_next = g_B + k_start_next * params.ldb; // Advance B by K rows

            int load_a_elems_total = SMEM_M * SMEM_K;
            int load_b_elems_total = SMEM_K * SMEM_N;

            for (int i = thread_id; i < load_a_elems_total; i += THREADS_PER_BLOCK) {
                 int m = i / SMEM_K; int k = i % SMEM_K;
                 int g_m = blockIdx_m * BLOCK_M + m; int g_k = k_start_next + k;
                 if (g_m < params.M && g_k < params.K) {
                     s_A_write_ptr[m * SMEM_K + k] = g_A_next[m * params.lda + k]; // k is offset within tile
                 } else { s_A_write_ptr[m * SMEM_K + k] = __float2bfloat16(0.0f); }
            }
            for (int i = thread_id; i < load_b_elems_total; i += THREADS_PER_BLOCK) {
                 int k = i / SMEM_N; int n = i % SMEM_N;
                 int g_k = k_start_next + k; int g_n = blockIdx_n * BLOCK_N + n;
                 if (g_k < params.K && g_n < params.N) {
                     s_B_write_ptr[k * SMEM_N + n] = g_B_next[k * params.ldb + g_n]; // Read B[g_k, g_n]
                 } else { s_B_write_ptr[k * SMEM_N + n] = __float2bfloat16(0.0f); }
            }
        } // End Stage 1 (Load)

        // --- Stage 2: Compute using k tile ---
        // Loop over K within the tile using MMA_K stride
        #pragma unroll
        for (int k_frag_idx = 0; k_frag_idx < BLOCK_K; k_frag_idx += MMA_K) // BLOCK_K=8, MMA_K=4 -> 2 iterations
        {
            // Define MMA fragment types for this K-iteration
            using MmaFragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, nvcuda::wmma::row_major>;
            using MmaFragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, nvcuda::wmma::row_major>; // SMEM B is KxN -> RowMajor view needed for MMA load

            MmaFragA frag_a[ACCUM_M_PER_THREAD]; // Thread needs 4 A fragments (for M=32)
            MmaFragB frag_b[ACCUM_N_PER_THREAD]; // Thread needs 8 B fragments (for N=64)

            // Load fragments for all MMAs this thread participates in
            #pragma unroll
            for (int m = 0; m < ACCUM_M_PER_THREAD; ++m) {
                // Calculate SMEM row address for A fragment load
                // Needs careful mapping: warp_id -> warp base row; lane_id -> offset within warp
                int warp_m_base = (warp_id / WARPS_PER_BLOCK_N) * WARP_M;
                int mma_m_offset = m * MMA_M; // Which 8x8 block vertically
                // Map lane to row within the 8x8 MMA A fragment (this is complex, simplified)
                int smem_a_row = warp_m_base + mma_m_offset + (lane_id % MMA_M); // Highly simplified mapping
                const __nv_bfloat16* a_frag_ptr = s_A_read_ptr + smem_a_row * SMEM_K + k_frag_idx;
                nvcuda::wmma::load_matrix_sync(frag_a[m], a_frag_ptr, SMEM_K);
            }
            #pragma unroll
            for (int n = 0; n < ACCUM_N_PER_THREAD; ++n) {
                 // Calculate SMEM row/col address for B fragment load (SMEM B is KxN)
                int warp_n_base = (warp_id % WARPS_PER_BLOCK_N) * WARP_N;
                int mma_n_offset = n * MMA_N;
                // Map lane to K-row within the 8x8 MMA B fragment (simplified)
                int smem_b_k_row = k_frag_idx + (lane_id / (MMA_M * MMA_N / 2)); // Lane mapping over K
                int smem_b_col = warp_n_base + mma_n_offset + (lane_id % MMA_N); // Simplified
                const __nv_bfloat16* b_frag_ptr = s_B_read_ptr + smem_b_k_row * SMEM_N + smem_b_col;
                nvcuda::wmma::load_matrix_sync(frag_b[n], b_frag_ptr, SMEM_N); // Treat KxN block as RowMajor for load
            }

            // Perform MMAs
            #pragma unroll
            for (int m = 0; m < ACCUM_M_PER_THREAD; ++m) {
                #pragma unroll
                for (int n = 0; n < ACCUM_N_PER_THREAD; ++n) {
                    nvcuda::wmma::mma_sync(accum_frags[m][n], frag_a[m], frag_b[n], accum_frags[m][n]);
                }
            }
        } // k_frag_idx

        __syncthreads(); // Wait for compute & k+1 load before starting k+2 load

    } // End Main K loop (k_tile_iter)

    // --- Process Last Tile ---
    {
        int read_stage = k_tile_iter % NumStages; // Last loaded tile
        const __nv_bfloat16* s_A_read_ptr = s_A_raw + read_stage * SMEM_A_BUFFER_SIZE_ELEM;
        const __nv_bfloat16* s_B_read_ptr = s_B_raw + read_stage * SMEM_B_BUFFER_SIZE_ELEM;

        #pragma unroll
        for (int k_frag_idx = 0; k_frag_idx < BLOCK_K; k_frag_idx += MMA_K)
        {
            using MmaFragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, nvcuda::wmma::row_major>;
            using MmaFragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, nvcuda::wmma::row_major>;
            MmaFragA frag_a[ACCUM_M_PER_THREAD];
            MmaFragB frag_b[ACCUM_N_PER_THREAD];

             #pragma unroll
            for (int m = 0; m < ACCUM_M_PER_THREAD; ++m) {
                int warp_m_base = (warp_id / WARPS_PER_BLOCK_N) * WARP_M;
                int mma_m_offset = m * MMA_M;
                int smem_a_row = warp_m_base + mma_m_offset + (lane_id % MMA_M); // Simplified
                const __nv_bfloat16* a_frag_ptr = s_A_read_ptr + smem_a_row * SMEM_K + k_frag_idx;
                nvcuda::wmma::load_matrix_sync(frag_a[m], a_frag_ptr, SMEM_K);
            }
            #pragma unroll
            for (int n = 0; n < ACCUM_N_PER_THREAD; ++n) {
                int warp_n_base = (warp_id % WARPS_PER_BLOCK_N) * WARP_N;
                int mma_n_offset = n * MMA_N;
                int smem_b_k_row = k_frag_idx + (lane_id / (MMA_M * MMA_N / 2)); // Simplified
                int smem_b_col = warp_n_base + mma_n_offset + (lane_id % MMA_N); // Simplified
                const __nv_bfloat16* b_frag_ptr = s_B_read_ptr + smem_b_k_row * SMEM_N + smem_b_col;
                nvcuda::wmma::load_matrix_sync(frag_b[n], b_frag_ptr, SMEM_N);
            }

            #pragma unroll
            for (int m = 0; m < ACCUM_M_PER_THREAD; ++m) {
                #pragma unroll
                for (int n = 0; n < ACCUM_N_PER_THREAD; ++n) {
                    nvcuda::wmma::mma_sync(accum_frags[m][n], frag_a[m], frag_b[n], accum_frags[m][n]);
                }
            }
        }
    } // End compute last tile

    // --- Epilogue ---
    __nv_bfloat16 alpha = *params.ptr_alpha; // Dereference device pointer
    __nv_bfloat16 beta  = *params.ptr_beta;  // Dereference device pointer
    bool beta_is_zero = (__bfloat162float(beta) == 0.0f);

    // Each thread stores its portion of the output tile
    #pragma unroll
    for (int m_frag = 0; m_frag < ACCUM_M_PER_THREAD; ++m_frag) {
         #pragma unroll
         for (int n_frag = 0; n_frag < ACCUM_N_PER_THREAD; ++n_frag) {
            // Convert accumulator fragment back to temporary storage for element access
            __nv_bfloat16 accum_tile[ACCUM_M_PER_MMA * ACCUM_N_PER_MMA]; // 8x8=64 elements
            nvcuda::wmma::store_matrix_sync(accum_tile, accum_frags[m_frag][n_frag], ACCUM_N_PER_MMA, nvcuda::wmma::mem_row_major);

            // Map fragment elements back to global D coordinates
            // This mapping depends heavily on the WMMA layout and lane participation
            // A simplified 1:1 mapping is used here, assuming each thread "owns"
            // specific output elements derived from its accumulator fragments.
             #pragma unroll
             for (int i = 0; i < ACCUM_M_PER_MMA * ACCUM_N_PER_MMA; ++i) {
                 if (accum_frags[m_frag][n_frag].lane_id() == lane_id) // Only store elements this lane is responsible for
                 {
                    int m_in_frag = accum_frags[m_frag][n_frag]. MmaFragC::get_row_index(i);
                    int n_in_frag = accum_frags[m_frag][n_frag]. MmaFragC::get_col_index(i);

                    int warp_m_base = (warp_id / WARPS_PER_BLOCK_N) * WARP_M;
                    int warp_n_base = (warp_id % WARPS_PER_BLOCK_N) * WARP_N;

                    int m = warp_m_base + m_frag * MMA_M + m_in_frag; // Global M within block tile
                    int n = warp_n_base + n_frag * MMA_N + n_in_frag; // Global N within block tile

                    int g_m = blockIdx_m * BLOCK_M + m;
                    int g_n = blockIdx_n * BLOCK_N + n;

                    // Boundary check
                    if (g_m < params.M && g_n < params.N) {
                        __nv_bfloat16 acc_val = accum_tile[i]; // Value from this thread's fragment storage
                        __nv_bfloat16 scaled_acc = __hmul(alpha, acc_val);
                        __nv_bfloat16 result_d;

                        if (!beta_is_zero) {
                             // Note: Loading C individually per thread can be slow.
                             // A shared memory load & WMMA approach would be better for C.
                             // But for simplicity matching PTX epilogue structure:
                            __nv_bfloat16 c_val = g_C[m * params.ldc + n];
                            result_d = __hadd(__hmul(beta, c_val), scaled_acc);
                        } else {
                            result_d = scaled_acc;
                        }
                        g_D[m * params.ldd + n] = result_d;
                    }
                 }
             } // end loop over fragment elements
        }
    }
}


// --- Helper function to calculate shared memory size ---
size_t get_shared_memory_size() {
     size_t size_a = NumStages * SMEM_A_BUFFER_SIZE_ELEM * sizeof(__nv_bfloat16);
     size_t size_b = NumStages * SMEM_B_BUFFER_SIZE_ELEM * sizeof(__nv_bfloat16);
     return size_a + size_b;
}

// --- Naive Host Implementation (using float for simplicity) ---
void naive_matmul_bf16_host(int M, int N, int K,
                            float alpha_f, float beta_f,
                            const __nv_bfloat16* A, int lda, // A is RowMajor
                            const __nv_bfloat16* B, int ldb, // B is ColMajor
                            const __nv_bfloat16* C, int ldc, // C is RowMajor
                                  __nv_bfloat16* D, int ldd) // D is RowMajor
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) { // Row index for A, C, D
        for (int j = 0; j < N; ++j) { // Column index for B, C, D
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i, k] (RowMajor)
                float a_val = __bfloat162float(A[i * lda + k]);
                // B[k, j] (ColMajor)
                float b_val = __bfloat162float(B[k * ldb + j]);
                acc += a_val * b_val;
            }

            float c_val = __bfloat162float(C[i * ldc + j]);
            float result_f = alpha_f * acc + beta_f * c_val;

            D[i * ldd + j] = __float2bfloat16(result_f);
        }
    }
}

// --- Comparison Function ---
bool compare_matrices_bf16(int M, int N,
                           const __nv_bfloat16* gpu_result,
                           const __nv_bfloat16* cpu_result, int ldd,
                           float tolerance = 1e-1) // BF16 tolerance needs to be higher
{
    bool match = true;
    #pragma omp parallel for reduction(&:match)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = __bfloat162float(gpu_result[i * ldd + j]);
            float cpu_val = __bfloat162float(cpu_result[i * ldd + j]);
            float diff = std::fabs(gpu_val - cpu_val);

            // Simple absolute tolerance check
            if (diff > tolerance) {
                // Add relative check for robustness if needed
                // float max_val = std::max(std::fabs(gpu_val), std::fabs(cpu_val));
                // if (diff > tolerance && diff / (max_val + 1e-6f) > tolerance) {
                    if (match) { // Print only the first mismatch found by this thread
                         #pragma omp critical
                         {
                            if (match) { // Double check inside critical section
                                std::cerr << "Mismatch found at (" << i << ", " << j << "): "
                                          << "GPU=" << gpu_val << ", CPU=" << cpu_val << ", Diff=" << diff << std::endl;
                                match = false;
                            }
                         }
                    }
                // }
            }
        }
    }
    return match;
}


// --- Main Function ---
int main() {
    // Problem Dimensions (Make multiples of block size for simplicity here)
    int M = 256; // Must be multiple of BLOCK_M (128)
    int N = 256; // Must be multiple of BLOCK_N (128)
    int K = 128; // Multiple of BLOCK_K (8) is good

    // Leading dimensions (assuming tightly packed matrices)
    int lda = K; // A (MxK) is RowMajor
    int ldb = K; // B (KxN) is ColMajor
    int ldc = N; // C (MxN) is RowMajor
    int ldd = N; // D (MxN) is RowMajor

    // Host alpha/beta
    float alpha_f = 1.5f;
    float beta_f  = 0.5f;
    __nv_bfloat16 alpha_bf16 = __float2bfloat16(alpha_f);
    __nv_bfloat16 beta_bf16  = __float2bfloat16(beta_f);


    // --- Memory Allocation ---
    size_t size_A = (size_t)M * lda;
    size_t size_B = (size_t)K * ldb; // Note: B is K rows x N cols storage = K * ldb (where ldb=N) - Error here! B is KxN stored ColMajor, size = K * N, ldb = K
    size_B = (size_t)K * N; // Correct size for KxN ColMajor B
    ldb = K; // Correct LDB for ColMajor
    size_t size_C = (size_t)M * ldc;
    size_t size_D = (size_t)M * ldd;

    std::vector<__nv_bfloat16> h_A(size_A);
    std::vector<__nv_bfloat16> h_B(size_B);
    std::vector<__nv_bfloat16> h_C(size_C);
    std::vector<__nv_bfloat16> h_D_gpu(size_D); // For GPU result
    std::vector<__nv_bfloat16> h_D_cpu(size_D); // For CPU reference

    // --- Initialization ---
    std::cout << "Initializing matrices..." << std::endl;
    for (size_t i = 0; i < size_A; ++i) h_A[i] = __float2bfloat16(static_cast<float>(rand() % 10 - 5) / 5.0f); // Small range [-1, 0.8]
    for (size_t i = 0; i < size_B; ++i) h_B[i] = __float2bfloat16(static_cast<float>(rand() % 10 - 5) / 5.0f);
    for (size_t i = 0; i < size_C; ++i) h_C[i] = __float2bfloat16(static_cast<float>(rand() % 10 - 5) / 5.0f);

    // --- Device Allocation & Copy ---
    __nv_bfloat16 *d_A, *d_B, *d_C, *d_D;
    __nv_bfloat16 *d_alpha, *d_beta;

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_D, size_D * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_beta, sizeof(__nv_bfloat16)));

    std::cout << "Copying data to device..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), size_C * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha_bf16, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, &beta_bf16, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    GemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.ptr_A = d_A; params.lda = lda; params.stride_a = 0;
    params.ptr_B = d_B; params.ldb = ldb; params.stride_b = 0;
    params.ptr_C = d_C; params.ldc = ldc; params.stride_c = 0;
    params.ptr_D = d_D; params.ldd = ldd; params.stride_d = 0;
    params.ptr_alpha = d_alpha;
    params.ptr_beta = d_beta;

    // Grid dim calculation based on M, N tiles
    dim3 gridDim((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N, 1);
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    size_t sharedMemBytes = get_shared_memory_size();

    std::cout << "Launching kernel..." << std::endl;
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
    std::cout << "Shared Memory: " << sharedMemBytes << " bytes" << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    pure_cuda_gemm_bf16<<<gridDim, blockDim, sharedMemBytes>>>(params);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop)); // Wait for kernel completion

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    // --- Copy Result Back ---
    std::cout << "Copying result from device..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_D_gpu.data(), d_D, size_D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // --- CPU Calculation for Verification ---
    std::cout << "Calculating reference on CPU..." << std::endl;
    naive_matmul_bf16_host(M, N, K, alpha_f, beta_f,
                           h_A.data(), lda, h_B.data(), ldb, h_C.data(), ldc,
                           h_D_cpu.data(), ldd);

    // --- Compare Results ---
    std::cout << "Comparing GPU and CPU results..." << std::endl;
    bool results_match = compare_matrices_bf16(M, N, h_D_gpu.data(), h_D_cpu.data(), ldd);

    if (results_match) {
        std::cout << "SUCCESS: GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "FAILURE: GPU and CPU results do NOT match!" << std::endl;
    }

    // --- Cleanup ---
    std::cout << "Cleaning up..." << std::endl;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));

    std::cout << "Finished." << std::endl;

    return results_match ? 0 : 1;
}