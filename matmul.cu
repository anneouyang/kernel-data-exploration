#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matmulKernel(float* A, float* B, float* C, int N) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < N / BLOCK_SIZE; ++i) {
        Asub[threadIdx.y][threadIdx.x] = A[row * N + (i * BLOCK_SIZE + threadIdx.x)];
        Bsub[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * N + col];
        
        __syncthreads();
        
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

void matmulHost(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(N / BLOCK_SIZE, N / BLOCK_SIZE);
    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 256; // Matrix size (N x N)
    size_t size = N * N * sizeof(float);
    
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);
    
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    
    matmulHost(A, B, C, N);
    
    printf("C[0,0] = %f\n", C[0]);
    
    free(A);
    free(B);
    free(C);
    return 0;
}