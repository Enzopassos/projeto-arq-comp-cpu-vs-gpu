#include "kernels.h"

// Kernel Naive: Cada thread calcula um elemento da matriz C acessando a VRAM diretamente.
__global__ void multiplicaKernelNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float soma = 0.0f;
        for (int k = 0; k < N; k++) {
            soma += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = soma;
    }
}

// Kernel Tiled (Shared Memory): Reduz o tráfego com a VRAM reutilizando sub-matrizes (Tiles) em cache L1.
__global__ void multiplicaKernelTiled(float *A, float *B, float *C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float soma = 0.0f;

    // Loop sobre todos os tiles necessários para realizar a multiplicação
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < N) {
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Carrega dados da matriz B na Shared Memory s_B
        if (t * TILE_SIZE + ty < N && col < N) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Sincroniza para garantir que todo o tile foi carregado nas memórias compartilhadas
        __syncthreads();

        // Computa a multiplicação do bloco atual
        for (int k = 0; k < TILE_SIZE; ++k) {
            soma += s_A[ty][k] * s_B[k][tx];
        }

        // Sincroniza novamente antes de carregar o próximo tile
        __syncthreads();
    }

    // Grava o elemento final na matriz de resultados global
    if (row < N && col < N) {
        C[row * N + col] = soma;
    }
}
