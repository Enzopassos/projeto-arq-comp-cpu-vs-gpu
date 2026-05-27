#ifndef KERNELS_H
#define KERNELS_H

#define TILE_SIZE 16

// Kernel Naive: Cada thread calcula um elemento da matriz C acessando a VRAM diretamente.
__global__ void multiplicaKernelNaive(float *A, float *B, float *C, int N);

// Kernel Tiled (Shared Memory): Reduz o tráfego com a VRAM reutilizando sub-matrizes (Tiles) em cache L1.
__global__ void multiplicaKernelTiled(float *A, float *B, float *C, int N);

#endif
