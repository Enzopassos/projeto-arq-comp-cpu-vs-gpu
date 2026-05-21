// openmp.cpp
#include <omp.h>
#include "openmp.h"

// Versão otimizada (i-k-j) aproveitando localidade de cache e vetorização
void multiplicaOpenMP(float *A, float *B, float *C, int N) {
    // Inicializa a matriz C com zeros de forma paralela
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    // Multiplicação otimizada com a ordem i-k-j
    #pragma omp parallel for 
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_temp = A[i * N + k]; 
            
            #pragma omp simd
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_temp * B[k * N + j];
            }
        }
    }
}