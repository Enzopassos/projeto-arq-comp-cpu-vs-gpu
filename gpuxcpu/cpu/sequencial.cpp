// sequencial.cpp
#include "sequencial.h"

void multiplicaSequencial(float *A, float *B, float *C, int N) {
    // Inicializa a matriz com zeros
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_temp * B[k * N + j];
            }
        }
    }
}