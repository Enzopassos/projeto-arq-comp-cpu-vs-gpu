#include <omp.h>

void multiplicaOpenMP(float *A, float *B, float *C, int N) {
    // Essa linha diz ao compilador para paralelizar os dois primeiros 'for'
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;
            for (int k = 0; k < N; k++) {
                soma += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = soma;
        }
    }
}