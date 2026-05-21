// sequencial.cpp
#include <iostream>
#include "sequencial.h"

void multiplicaSequencial(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        // Imprime o progresso a cada linha processada
        if (i % 10 == 0 || i == N - 1) { // % 10 evita imprimir toda hora e perder performance
            std::cout << "\r[Sequencial] Processando: " << (i * 100) / N << "%..." << std::flush;
        }

        for (int k = 0; k < N; k++) {
            float a_temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_temp * B[k * N + j];
            }
        }
    }
    std::cout << "\r[Sequencial] Concluido!                      \n"; // Limpa a linha no final
}