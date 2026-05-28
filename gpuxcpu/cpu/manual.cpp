// manual.cpp
#include <iostream>
#include <vector>
#include <thread>
#include "manual.h"

void trabalhador(float *A, float *B, float *C, int N, int linha_inicio, int linha_fim) {
    
    // First Touch Policy: O próprio trabalhador zera sua parte da matriz!
    // Isso garante que a memória seja alocada no núcleo físico correto.
    for (int i = linha_inicio; i < linha_fim; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    // A nossa esteira otimizada (i-k-j)
    for (int i = linha_inicio; i < linha_fim; i++) {
        for (int k = 0; k < N; k++) {
            float a_temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_temp * B[k * N + j];
            }
        }
    }
}

void multiplicaManual(float *A, float *B, float *C, int N, int num_threads) {
    std::vector<std::thread> threads;
    int linhas_por_thread = N / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int linha_inicio = t * linhas_por_thread;
        // O último trabalhador pega qualquer linha que tenha sobrado
        int linha_fim = (t == num_threads - 1) ? N : linha_inicio + linhas_por_thread;

        threads.push_back(std::thread(trabalhador, A, B, C, N, linha_inicio, linha_fim));
    }

    // Espera todos terminarem
    for (auto &th : threads) {
        th.join();
    }
}