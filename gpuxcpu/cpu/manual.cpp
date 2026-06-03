#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h> // A biblioteca mágica dos "Pacotes" SIMD
#include "manual.h"

// Nosso peão da fábrica, agora equipado com a Empilhadeira (AVX2)
void trabalhador(float *A, float *B, float *C, int N, int linha_inicio, int linha_fim) {
    
    // First Touch Policy (Limpar a mesa)
    for (int i = linha_inicio; i < linha_fim; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    // A esteira de pacotes
    for (int i = linha_inicio; i < linha_fim; i++) {
        for (int k = 0; k < N; k++) {
            
            float a_temp = A[i * N + k];
            
            // Cria um pacote repetindo o a_temp 8 vezes
            // [a_temp, a_temp, a_temp, a_temp, a_temp, a_temp, a_temp, a_temp]
            __m256 pacote_A = _mm256_set1_ps(a_temp);

            int j = 0;
            // O laço agora pula de 8 em 8 espaços da memória!
            for (; j <= N - 8; j += 8) {
                
                // Carrega um pacote de 8 floats da Matriz C e outro da Matriz B
                __m256 pacote_C = _mm256_loadu_ps(&C[i * N + j]);
                __m256 pacote_B = _mm256_loadu_ps(&B[k * N + j]);
                
                // FMA: pacote_C = (pacote_A * pacote_B) + pacote_C
                pacote_C = _mm256_fmadd_ps(pacote_A, pacote_B, pacote_C);
                
                // Salva o pacote processado de volta na memória RAM
                _mm256_storeu_ps(&C[i * N + j], pacote_C);
            }

            // O LIXEIRO: Resolve os números que sobraram no final da linha
            // (se N não for múltiplo exato de 8, ele faz a conta velha 1 por 1)
            for (; j < N; j++) {
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
        int linha_fim = (t == num_threads - 1) ? N : linha_inicio + linhas_por_thread;

        threads.push_back(std::thread(trabalhador, A, B, C, N, linha_inicio, linha_fim));
    }

    for (auto &th : threads) {
        th.join();
    }
}