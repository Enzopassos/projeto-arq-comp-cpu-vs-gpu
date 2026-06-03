// cpu_benchmark.h
// Autor: Arthur Iwankiu Castro
// Declara a função de benchmark de CPU que executa e mede o desempenho das
// três implementações de multiplicação de matrizes na CPU:
//   - Sequencial (i-k-j cache-friendly)
//   - Paralela com OpenMP
//   - Manual com AVX2 SIMD + std::thread
#ifndef CPU_BENCHMARK_H
#define CPU_BENCHMARK_H

#include "../common.h"

// Executa a bateria completa de benchmarks da CPU para um dado tamanho N.
//
// Parâmetros:
//   N           - Dimensão da matriz N×N
//   runs        - Número de repetições para cálculo da média
//   max_n_seq   - Tamanho máximo de N para executar a versão sequencial
//                 (pulado se N > max_n_seq, pois pode demorar horas)
//   res         - Struct de resultados que será preenchida com os tempos medidos
//   h_A, h_B    - Ponteiros para as matrizes de entrada já inicializadas
//   h_C         - Buffer de saída para a multiplicação (será zerado internamente)
void runCPUBenchmark(int N, int runs,
                     ResResultados &res,
                     float *h_A, float *h_B, float *h_C);

#endif // CPU_BENCHMARK_H
