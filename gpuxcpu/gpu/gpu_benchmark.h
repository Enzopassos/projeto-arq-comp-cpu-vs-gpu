#ifndef GPU_BENCHMARK_H
#define GPU_BENCHMARK_H

#include "../common.h"

// Força a inicialização do contexto CUDA (deve ser chamada uma vez antes dos benchmarks)
void inicializarGPUDevice();

// Executa a bateria completa de benchmarks da GPU para um dado tamanho N.
//
// Parâmetros:
//   N              - Dimensão da matriz N×N
//   total_runs     - Número de repetições para cálculo da média
//   res            - Struct de resultados que será preenchida com os tempos medidos
//   h_A, h_B       - Ponteiros para as matrizes de entrada (devem ser pinned memory)
//   h_C_ref        - Matriz de referência da CPU para validação (nullptr = sem validação)
//   run_validation - Se true, valida o resultado da GPU contra h_C_ref
void runGPUBenchmark(int N, int total_runs, ResResultados &res,
                     const float *h_A, const float *h_B,
                     const float *h_C_ref, bool run_validation);

// Retorna o modelo comercial da GPU detectada (usa cudaGetDeviceProperties)
const char* obterModeloGPU();

#endif // GPU_BENCHMARK_H
