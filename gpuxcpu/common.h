// common.h
// Estrutura de dados compartilhada entre o benchmark de CPU e de GPU.
// Incluída por cpu_benchmark.h e gpu_benchmark.h sem depender de CUDA headers.
#ifndef COMMON_H
#define COMMON_H

struct ResResultados {
    int N;

    // Tempos de execução da CPU (em milissegundos)
    double t_cpu_seq;       // CPU Sequencial (i-k-j cache-friendly)
    double t_cpu_omp;       // CPU OpenMP (paralelo com diretivas pragma)
    double t_cpu_manual;    // CPU Manual (AVX2 SIMD + std::thread)

    // Tempos de execução da GPU Naive (em milissegundos)
    float t_gpu_naive_h2d;
    float t_gpu_naive_kernel;
    float t_gpu_naive_d2h;
    float t_gpu_naive_total;

    // Tempos de execução da GPU Tiled / Shared Memory (em milissegundos)
    float t_gpu_tiled_h2d;
    float t_gpu_tiled_kernel;
    float t_gpu_tiled_d2h;
    float t_gpu_tiled_total;

    // Resultado da validação matemática (GPU vs CPU)
    bool validacao_ok;
};

#endif // COMMON_H
