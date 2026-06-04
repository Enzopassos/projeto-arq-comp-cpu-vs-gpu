#ifndef GPU_BENCHMARK_H
#define GPU_BENCHMARK_H

#include "../common.h"

void inicializarGPUDevice();

void runGPUBenchmark(int N, int total_runs, ResResultados &res,
                     const float *h_A, const float *h_B,
                     const float *h_C_ref, bool run_validation);

const char* obterModeloGPU();

#endif
