// gpu_benchmark.cu
// Autor: Enzo da Silva Passos
// Implementa a bateria de benchmarks da GPU usando CUDA, medindo separadamente
// o tempo de: transferência H2D (Host→Device), execução do Kernel e D2H (Device→Host).
// Dois kernels são testados: Naive (acesso direto à VRAM) e Tiled (Shared Memory).
#include "gpu_benchmark.h"
#include "kernels.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#define SLEEP_MS(ms) Sleep(ms)
#else
#include <unistd.h>
#define SLEEP_MS(ms) usleep((ms) * 1000)
#endif

using namespace std;

static char s_gpu_name[256] = "GPU NVIDIA Compativel";

const char* obterModeloGPU() {
    return s_gpu_name;
}

// Valida resultado da GPU contra referência da CPU com tolerância para float32
static bool validarMatrizes(const float *ref, const float *teste, int N) {
    if (ref == nullptr) return true;
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(ref[i] - teste[i]);
        if (diff > max_diff) max_diff = diff;
    }
    const float tolerancia = 1e-2f;
    if (max_diff > tolerancia) {
        cout << "\n  [ERRO DE VALIDACAO] Diferenca maxima: " << max_diff << "\n";
        return false;
    }
    return true;
}

// Determina o intervalo de cooldown entre rodadas para evitar throttling térmico
static int cooldownMs(int N) {
    if (N >= 10000) return 3000; // 3 segundos entre rodadas para N=10000
    if (N >= 5000)  return 1500; // 1.5 segundo para N=5000
    if (N >= 2000)  return 500;  // 0.5 segundo para N=2000
    return 0;
}

void inicializarGPUDevice() {
    cudaFree(0);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        strncpy(s_gpu_name, prop.name, sizeof(s_gpu_name) - 1);
    }
}

void runGPUBenchmark(int N, int total_runs, ResResultados &res,
                     const float *h_A, const float *h_B,
                     const float *h_C_ref, bool run_validation) {
    size_t bytes = N * N * sizeof(float);

    // Alocação de memória no Device
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    float *h_C_gpu = nullptr;
    cudaMallocHost(&h_C_gpu, bytes);

    // Configuração de Grid/Block
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    // Warm-up
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    multiplicaKernelNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Eventos CUDA para medição de tempo
    cudaEvent_t ev_h2d_s, ev_h2d_e;
    cudaEvent_t ev_ker_s, ev_ker_e;
    cudaEvent_t ev_d2h_s, ev_d2h_e;
    cudaEventCreate(&ev_h2d_s); cudaEventCreate(&ev_h2d_e);
    cudaEventCreate(&ev_ker_s); cudaEventCreate(&ev_ker_e);
    cudaEventCreate(&ev_d2h_s); cudaEventCreate(&ev_d2h_e);

    int cd_ms = cooldownMs(N);

    // BENCHMARK GPU KERNEL NAIVE
    float sum_naive_h2d = 0.0f, sum_naive_ker = 0.0f, sum_naive_d2h = 0.0f;
    cout << "  [GPU Naive]      Rodando " << total_runs << " execucoes";
    if (cd_ms > 0) cout << " (cooldown " << cd_ms << "ms entre rodadas)";
    cout << ":\n" << flush;

    for (int r = 0; r < total_runs; r++) {
        cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;

        if (r > 0 && cd_ms > 0) SLEEP_MS(cd_ms);

        // H2D
        cudaEventRecord(ev_h2d_s);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(ev_h2d_e);

        // Kernel
        cudaEventRecord(ev_ker_s);
        multiplicaKernelNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(ev_ker_e);

        // D2H
        cudaEventRecord(ev_d2h_s);
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h_e);

        cudaEventSynchronize(ev_d2h_e);

        float t_h2d = 0, t_ker = 0, t_d2h = 0;
        cudaEventElapsedTime(&t_h2d, ev_h2d_s, ev_h2d_e);
        cudaEventElapsedTime(&t_ker,  ev_ker_s,  ev_ker_e);
        cudaEventElapsedTime(&t_d2h, ev_d2h_s, ev_d2h_e);

        sum_naive_h2d += t_h2d;
        sum_naive_ker  += t_ker;
        sum_naive_d2h += t_d2h;
        cout << fixed << setprecision(2) << (t_h2d + t_ker + t_d2h) << " ms\n";
    }

    sum_naive_h2d /= total_runs;
    sum_naive_ker  /= total_runs;
    sum_naive_d2h /= total_runs;
    float naive_total = sum_naive_h2d + sum_naive_ker + sum_naive_d2h;

    cout << "  [GPU Naive]      Media Final: " << naive_total << " ms"
         << " (Kernel: " << sum_naive_ker
         << " ms, H2D: " << sum_naive_h2d
         << " ms, D2H: " << sum_naive_d2h << " ms)\n\n";

    // Validação do Naive
    bool naive_ok = true;
    if (run_validation) naive_ok = validarMatrizes(h_C_ref, h_C_gpu, N);

    // BENCHMARK GPU KERNEL TILED
    float sum_tiled_h2d = 0.0f, sum_tiled_ker = 0.0f, sum_tiled_d2h = 0.0f;
    cout << "  [GPU Tiled]      Rodando " << total_runs << " execucoes";
    if (cd_ms > 0) cout << " (cooldown " << cd_ms << "ms entre rodadas)";
    cout << ":\n" << flush;

    for (int r = 0; r < total_runs; r++) {
        cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;

        if (r > 0 && cd_ms > 0) SLEEP_MS(cd_ms);

        // H2D
        cudaEventRecord(ev_h2d_s);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(ev_h2d_e);

        // Kernel
        cudaEventRecord(ev_ker_s);
        multiplicaKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(ev_ker_e);

        // D2H
        cudaEventRecord(ev_d2h_s);
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h_e);

        cudaEventSynchronize(ev_d2h_e);

        float t_h2d = 0, t_ker = 0, t_d2h = 0;
        cudaEventElapsedTime(&t_h2d, ev_h2d_s, ev_h2d_e);
        cudaEventElapsedTime(&t_ker,  ev_ker_s,  ev_ker_e);
        cudaEventElapsedTime(&t_d2h, ev_d2h_s, ev_d2h_e);

        sum_tiled_h2d += t_h2d;
        sum_tiled_ker  += t_ker;
        sum_tiled_d2h += t_d2h;
        cout << fixed << setprecision(2) << (t_h2d + t_ker + t_d2h) << " ms\n";
    }

    sum_tiled_h2d /= total_runs;
    sum_tiled_ker  /= total_runs;
    sum_tiled_d2h /= total_runs;
    float tiled_total = sum_tiled_h2d + sum_tiled_ker + sum_tiled_d2h;

    cout << "  [GPU Tiled]      Media Final: " << tiled_total << " ms"
         << " (Kernel: " << sum_tiled_ker
         << " ms, H2D: " << sum_tiled_h2d
         << " ms, D2H: " << sum_tiled_d2h << " ms)\n\n";

    // Validação do Tiled
    bool tiled_ok = true;
    if (run_validation) tiled_ok = validarMatrizes(h_C_ref, h_C_gpu, N);

    // Resultado de Validação
    res.validacao_ok = naive_ok && tiled_ok;
    if (run_validation && h_C_ref != nullptr) {
        if (res.validacao_ok)
            cout << "  [Validacao GPU]  PASSOU com sucesso contra a CPU!\n";
        else
            cout << "  [Validacao GPU]  FALHOU! Os dados da GPU divergem da CPU.\n";
    }

    // Preenchimento dos resultados
    res.t_gpu_naive_h2d    = sum_naive_h2d;
    res.t_gpu_naive_kernel = sum_naive_ker;
    res.t_gpu_naive_d2h    = sum_naive_d2h;
    res.t_gpu_naive_total  = naive_total;

    res.t_gpu_tiled_h2d    = sum_tiled_h2d;
    res.t_gpu_tiled_kernel = sum_tiled_ker;
    res.t_gpu_tiled_d2h    = sum_tiled_d2h;
    res.t_gpu_tiled_total  = tiled_total;

    cudaEventDestroy(ev_h2d_s); cudaEventDestroy(ev_h2d_e);
    cudaEventDestroy(ev_ker_s);  cudaEventDestroy(ev_ker_e);
    cudaEventDestroy(ev_d2h_s); cudaEventDestroy(ev_d2h_e);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_C_gpu);
}
