#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "kernels.h"
#include "../cpu/sequencial.h"
#include "../cpu/openmp.h"

using namespace std;

struct ResResultados {
    int N;
    double t_cpu_seq;
    double t_cpu_omp;
    float t_gpu_naive_h2d;
    float t_gpu_naive_kernel;
    float t_gpu_naive_d2h;
    float t_gpu_naive_total;
    float t_gpu_tiled_h2d;
    float t_gpu_tiled_kernel;
    float t_gpu_tiled_d2h;
    float t_gpu_tiled_total;
    bool validacao_ok;
};

// Validação dos dados
bool validarMatrizes(float *ref, float *teste, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(ref[i] - teste[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    float tolerancial_float = 1e-2f; // Tolerância segura para operações de grande escala em float
    if (max_diff > tolerancial_float) {
        cout << "\n[ERRO DE VALIDACAO] Diferenca maxima de " << max_diff << " detectada!\n";
        return false;
    }
    return true;
}

// Inicializa matriz com floats determinísticos
void inicializarMatrizes(float *A, float *B, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f + 0.5f;
        B[i] = (float)((i * 3) % 100) / 100.0f + 0.5f;
    }
}

// DRIVER DE BENCHMARK
int main() {
    // Semente para geração
    srand(42);

    // Listagem de tamanhos N definidos
    vector<int> tamanhos = {100, 200, 500, 1000, 2000, 5000, 10000};
    vector<ResResultados> resultados;

    cout << "=========================================================================\n";
    cout << "          BENCHMARK DE ARQUITETURA DE COMPUTADORES: CPU vs GPU\n";
    cout << "=========================================================================\n";
    cout << "Hardware Alvo:\n";
    cout << "  - CPU: AMD Zen 3 (Sequencial & OpenMP Multithreaded)\n";
    cout << "  - GPU: NVIDIA Ampere/Ada Lovelace (CUDA Naive & Tiled Shared Memory)\n";
    cout << "=========================================================================\n";
    cout << "Metodologia Cientifica:\n";
    cout << "  - Rigor estatistico total: Exatamente 10 execucoes por teste.\n";
    cout << "  - Calculo de Media Aritmetica pura de todos os tempos.\n";
    cout << "=========================================================================\n\n";

    for (int N : tamanhos) {
        size_t bytes = N * N * sizeof(float);
        cout << ">>> INICIANDO TESTES PARA TAMANHO N = " << N << " (" << (double)bytes / (1024 * 1024) << " MB por matriz)\n";

        const int total_runs = 10;

        // Alocação Host
        float *h_A = new float[N * N];
        float *h_B = new float[N * N];
        float *h_C_cpu = new float[N * N];
        float *h_C_gpu = new float[N * N];

        inicializarMatrizes(h_A, h_B, N);

        // 1. BENCHMARK CPU SEQUENCIAL
        double t_seq_total = 0.0;
        cout << "  [CPU Sequencial] Rodando " << total_runs << " execucoes:\n" << flush;
        for (int r = 0; r < total_runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;
            auto start = chrono::high_resolution_clock::now();
            multiplicaSequencial(h_A, h_B, h_C_cpu, N);
            auto end = chrono::high_resolution_clock::now();
            double dur_ms = chrono::duration<double, std::milli>(end - start).count();
            t_seq_total += dur_ms;
            cout << fixed << setprecision(2) << dur_ms << " ms\n";
        }
        t_seq_total /= total_runs;
        cout << "  [CPU Sequencial] Media Final: " << fixed << setprecision(2) << t_seq_total << " ms\n\n";

        // 2. BENCHMARK CPU OPENMP
        double t_omp_total = 0.0;
        cout << "  [CPU OpenMP]     Rodando " << total_runs << " execucoes:\n" << flush;
        for (int r = 0; r < total_runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;
            auto start = chrono::high_resolution_clock::now();
            multiplicaOpenMP(h_A, h_B, h_C_cpu, N);
            auto end = chrono::high_resolution_clock::now();
            double dur_ms = chrono::duration<double, std::milli>(end - start).count();
            t_omp_total += dur_ms;
            cout << fixed << setprecision(2) << dur_ms << " ms\n";
        }
        t_omp_total /= total_runs;
        cout << "  [CPU OpenMP]     Media Final: " << fixed << setprecision(2) << t_omp_total << " ms\n\n";

        // Alocação Device
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // Configuração da Grid/Block para a GPU
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        // AQUECIMENTO DA GPU
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        multiplicaKernelNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        // Eventos CUDA para Medições
        cudaEvent_t start_h2d, stop_h2d;
        cudaEvent_t start_kernel, stop_kernel;
        cudaEvent_t start_d2h, stop_d2h;
        cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
        cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
        cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);

        // 3. BENCHMARK GPU CUDA NAIVE
        float sum_naive_h2d = 0.0f, sum_naive_kernel = 0.0f, sum_naive_d2h = 0.0f;
        cout << "  [GPU Naive]      Rodando " << total_runs << " execucoes:\n" << flush;
        for (int r = 0; r < total_runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;
            
            // H2D
            cudaEventRecord(start_h2d);
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(stop_h2d);

            // Kernel
            cudaEventRecord(start_kernel);
            multiplicaKernelNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop_kernel);

            // D2H
            cudaEventRecord(start_d2h);
            cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_d2h);

            cudaEventSynchronize(stop_d2h);

            float t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;
            cudaEventElapsedTime(&t_h2d, start_h2d, stop_h2d);
            cudaEventElapsedTime(&t_kernel, start_kernel, stop_kernel);
            cudaEventElapsedTime(&t_d2h, start_d2h, stop_d2h);

            sum_naive_h2d += t_h2d;
            sum_naive_kernel += t_kernel;
            sum_naive_d2h += t_d2h;
            cout << fixed << setprecision(2) << (t_h2d + t_kernel + t_d2h) << " ms\n";
        }
        sum_naive_h2d /= total_runs;
        sum_naive_kernel /= total_runs;
        sum_naive_d2h /= total_runs;
        float total_naive_time = sum_naive_h2d + sum_naive_kernel + sum_naive_d2h;
        cout << "  [GPU Naive]      Media Final: " << total_naive_time << " ms (Kernel: " << sum_naive_kernel << " ms, H2D: " << sum_naive_h2d << " ms, D2H: " << sum_naive_d2h << " ms)\n\n";

        // Validação da GPU Naive contra CPU
        bool naive_valid = validarMatrizes(h_C_cpu, h_C_gpu, N);

        // 4. BENCHMARK GPU CUDA TILED (OTIMIZADO)
        float sum_tiled_h2d = 0.0f, sum_tiled_kernel = 0.0f, sum_tiled_d2h = 0.0f;
        cout << "  [GPU Tiled]      Rodando " << total_runs << " execucoes:\n" << flush;
        for (int r = 0; r < total_runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << total_runs << "... " << flush;
            
            // H2D
            cudaEventRecord(start_h2d);
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(stop_h2d);

            // Kernel
            cudaEventRecord(start_kernel);
            multiplicaKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop_kernel);

            // D2H
            cudaEventRecord(start_d2h);
            cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_d2h);

            cudaEventSynchronize(stop_d2h);

            float t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;
            cudaEventElapsedTime(&t_h2d, start_h2d, stop_h2d);
            cudaEventElapsedTime(&t_kernel, start_kernel, stop_kernel);
            cudaEventElapsedTime(&t_d2h, start_d2h, stop_d2h);

            sum_tiled_h2d += t_h2d;
            sum_tiled_kernel += t_kernel;
            sum_tiled_d2h += t_d2h;
            cout << fixed << setprecision(2) << (t_h2d + t_kernel + t_d2h) << " ms\n";
        }
        sum_tiled_h2d /= total_runs;
        sum_tiled_kernel /= total_runs;
        sum_tiled_d2h /= total_runs;
        float total_tiled_time = sum_tiled_h2d + sum_tiled_kernel + sum_tiled_d2h;
        cout << "  [GPU Tiled]      Media Final: " << total_tiled_time << " ms (Kernel: " << sum_tiled_kernel << " ms, H2D: " << sum_tiled_h2d << " ms, D2H: " << sum_tiled_d2h << " ms)\n\n";

        // Validação da GPU Tiled contra CPU
        bool tiled_valid = validarMatrizes(h_C_cpu, h_C_gpu, N);

        bool validacao_geral = naive_valid && tiled_valid;
        if (validacao_geral) {
            cout << "  [Validacao]      PASSOU com sucesso!\n";
        } else {
            cout << "  [Validacao]      FALHOU! Os dados estao incorretos.\n";
        }

        // Armazenamento
        ResResultados res = {
            N,
            t_seq_total,
            t_omp_total,
            sum_naive_h2d,
            sum_naive_kernel,
            sum_naive_d2h,
            total_naive_time,
            sum_tiled_h2d,
            sum_tiled_kernel,
            sum_tiled_d2h,
            total_tiled_time,
            validacao_geral
        };
        resultados.push_back(res);

        // Desalocação
        cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
        
        cout << "-------------------------------------------------------------------------\n\n";
    }

    // APRESENTAÇÃO DE DADOS
    cout << "=========================================================================================================\n";
    cout << "                                  TABELA RESUMO DO BENCHMARK (TEMPOS EM MS)\n";
    cout << "=========================================================================================================\n";
    cout << "   N   | CPU Seq  | CPU OMP  | GPU Naive (Kernel/PCIe)      | GPU Tiled (Kernel/PCIe)      | Validacao\n";
    cout << "-------+----------+----------+------------------------------+------------------------------+-------------\n";
    for (auto const &r : resultados) {
        cout << setw(6) << r.N << " | "
             << setw(6) << (int)r.t_cpu_seq << " ms | "
             << setw(6) << (int)r.t_cpu_omp << " ms | "
             << setw(6) << fixed << setprecision(1) << r.t_gpu_naive_total << " ms ("
             << setprecision(1) << r.t_gpu_naive_kernel << "/" << (r.t_gpu_naive_h2d + r.t_gpu_naive_d2h) << ") | "
             << setw(6) << fixed << setprecision(1) << r.t_gpu_tiled_total << " ms ("
             << setprecision(1) << r.t_gpu_tiled_kernel << "/" << (r.t_gpu_tiled_h2d + r.t_gpu_tiled_d2h) << ") | "
             << (r.validacao_ok ? "SUCESSO" : "FALHA") << "\n";
    }
    cout << "=========================================================================================================\n";
    cout << "Legenda: Tempos indicados como Total (Kernel / PCIe Transferências H2D+D2H)\n\n";

    // EXPORTAÇÃO CSV
    string csv_filename = "benchmarks.csv";
    ofstream csv(csv_filename);
    csv << "N,CPU_Seq_ms,CPU_OMP_ms,GPU_Naive_H2D_ms,GPU_Naive_Kernel_ms,GPU_Naive_D2H_ms,GPU_Naive_Total_ms,GPU_Tiled_H2D_ms,GPU_Tiled_Kernel_ms,GPU_Tiled_D2H_ms,GPU_Tiled_Total_ms,Valido\n";
    for (auto const &r : resultados) {
        csv << r.N << ","
            << r.t_cpu_seq << ","
            << r.t_cpu_omp << ","
            << r.t_gpu_naive_h2d << ","
            << r.t_gpu_naive_kernel << ","
            << r.t_gpu_naive_d2h << ","
            << r.t_gpu_naive_total << ","
            << r.t_gpu_tiled_h2d << ","
            << r.t_gpu_tiled_kernel << ","
            << r.t_gpu_tiled_d2h << ","
            << r.t_gpu_tiled_total << ","
            << (r.validacao_ok ? "1" : "0") << "\n";
    }
    csv.close();
    cout << ">>> Resultados do benchmark exportados com sucesso para o arquivo '" << csv_filename << "'!\n";
    cout << "=========================================================================\n";

    return 0;
}
