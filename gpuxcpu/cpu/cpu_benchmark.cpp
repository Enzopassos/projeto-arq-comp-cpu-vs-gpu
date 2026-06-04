// cpu_benchmark.cpp
// Autor: Arthur Iwankiu Castro
// Implementa a bateria de benchmarks de CPU para multiplicação de matrizes N×N,
// medindo o tempo de execução das versões Sequencial, OpenMP e AVX2+Threads.
#include "cpu_benchmark.h"
#include "sequencial.h"
#include "openmp.h"
#include "manual.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <thread>

using namespace std;

void runCPUBenchmark(int N, int runs,
                     ResResultados &res,
                     float *h_A, float *h_B, float *h_C) {
    size_t bytes = N * N * sizeof(float);

    // ==========================================
    // 1. CPU SEQUENCIAL
    // ==========================================
    {
        double t_total = 0.0;
        cout << "  [CPU Sequencial] Rodando " << runs << " execucao(oes):\n" << flush;
        for (int r = 0; r < runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << runs << "... " << flush;

            // Zera a matriz de saída antes de cada rodada para evitar acumulação
            memset(h_C, 0, bytes);

            auto t0 = chrono::high_resolution_clock::now();
            multiplicaSequencial(h_A, h_B, h_C, N);
            auto t1 = chrono::high_resolution_clock::now();

            double dur_ms = chrono::duration<double, milli>(t1 - t0).count();
            t_total += dur_ms;
            cout << fixed << setprecision(2) << dur_ms << " ms\n";
        }
        res.t_cpu_seq = t_total / runs;
        cout << "  [CPU Sequencial] Media Final: "
             << fixed << setprecision(2) << res.t_cpu_seq << " ms\n\n";
    }

    // ==========================================
    // 2. CPU OPENMP
    // ==========================================
    {
        double t_total = 0.0;
        cout << "  [CPU OpenMP]     Rodando " << runs << " execucoes:\n" << flush;
        for (int r = 0; r < runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << runs << "... " << flush;

            // OpenMP inicializa C internamente, mas zeramos por consistência
            memset(h_C, 0, bytes);

            auto t0 = chrono::high_resolution_clock::now();
            multiplicaOpenMP(h_A, h_B, h_C, N);
            auto t1 = chrono::high_resolution_clock::now();

            double dur_ms = chrono::duration<double, milli>(t1 - t0).count();
            t_total += dur_ms;
            cout << fixed << setprecision(2) << dur_ms << " ms\n";
        }
        res.t_cpu_omp = t_total / runs;
        cout << "  [CPU OpenMP]     Media Final: "
             << fixed << setprecision(2) << res.t_cpu_omp << " ms\n\n";
    }

    // ==========================================
    // 3. CPU MANUAL (AVX2 + std::thread)
    // ==========================================
    {
        double t_total = 0.0;
        int num_threads = (int)std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 8;

        cout << "  [CPU AVX2+Thrs]  Rodando " << runs << " execucoes"
             << " (" << num_threads << " threads):\n" << flush;
        for (int r = 0; r < runs; r++) {
            cout << "    -> Execucao " << (r + 1) << "/" << runs << "... " << flush;

            // Manual também zera internamente via First-Touch, mas garantimos aqui
            memset(h_C, 0, bytes);

            auto t0 = chrono::high_resolution_clock::now();
            multiplicaManual(h_A, h_B, h_C, N, num_threads);
            auto t1 = chrono::high_resolution_clock::now();

            double dur_ms = chrono::duration<double, milli>(t1 - t0).count();
            t_total += dur_ms;
            cout << fixed << setprecision(2) << dur_ms << " ms\n";
        }
        res.t_cpu_manual = t_total / runs;
        cout << "  [CPU AVX2+Thrs]  Media Final: "
             << fixed << setprecision(2) << res.t_cpu_manual << " ms\n\n";
    }
}
