#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <intrin.h>
#include <cuda_runtime.h>

#include "../common.h"
#include "gpu_benchmark.h"
#include "../cpu/cpu_benchmark.h"
#include "../cpu/openmp.h"

using namespace std;

static string obterModeloCPU() {
    int info[4] = {};
    char brand[49] = {};
    __cpuid(info, 0x80000002); memcpy(brand,      info, 16);
    __cpuid(info, 0x80000003); memcpy(brand + 16, info, 16);
    __cpuid(info, 0x80000004); memcpy(brand + 32, info, 16);
    string s(brand);
    size_t p = s.find_first_not_of(' ');
    return (p != string::npos) ? s.substr(p) : s;
}

static string sanitizarNome(string s) {
    string r = "";
    for (char c : s) {
        if (isalnum((unsigned char)c)) {
            r += c;
        } else if (c == ' ' || c == '_' || c == '-') {
            if (!r.empty() && r.back() != '_') {
                r += '_';
            }
        }
    }
    while (!r.empty() && r.back() == '_') {
        r.pop_back();
    }
    return r;
}

// Inicialização das matrizes com valores determinísticos
static void inicializarMatrizes(float *A, float *B, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f + 0.5f;
        B[i] = (float)((i * 3) % 100) / 100.0f + 0.5f;
    }
}

static int rodadasCPU(int N, int default_runs) {
    return default_runs; 
}

int main(int argc, char *argv[]) {
    string mode        = "all";
    int    total_runs  = 10;
    bool   validation  = true;
    string custom_output = "";
    vector<int> sizes  = {100, 200, 500, 1000, 2000, 5000, 10000};
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if      (a == "--mode"         && i+1 < argc) mode       = argv[++i];
        else if (a == "--runs"         && i+1 < argc) total_runs = stoi(argv[++i]);
        else if (a == "--no-validation")              validation  = false;
        else if (a == "--out"          && i+1 < argc) custom_output = argv[++i];
        else if (a == "--sizes"        && i+1 < argc) {
            string tok = argv[++i];
            sizes.clear();
            size_t pos;
            while ((pos = tok.find(',')) != string::npos) {
                sizes.push_back(stoi(tok.substr(0, pos)));
                tok.erase(0, pos + 1);
            }
            sizes.push_back(stoi(tok));
        }
        else if (a == "--help" || a == "-h") {
            cout
                << "Uso: " << argv[0] << " [opcoes]\n\n"
                << "  --mode <all|cpu|gpu>        Modo de execucao (padrao: all)\n"
                << "  --sizes <n1,n2,...>          Tamanhos N das matrizes (padrao: 100,200,500,1000,2000,5000,10000)\n"
                << "  --runs <n>                   Repeticoes por teste (padrao: 10, reduzido automaticamente para N grande)\n"
                << "  --no-validation              Desabilita validacao matematica GPU vs CPU\n"
                << "  --out <sufixo>               Sufixo personalizado para o arquivo CSV\n"
                << "  -h, --help                   Mostra esta ajuda\n\n"
                << "Exemplos:\n"
                << "  " << argv[0] << "                           # Benchmark completo\n"
                << "  " << argv[0] << " --mode gpu               # Apenas GPU\n"
                << "  " << argv[0] << " --mode cpu --sizes 1000,2000,5000\n";
            return 0;
        }
    }

    // Inicializa CUDA e lê nome da GPU
    inicializarGPUDevice();

    string cpu_name = obterModeloCPU();
    string gpu_name = string(obterModeloGPU());

    cout << "=========================================================================\n"
         << "          BENCHMARK DE ARQUITETURA DE COMPUTADORES: CPU vs GPU\n"
         << "=========================================================================\n"
         << "Hardware Detectado:\n"
         << "  - CPU : " << cpu_name << "\n"
         << "  - GPU : " << gpu_name << "\n"
         << "=========================================================================\n"
         << "Configuracoes:\n"
         << "  - Modo            : " << mode       << "\n"
         << "  - Repeticoes GPU  : " << total_runs << "\n"
         << "  - Validacao       : " << (validation ? "Ativa" : "Inativa") << "\n"
         << "=========================================================================\n\n";

    vector<ResResultados> resultados;

    for (int N : sizes) {
        size_t bytes = (size_t)N * N * sizeof(float);
        cout << ">>> INICIANDO TESTES PARA N = " << N
             << "  (" << (double)bytes / (1024.0 * 1024.0) << " MB por matriz)\n";

        // Aloca matrizes de entrada como pinned memory (otimiza transferências PCIe)
        float *h_A = nullptr, *h_B = nullptr;
        cudaMallocHost(&h_A, bytes);
        cudaMallocHost(&h_B, bytes);

        // Matriz de saída para a CPU (não precisa ser pinned)
        float *h_C = new float[N * N];
        memset(h_C, 0, bytes);

        inicializarMatrizes(h_A, h_B, N);

        // Struct de resultados inicializada com zeros
        ResResultados res{};
        res.N           = N;
        res.validacao_ok = true;

        // Benchmark de CPU
        if (mode == "all" || mode == "cpu") {
            int runs_cpu = rodadasCPU(N, total_runs);
            runCPUBenchmark(N, runs_cpu, res, h_A, h_B, h_C);
        }

        // Benchmark de GPU
        if (mode == "all" || mode == "gpu") {
            if (validation && mode == "gpu") {
                cout << "  [Validacao]      Gerando referencia CPU (OpenMP)... " << flush;
                memset(h_C, 0, bytes);
                multiplicaOpenMP(h_A, h_B, h_C, N);
                cout << "Pronto.\n";
            }
            runGPUBenchmark(N, total_runs, res, h_A, h_B,
                            (validation ? h_C : nullptr), validation);
        }

        resultados.push_back(res);

        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        delete[] h_C;

        cout << "-------------------------------------------------------------------------\n\n";
    }

    // TABELA RESUMO NO CONSOLE
    cout << "\n"
         << "=======================================================================================================================================\n"
         << "                                              TABELA RESUMO DO BENCHMARK (TEMPOS EM MS)\n"
         << "=======================================================================================================================================\n"
         << "   N   | CPU Seq    | CPU OMP    | CPU AVX2   | GPU Naive (Kernel / PCIe)     | GPU Tiled (Kernel / PCIe)     | Valido\n"
         << "-------+------------+------------+------------+-------------------------------+-------------------------------+--------\n";

    for (auto const &r : resultados) {
        auto fmt_cpu = [](double v) -> string {
            if (v <= 0.0) return "  N/A     ";
            char buf[32];
            if (v >= 1000.0)
                snprintf(buf, sizeof(buf), "%7.0f ms", v);
            else
                snprintf(buf, sizeof(buf), "%7.2f ms", v);
            return buf;
        };

        cout << setw(6) << r.N << " | "
             << fmt_cpu(r.t_cpu_seq)    << " | "
             << fmt_cpu(r.t_cpu_omp)    << " | "
             << fmt_cpu(r.t_cpu_manual) << " | "
             << fixed << setprecision(2)
             << setw(8) << r.t_gpu_naive_total << " ms ("
             << setw(7) << r.t_gpu_naive_kernel << " / "
             << setw(5) << (r.t_gpu_naive_h2d + r.t_gpu_naive_d2h) << ") | "
             << setw(8) << r.t_gpu_tiled_total << " ms ("
             << setw(7) << r.t_gpu_tiled_kernel << " / "
             << setw(5) << (r.t_gpu_tiled_h2d + r.t_gpu_tiled_d2h) << ") | "
             << (r.validacao_ok ? "SUCESSO" : "FALHA") << "\n";
    }
    cout << "=======================================================================================================================================\n"
         << "Legenda: GPU = Total (Kernel puro / PCIe H2D+D2H)  |  N/A = teste pulado\n\n";

    //EXPORTAÇÃO CSV
    string csv_file = "benchmarks.csv";
    if (!custom_output.empty()) {
        csv_file = "benchmarks_" + custom_output + ".csv";
    } else {
        string s_cpu = sanitizarNome(cpu_name);
        string s_gpu = sanitizarNome(gpu_name);
        csv_file = "benchmarks_" + s_cpu + "_" + s_gpu + ".csv";
    }
    ofstream csv(csv_file);
    csv << "N,CPU_Seq_ms,CPU_OMP_ms,CPU_Manual_ms,"
        << "GPU_Naive_H2D_ms,GPU_Naive_Kernel_ms,GPU_Naive_D2H_ms,GPU_Naive_Total_ms,"
        << "GPU_Tiled_H2D_ms,GPU_Tiled_Kernel_ms,GPU_Tiled_D2H_ms,GPU_Tiled_Total_ms,"
        << "Valido\n";
    for (auto const &r : resultados) {
        csv << r.N              << ","
            << r.t_cpu_seq      << ","
            << r.t_cpu_omp      << ","
            << r.t_cpu_manual   << ","
            << r.t_gpu_naive_h2d    << ","
            << r.t_gpu_naive_kernel << ","
            << r.t_gpu_naive_d2h    << ","
            << r.t_gpu_naive_total  << ","
            << r.t_gpu_tiled_h2d    << ","
            << r.t_gpu_tiled_kernel << ","
            << r.t_gpu_tiled_d2h    << ","
            << r.t_gpu_tiled_total  << ","
            << (r.validacao_ok ? "1" : "0") << "\n";
    }
    csv.close();
    cout << ">>> CSV exportado para '" << csv_file << "' com sucesso!\n"
         << "=========================================================================\n";

    return 0;
}
