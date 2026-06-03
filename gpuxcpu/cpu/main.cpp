// main.cpp
#include <iostream>
#include <chrono>
#include <thread>
#include "sequencial.h"
#include "openmp.h"
#include "manual.h"

using namespace std;

int main() {
    int N = 1000;

    // Alocação de memória no Heap
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Inicializa A e B
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    cout << "=========================================\n";
    cout << "   MULTIPLICACAO DE MATRIZES (N=" << N << ")\n";
    cout << "=========================================\n";
    cout << "1 - Executar Versao Sequencial\n";
    cout << "2 - Executar Versao OpenMP\n";
    cout << "3 - Executar Versao Threads Manuais\n";
    cout << "Escolha uma opcao: ";
    
    int opcao;
    cin >> opcao;
    cout << "-----------------------------------------\n";

    if (opcao == 1) {
        // ---- EXECUÇÃO SEQUENCIAL ----
        cout << "[Info] Iniciando Warm-up (Aquecimento de Cache)...\n";
        for (int i = 0; i < N * N; i++) C[i] = 0.0f; // Zera para o Warm-up
        multiplicaSequencial(A, B, C, N); 
        
        cout << "[Info] Iniciando medicao oficial...\n";
        for (int i = 0; i < N * N; i++) C[i] = 0.0f; // Zera para a medição oficial

        auto inicio = chrono::high_resolution_clock::now();
        multiplicaSequencial(A, B, C, N);
        auto fim = chrono::high_resolution_clock::now();
        
        chrono::duration<double, std::milli> tempo = fim - inicio;
        cout << "Executado: SEQUENCIAL\nTempo: " << tempo.count() << " ms\n";
    } 
    else if (opcao == 2) {
        // ---- EXECUÇÃO OPENMP ----
        // O OpenMP já zera a matriz C dentro dele (First Touch Policy)
        cout << "[Info] Iniciando Warm-up (Aquecimento de Cache)...\n";
        multiplicaOpenMP(A, B, C, N);
        
        cout << "[Info] Iniciando medicao oficial...\n";
        auto inicio = chrono::high_resolution_clock::now();
        multiplicaOpenMP(A, B, C, N);
        auto fim = chrono::high_resolution_clock::now();
        
        chrono::duration<double, std::milli> tempo = fim - inicio;
        cout << "Executado: OPENMP\nTempo: " << tempo.count() << " ms\n";
    } 
    else if (opcao == 3) {
        // ---- EXECUÇÃO THREADS MANUAIS ----
        int num_threads = std::thread::hardware_concurrency(); 
        if (num_threads == 0) num_threads = 8;

        // As threads já zeram a matriz C dentro delas (First Touch Policy)
        cout << "[Info] Iniciando Warm-up (Aquecimento de Cache)...\n";
        multiplicaManual(A, B, C, N, num_threads);
        
        cout << "[Info] Iniciando medicao oficial...\n";
        auto inicio = chrono::high_resolution_clock::now();
        multiplicaManual(A, B, C, N, num_threads);
        auto fim = chrono::high_resolution_clock::now();
        
        chrono::duration<double, std::milli> tempo = fim - inicio;
        cout << "Executado: THREADS MANUAIS (" << num_threads << " Threads)\nTempo: " << tempo.count() << " ms\n";
    } 
    else {
        cout << "Opcao invalida!\n";
    }

    cout << "=========================================\n";

    delete[] A; delete[] B; delete[] C;
    return 0;
}