// main.cpp
#include <iostream>
#include <chrono>
#include "sequencial.h"
#include "openmp.h"

using namespace std;

int main() {
    int N = 1000;

    // Alocação de memória no Heap (RAM)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    //hora de inicializar as matrizes:
for (int i = 0; i < N * N; i++) {
    A[i] = (float)rand() / RAND_MAX;
    B[i] = (float)rand() / RAND_MAX;
    C[i] = 0.0f;
}

    // Menu de escolha
    cout << "=========================================\n";
    cout << "   MULTIPLICACAO DE MATRIZES (N=" << N << ")\n";
    cout << "=========================================\n";
    cout << "1 - Executar Versao Sequencial\n";
    cout << "2 - Executar Versao OpenMP\n";
    cout << "Escolha uma opcao: ";
    
    int opcao;
    cin >> opcao;
    cout << "-----------------------------------------\n";

    if (opcao == 1) {
        // ---- EXECUÇÃO SEQUENCIAL ----
        auto inicio = chrono::high_resolution_clock::now();
        
        multiplicaSequencial(A, B, C, N);
        
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> tempo = fim - inicio;
        cout << "Executado: SEQUENCIAL\n";
        cout << "Tempo de execucao: " << tempo.count() << " ms\n";
    } 
    else if (opcao == 2) {
        // ---- EXECUÇÃO OPENMP ----
        auto inicio = chrono::high_resolution_clock::now();
        
        multiplicaOpenMP(A, B, C, N);
        
        auto fim = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> tempo = fim - inicio;
        cout << "Executado: OPENMP\n";
        cout << "Tempo de execucao: " << tempo.count() << " ms\n";
    } 
    else {
        cout << "Opcao invalida! Encerrando o programa.\n";
    }

    cout << "=========================================\n";

    // Libera a memória
    delete[] A; 
    delete[] B; 
    delete[] C;
    
    return 0;
}