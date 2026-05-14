#include <iostream>
#include <chrono>

using namespace std;

void multiplicaSequencial(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;
            for (int k = 0; k < N; k++) {
                soma += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = soma;
        }
    }
}

int main() {
    int N = 1000;
    size_t bytes = N * N * sizeof(float);

    // Alocação de memória no Heap (RAM)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // Marca o tempo de início
    auto inicio = chrono::high_resolution_clock::now();

    multiplicaSequencial(A, B, C, N);

    // Marca o tempo de fim
    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> tempo = fim - inicio;

    cout << "Tempo Sequencial (N=" << N << "): " << tempo.count() << " ms\n";

    // Libera a memória
    delete[] A; delete[] B; delete[] C;
    return 0;
}