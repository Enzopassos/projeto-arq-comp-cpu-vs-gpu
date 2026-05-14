#include <iostream>

using namespace std;

// Cada "thread" da GPU vai calcular apenas UM elemento da matriz C.
__global__ void multiplicaKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Linha
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Coluna

    if (row < N && col < N) {
        float soma = 0.0f;
        for (int k = 0; k < N; k++) {
            soma += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = soma;
    }
}

int main() {
    int N = 1000;
    size_t bytes = N * N * sizeof(float);

    // Alocação na RAM (Host)
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Preenche matrizes
    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    // Alocação na VRAM (Device - Placa de vídeo)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Eventos CUDA para medir o tempo exigidos no PDF
    cudaEvent_t inicio, fim;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fim);

    // Começa a contar o tempo TOTAL (Transferência + Computação)
    cudaEventRecord(inicio);

    // Copia A e B da RAM para a VRAM (Barramento PCIe)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Configura a "malha" de Threads da GPU (blocos de 16x16)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Chama o Kernel na GPU
    multiplicaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copia C de volta da VRAM para a RAM
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Para de contar o tempo
    cudaEventRecord(fim);
    cudaEventSynchronize(fim);

    float tempo_ms = 0;
    cudaEventElapsedTime(&tempo_ms, inicio, fim);

    cout << "Tempo GPU (N=" << N << "): " << tempo_ms << " ms\n";

    // Limpeza
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}