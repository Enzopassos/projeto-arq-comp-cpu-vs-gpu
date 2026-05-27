# Análise de Desempenho em Arquiteturas Paralelas: CPU vs. GPU

**Disciplina:** Arquitetura de Computadores  
**Autores:** Arthur Iwankiu Castro e Enzo da Silva Passos  
**Instituição:** Universidade Católica de Santos (UniSantos)  
**Repositório Oficial:** [github.com/Enzopassos/projeto-arq-comp-cpu-vs-gpu](https://github.com/Enzopassos/projeto-arq-comp-cpu-vs-gpu.git)

---

## 📌 Sobre o Projeto

Este projeto apresenta um estudo comparativo e experimental de desempenho entre a execução sequencial na CPU, o paralelismo multithreaded na CPU (OpenMP) e a aceleração paralela massiva na GPU (NVIDIA CUDA C++) para a **multiplicação de matrizes de grande escala** ($N \times N$, variando de $100 \times 100$ até $10.000 \times 10.000$).

O foco principal do estudo é analisar como as diferenças microarquiteturais das plataformas determinam sua eficiência. Investigamos dois tópicos centrais de Arquitetura de Computadores:
1. **O Gargalo do Barramento PCIe**: A latência de inicialização e transferência física de dados (*Host-to-Device* e *Device-to-Host*) comparada ao tempo computacional do kernel, demonstrando na prática a transição entre regimes de limitação de banda de memória (*Memory-Bound*) e limitação de poder de computação (*Compute-Bound*).
2. **Hierarquia de Memória na GPU**: O ganho de desempenho obtido ao mitigar acessos repetitivos à memória global (VRAM) por meio da técnica de **Shared Memory Tiling** (reúso de dados local em cache L1/SRAM interno).

---

## 💻 Ambientes de Testes Experimentais

A metodologia científica adotada baseia-se em execuções de testes em dois ambientes distintos, permitindo avaliar o impacto de diferentes gerações de microarquiteturas de hardware no processamento paralelo:

* **PC 1**: CPU AMD Ryzen 5 5600 (Zen 3, 6 Cores / 12 Threads) + GPU NVIDIA GeForce RTX 3080 (Ampere)
* **PC 2**: CPU AMD Ryzen 7 5700G (Zen 3, 8 Cores / 16 Threads) + GPU NVIDIA GeForce RTX 4060 (Ada Lovelace)

---

## 📂 Estrutura do Repositório

```text
├── cpu/
│   ├── sequencial.h       # Declaração do algoritmo sequencial
│   ├── sequencial.cpp     # Implementação i-k-j otimizada para cache da CPU
│   ├── openmp.h           # Declaração do algoritmo paralelo em CPU
│   └── openmp.cpp         # Implementação paralela com diretivas OpenMP e SIMD
├── gpu/
│   ├── kernels.h          # Cabeçalho dos Kernels CUDA (Naive & Tiled)
│   ├── kernels.cu         # Implementação física dos Kernels CUDA
│   └── main.cu            # Driver de benchmark unificado (10 rodadas e estatísticas)
├── plot_benchmarks.py     # Script em Python para geração automática dos gráficos
├── benchmarks.csv         # Tabela de saída contendo a coleta de todas as rodadas
└── README.md              # Documentação principal
```

---

## 🚀 Como Compilar e Executar

O driver de benchmark unificado em `gpu/main.cu` gerencia a inicialização determinística das matrizes, a execução de **10 rodadas completas de computação para cada tamanho $N$ e paradigma**, a validação de tolerância a precisão de ponto flutuante (`1e-2f`) e a exportação das médias de tempo em CSV.

### 1. Compilação (Windows PowerShell)
Para compilar todas as fontes modularizadas simultaneamente com suporte ao CUDA (NVCC) e ao OpenMP experimental do MSVC (Visual Studio), execute o seguinte comando no PowerShell:

```powershell
cmd.exe /c "call `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" && nvcc -O3 -Xcompiler /openmp:experimental gpu/main.cu gpu/kernels.cu cpu/sequencial.cpp cpu/openmp.cpp -o gpuxcpu.exe"
```

*Nota: Se o caminho do Visual Studio for diferente no seu sistema, certifique-se de ajustar o diretório correspondente ao arquivo `vcvars64.bat`.*

### 2. Execução do Benchmark
Inicie a bateria de benchmarks executando o binário compilado:
```powershell
./gpuxcpu.exe
```

O console exibirá um contador de progresso detalhado rodada a rodada (essencial para acompanhar a execução cúbica da CPU Sequencial nos tamanhos de $N \ge 5000$). No fim do benchmark, a tabela de resultados consolidados será impressa e os dados serão salvos em `benchmarks.csv`.

---

## 📊 Geração dos Gráficos Acadêmicos (Python)

Criamos o script `plot_benchmarks.py` para automatizar a renderização de gráficos em alta resolução científica (300 DPI, prontos para inclusão em artigos). O script consome o arquivo `benchmarks.csv` gerado e plota:

1. **`tempo_execucao.png`**: Curva de escalonamento dos tempos (CPU Seq, CPU OMP, GPU Naive e GPU Tiled) em escala Logarítmica bidirecional.
2. **`speedup.png`**: Gráfico demonstrando o ganho de aceleração das versões paralelas e aceleradas em relação à CPU Sequencial base.
3. **`gargalo_pcie.png`**: Gráfico de barras empilhadas que decompõe a distribuição percentual do tempo total da GPU Tiled entre **Transferência de dados via PCIe (H2D + D2H)** e **Computação Pura do Kernel**.

### Como Rodar a Plotagem:
Instale as dependências padrão do ecossistema científico e execute o script:
```powershell
pip install pandas matplotlib numpy
python plot_benchmarks.py
```

---

## 📈 Exemplo Prático de Resultados Coletados

Abaixo estão os resultados consolidados coletados experimentalmente na máquina de testes:

| Tamanho $N$ | CPU Sequencial | CPU OpenMP | GPU Naive (Kernel / PCIe) | GPU Tiled (Kernel / PCIe) | Speedup GPU Tiled vs CPU OMP | Validação |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.24 ms | 0.22 ms | 0.10 ms (0.01 / 0.08) ms | 0.09 ms (0.02 / 0.08) ms | **2.44x** | SUCESSO |
| **200** | 2.02 ms | 1.05 ms | 0.23 ms (0.06 / 0.18) ms | 0.13 ms (0.03 / 0.11) ms | **8.07x** | SUCESSO |
| **500** | 35.53 ms | 15.03 ms | 0.56 ms (0.20 / 0.36) ms | 0.50 ms (0.17 / 0.33) ms | **30.06x** | SUCESSO |
| **1000** | 257.13 ms | 145.43 ms | 2.26 ms (1.21 / 1.05) ms | 1.95 ms (0.99 / 0.96) ms | **74.57x** | SUCESSO |
| **2000** | 2033.04 ms | 959.30 ms | 16.78 ms (10.91 / 5.87) ms | 12.17 ms (6.77 / 5.41) ms | **78.82x** | SUCESSO |
| **5000** | 15.61 s | 15.60 s | 155.68 ms (124.40 / 31.28) ms | 124.31 ms (96.62 / 27.69) ms | **125.55x** | SUCESSO |
| **10000** | 177.57 s | 177.57 s | 1178.65 ms (1074.41 / 104.24) ms | 884.81 ms (772.36 / 112.45) ms | **200.68x** | SUCESSO |

---

## 🔍 Conclusões e Análise Teórica (Arquitetura)

### 1. Curva de Transição do Gargalo PCIe
O benchmark valida experimentalmente o modelo teórico de **Intensidade Aritmética**. Com matrizes pequenas ($N = 100$), a cópia física via barramento PCIe consome cerca de **88.8%** de todo o tempo da execução na GPU, devido à latência fixa associada a drivers e preparação de canais DMA. Com matrizes de grande escala ($N = 10.000$), a complexidade de processamento $O(N^3)$ domina amplamente sobre a cópia $O(N^2)$, e o overhead do barramento cai para insignificantes **12.7%**, abrindo espaço para um speedup massivo de **200.7x** sobre a execução paralelizada na CPU.

### 2. A Hierarquia de Memória (Shared Memory Tiling)
O kernel `multiplicaKernelTiled` provou o enorme impacto de otimizar acessos à memória de vídeo. Ao realizar o carregamento cooperativo de dados em blocos compartilhados na SRAM local (rápida como cache L1 da GPU), a necessidade de leituras redundantes na VRAM global cai por um fator de 16x (com base no bloco de $16 \times 16$). O resultado foi uma **redução direta de 28.1% no tempo de computação de kernel** para $N=10.000$ (de 1074.41 ms para 772.36 ms).
