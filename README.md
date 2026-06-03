# Análise de Desempenho em Arquiteturas Paralelas: CPU vs. GPU

**Disciplina:** Arquitetura de Computadores  
**Autores:** Arthur Iwankiu Castro e Enzo da Silva Passos  
**Instituição:** Universidade Católica de Santos (UniSantos)  
**Repositório Oficial:** [github.com/Enzopassos/projeto-arq-comp-cpu-vs-gpu](https://github.com/Enzopassos/projeto-arq-comp-cpu-vs-gpu.git)

---

## 📌 Sobre o Projeto

Este projeto apresenta um estudo comparativo e experimental de desempenho entre a execução sequencial na CPU, o paralelismo multithreaded na CPU (OpenMP e AVX2 Manual) e a aceleração paralela massiva na GPU (NVIDIA CUDA C++) para a **multiplicação de matrizes de grande escala** ($N \times N$, variando de $100 \times 100$ até $10.000 \times 10.000$).

O foco principal do estudo é analisar como as diferenças microarquiteturais das plataformas determinam sua eficiência. Investigamos três tópicos centrais de Arquitetura de Computadores:
1. **O Gargalo do Barramento PCIe**: A latência de inicialização e transferência física de dados (*Host-to-Device* e *Device-to-Host*) comparada ao tempo computacional do kernel, demonstrando na prática a transição entre regimes de limitação de banda de memória (*Memory-Bound*) e limitação de poder de computação (*Compute-Bound*).
2. **Hierarquia de Memória na GPU**: O ganho de desempenho obtido ao mitigar acessos repetitivos à memória global (VRAM) por meio da técnica de **Shared Memory Tiling** (reúso de dados local em cache L1/SRAM interno).
3. **Vetorização SIMD na CPU**: O impacto das instruções AVX2 (256-bit) combinadas com `std::thread` em comparação ao paralelismo gerenciado pelo compilador via OpenMP.

---

## 💻 Ambientes de Testes Experimentais

A metodologia científica adotada baseia-se em execuções de testes em dois ambientes distintos, permitindo avaliar o impacto de diferentes gerações de microarquiteturas de hardware no processamento paralelo:

* **PC 1**: CPU AMD Ryzen 5 5600 (Zen 3, 6 Cores / 12 Threads) + GPU NVIDIA GeForce RTX 3080 (Ampere)
* **PC 2**: CPU AMD Ryzen 7 5700G (Zen 3, 8 Cores / 16 Threads) + GPU NVIDIA GeForce RTX 4060 (Ada Lovelace)

---

## 📂 Estrutura do Repositório

```text
├── common.h               # Struct ResResultados compartilhada entre CPU e GPU
├── cpu/                   # Módulo de CPU — Autor: Arthur Iwankiu Castro
│   ├── sequencial.h / .cpp    # Algoritmo sequencial i-k-j (cache-friendly)
│   ├── openmp.h / .cpp        # Algoritmo paralelo com OpenMP + SIMD
│   ├── manual.h / .cpp        # Algoritmo manual: AVX2 (256-bit) + std::thread
│   ├── cpu_benchmark.h        # Declaração do driver de benchmark de CPU
│   └── cpu_benchmark.cpp      # Implementação do benchmark de CPU (Seq, OMP, AVX2)
├── gpu/                   # Módulo de GPU — Autor: Enzo da Silva Passos
│   ├── kernels.h / .cu        # Kernels CUDA: Naive (VRAM global) e Tiled (Shared Memory)
│   ├── gpu_benchmark.h        # Declaração do driver de benchmark de GPU
│   ├── gpu_benchmark.cu       # Implementação do benchmark de GPU (CUDA Events + cooldown)
│   └── main.cu                # Orquestrador principal: CLI, hardware, CSV
├── plot_benchmarks.py     # Script Python para geração automática dos gráficos
├── benchmarks.csv         # Tabela de saída contendo as médias de todos os testes
└── README.md              # Documentação principal
```

---

## 🚀 Como Compilar e Executar

### 1. Compilação (Windows PowerShell)

Utilize o **NVCC** com suporte ao compilador MSVC do Visual Studio, habilitando OpenMP e instruções AVX2:

```powershell
cmd.exe /c "call `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" && nvcc -O3 -Xcompiler `"/openmp:experimental /arch:AVX2`" gpu/main.cu gpu/gpu_benchmark.cu gpu/kernels.cu cpu/sequencial.cpp cpu/openmp.cpp cpu/manual.cpp cpu/cpu_benchmark.cpp -o gpuxcpu.exe"
```

*Nota: ajuste o caminho do `vcvars64.bat` se sua versão do Visual Studio for diferente.*

### 2. Execução do Benchmark

O binário aceita argumentos de linha de comando para controle total da bateria de testes:

```powershell
# Benchmark completo (CPU + GPU, todos os tamanhos, 10 rodadas na GPU)
./gpuxcpu.exe

# Apenas GPU (ideal para coletar dados rápido)
./gpuxcpu.exe --mode gpu

# Apenas CPU (útil para o PC do Arthur sem GPU dedicada)
./gpuxcpu.exe --mode cpu

# Customizar tamanhos e número de rodadas
./gpuxcpu.exe --mode all --sizes 500,1000,2000,5000 --runs 5

# Incluir N=10000 na CPU sequencial (pode demorar ~5 min)
./gpuxcpu.exe --max-seq-size 10000

# Ver todas as opções
./gpuxcpu.exe --help
```

#### Opções disponíveis

| Opção | Descrição | Padrão |
|---|---|---|
| `--mode <all\|cpu\|gpu>` | Módulos a executar | `all` |
| `--sizes <n1,n2,...>` | Tamanhos $N$ das matrizes | `100,200,500,1000,2000,5000,10000` |
| `--runs <n>` | Repetições por teste (GPU e CPU pequenos) | `10` |
| `--max-seq-size <n>` | Limite de N para CPU Sequencial | `5000` |
| `--no-validation` | Desabilita validação GPU vs CPU | validação ativa |

> **Nota sobre tempo de execução:** O benchmark completo com os padrões foi projetado para terminar em **menos de 30 minutos**. A CPU Sequencial é automaticamente pulada para $N > 5000$ (configure com `--max-seq-size` se necessário). Rodadas de GPU para $N \ge 2000$ incluem um **cooldown térmico automático** entre execuções para garantir leituras estáveis sem throttling.

---

## 📊 Geração dos Gráficos Acadêmicos (Python)

O script `plot_benchmarks.py` consome o arquivo `benchmarks.csv` e gera três gráficos em alta resolução (300 DPI):

1. **`tempo_execucao.png`**: Curva de escalonamento de todos os paradigmas em escala logarítmica bidirecional.
2. **`speedup.png`**: Fator de aceleração de cada versão paralela em relação à CPU Sequencial base.
3. **`gargalo_pcie.png`**: Decomposição percentual do tempo da GPU Tiled entre computação do kernel e transferência PCIe.

### Como executar:
```powershell
pip install pandas matplotlib numpy
python plot_benchmarks.py
```

---

## 📈 Exemplo Prático de Resultados Coletados

Abaixo estão os resultados consolidados coletados experimentalmente na máquina de testes (PC 1: Ryzen 5 5600 + RTX 3080):

| Tamanho $N$ | CPU Sequencial | CPU OpenMP | CPU AVX2 | GPU Naive (Kernel / PCIe) | GPU Tiled (Kernel / PCIe) | Speedup Tiled vs Seq | Validação |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.48 ms | 0.45 ms | 0.93 ms | 0.28 ms (0.05 / 0.22) | 0.31 ms (0.10 / 0.22) | 1.5x | SUCESSO |
| **500** | 45 ms | 19 ms | 3.7 ms | 0.37 ms (0.17 / 0.20) | 0.33 ms (0.14 / 0.19) | 136x | SUCESSO |
| **1000** | 258 ms | 121 ms | 23 ms | 1.85 ms (1.28 / 0.56) | 1.57 ms (0.99 / 0.58) | 164x | SUCESSO |
| **2000** | 2109 ms | 846 ms | 110 ms | 11.3 ms (9.3 / 2.0) | 8.7 ms (6.8 / 1.9) | 242x | SUCESSO |
| **5000** | 39154 ms | 15624 ms | 4646 ms | 160.8 ms | 134.8 ms | 290x | SUCESSO |
| **10000** | N/A* | — | — | 2186 ms | 2543 ms** | — | SUCESSO |

*\* CPU Sequencial para N=10.000 ultrapassaria 5 minutos e é pulada por padrão.*  
*\*\* Resultado de coleta com throttling térmico detectado; coleta com cooldown habilitado corrige este valor.*

---

## 🔍 Conclusões e Análise Teórica (Arquitetura)

### 1. Curva de Transição do Gargalo PCIe
O benchmark valida experimentalmente o modelo teórico de **Intensidade Aritmética**. Com matrizes pequenas ($N = 100$), a cópia física via barramento PCIe consome a maior parte do tempo total da execução na GPU, devido à latência fixa de drivers e canais DMA. Com matrizes de grande escala, a complexidade de processamento $O(N^3)$ domina sobre a cópia $O(N^2)$, e o overhead do barramento torna-se insignificante — abrindo espaço para speedups massivos.

### 2. A Hierarquia de Memória (Shared Memory Tiling)
O kernel `multiplicaKernelTiled` demonstra o impacto de otimizar acessos à memória de vídeo. Ao realizar o carregamento cooperativo de dados em blocos compartilhados na SRAM local (equivalente ao cache L1 da GPU), a necessidade de leituras redundantes na VRAM global cai por um fator de 16x (com base no bloco de $16 \times 16$), resultando em redução direta no tempo de computação do kernel.

### 3. Vetorização SIMD (AVX2) vs OpenMP
A implementação manual com AVX2 demonstra ganhos significativos em relação ao OpenMP padrão, processando 8 floats por instrução com FMA (`_mm256_fmadd_ps`), evidenciando a importância da exploração explícita das unidades de execução vetorial disponíveis na arquitetura Zen 3.
