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

| | **PC 1** | **PC 2** |
|---|---|---|
| **CPU** | AMD Ryzen 5 5600 (Zen 3, 6C/12T, 32 MB L3) | AMD Ryzen 7 5700G (Zen 3, 8C/16T, 16 MB L3) |
| **GPU** | NVIDIA GeForce RTX 3080 (Ampere, 8704 CUDA Cores) | NVIDIA GeForce RTX 4060 (Ada Lovelace, 3072 CUDA Cores) |
| **Barramento PCIe** | PCIe 4.0 x16 (~31.5 GB/s) | PCIe 3.0 x8 (~7.88 GB/s) ⚠️ |

> ⚠️ O Ryzen 7 5700G é uma APU e limita o barramento a PCIe 3.0. Combinado com a RTX 4060 (que opera em x8 fisicamente), a banda efetiva do PC 2 é ~4x menor que a do PC 1 — um dos pontos centrais de análise deste estudo.

---

## 📂 Estrutura do Repositório

```text
├── common.h                    # Struct ResResultados compartilhada entre CPU e GPU
├── cpu/                        # Módulo de CPU — Autor: Arthur Iwankiu Castro
│   ├── sequencial.h / .cpp         # Algoritmo sequencial i-k-j (cache-friendly)
│   ├── openmp.h / .cpp             # Algoritmo paralelo com OpenMP + SIMD
│   ├── manual.h / .cpp             # Algoritmo manual: AVX2 (256-bit) + std::thread
│   ├── cpu_benchmark.h             # Declaração do driver de benchmark de CPU
│   └── cpu_benchmark.cpp           # Implementação do benchmark de CPU (Seq, OMP, AVX2)
├── gpu/                        # Módulo de GPU — Autor: Enzo da Silva Passos
│   ├── kernels.h / .cu             # Kernels CUDA: Naive (VRAM global) e Tiled (Shared Memory)
│   ├── gpu_benchmark.h             # Declaração do driver de benchmark de GPU
│   ├── gpu_benchmark.cu            # Benchmark de GPU (CUDA Events + Warm-up + cooldown térmico)
│   └── main.cu                     # Orquestrador: CLI, detecção de hardware, exportação CSV
├── plot_benchmarks.py          # Script Python: gráficos individuais ou comparativos inter-PC
├── benchmarks_<CPU>_<GPU>.csv  # CSV gerado automaticamente com o nome do hardware detectado
└── README.md                   # Documentação principal
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
# Benchmark completo (CPU + GPU, todos os tamanhos, 10 rodadas)
./gpuxcpu.exe

# Apenas GPU
./gpuxcpu.exe --mode gpu

# Apenas CPU
./gpuxcpu.exe --mode cpu

# Customizar tamanhos e número de rodadas
./gpuxcpu.exe --sizes 500,1000,2000,5000 --runs 5

# Rodar apenas 1 vez (coleta rápida)
./gpuxcpu.exe --runs 1

# Definir um sufixo personalizado para o arquivo CSV de saída
./gpuxcpu.exe --out PC1

# Ver todas as opções
./gpuxcpu.exe --help
```

#### Opções disponíveis

| Opção | Descrição | Padrão |
|---|---|---|
| `--mode <all\|cpu\|gpu>` | Módulos a executar | `all` |
| `--sizes <n1,n2,...>` | Tamanhos $N$ das matrizes | `100,200,500,1000,2000,5000,10000` |
| `--runs <n>` | Repetições por teste | `10` |
| `--out <sufixo>` | Sufixo personalizado para o arquivo CSV de saída | (nome do hardware detectado) |
| `--no-validation` | Desabilita validação matemática GPU vs CPU | validação ativa |

> **Nota sobre tempo de execução:** O benchmark completo com os padrões pode levar até **40 minutos** dependendo da máquina (N=10.000 na CPU sequencial é o gargalo principal). Rodadas de GPU para $N \ge 2000$ incluem um **cooldown térmico automático** entre execuções para garantir leituras estáveis.

### 3. Nomeação Automática dos Arquivos CSV

Ao terminar, o binário detecta o hardware em uso e salva os resultados em um arquivo com nome único, por exemplo:

```
benchmarks_AMD_Ryzen_5_5600_6_Core_Processor_NVIDIA_GeForce_RTX_3080.csv
benchmarks_AMD_Ryzen_7_5700G_with_Radeon_Graphics_NVIDIA_GeForce_RTX_4060.csv
```

Isso permite coletar dados de diferentes máquinas sem sobrescrever arquivos.

---

## 📊 Geração dos Gráficos (Python)

O script `plot_benchmarks.py` detecta automaticamente quantos arquivos `benchmarks_*.csv` existem na pasta:

- **1 arquivo:** gera gráficos individuais da máquina em alta resolução (300 DPI).
- **2+ arquivos:** gera automaticamente **gráficos comparativos inter-máquinas**.

### Gráficos individuais (1 PC)
1. **`tempo_execucao.png`** — Curva de escalonamento de todos os paradigmas em escala log.
2. **`speedup.png`** — Fator de aceleração de cada versão paralela vs. CPU Sequencial.
3. **`gargalo_pcie.png`** — Decomposição percentual do tempo GPU entre kernel e PCIe.

### Gráficos comparativos (2 PCs)
1. **`comparativo_tempo_cpu.png`** — Tempos de CPU (Seq, OMP, AVX2) de cada máquina.
2. **`comparativo_tempo_gpu.png`** — Tempos de GPU (Naive, Tiled) de cada placa.
3. **`comparativo_gargalo_pcie.png`** — Tempo de transferência PCIe e largura de banda efetiva vs. limites teóricos.
4. **`comparativo_speedup.png`** — Fator de Speedup da GPU Tiled vs. CPU Sequencial de cada máquina.

### Como executar:
```powershell
pip install pandas matplotlib numpy
python plot_benchmarks.py
```

---

## 📈 Resultados Coletados (PC 1: Ryzen 5 5600 + RTX 3080)

Abaixo estão os resultados reais coletados experimentalmente no PC 1 (10 execuções por tamanho):

| Tamanho $N$ | CPU Sequencial | CPU OpenMP | CPU AVX2 | GPU Naive (Total) | GPU Tiled (Total) | Speedup Tiled vs Seq | Validação |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 1.69 ms | 0.96 ms | 1.11 ms | 0.45 ms | 0.39 ms | 4.3x | ✅ |
| **200** | 6.52 ms | 0.76 ms | 1.02 ms | 0.42 ms | 0.32 ms | 20.4x | ✅ |
| **500** | 69.4 ms | 15.6 ms | 4.20 ms | 0.41 ms | 0.50 ms | 138.9x | ✅ |
| **1000** | 335.7 ms | 102.3 ms | 18.1 ms | 2.31 ms | 2.06 ms | 162.9x | ✅ |
| **2000** | 2273.8 ms | 950.3 ms | 135.2 ms | 13.9 ms | 11.0 ms | 206.7x | ✅ |
| **5000** | 37887.8 ms | 12979.3 ms | 5581.5 ms | 174.4 ms | 139.7 ms | 271.2x | ✅ |
| **10000** | 284616 ms | 104319 ms | 85334 ms | 1407.8 ms | 1043.3 ms | 272.8x | ✅ |

---

## 🔍 Conclusões e Análise Teórica (Arquitetura)

### 1. Curva de Transição do Gargalo PCIe
O benchmark valida experimentalmente o modelo teórico de **Intensidade Aritmética**. Com matrizes pequenas ($N = 100$), a cópia física via barramento PCIe consome a maior parte do tempo total da GPU, devido à latência fixa de inicialização de drivers e canais DMA. Com matrizes de grande escala, a complexidade de processamento $O(N^3)$ domina sobre a cópia $O(N^2)$, e o overhead do barramento torna-se insignificante — abrindo espaço para speedups de centenas de vezes.

### 2. Impacto do Barramento PCIe na Comparação entre PCs
A diferença de barramento entre os dois ambientes de teste é um dos achados mais relevantes: o PC 1 opera em **PCIe 4.0 x16 (~31.5 GB/s)** enquanto o PC 2 é limitado pelo processador APU a **PCIe 3.0 x8 (~7.88 GB/s)**, uma diferença teórica de ~4x. Os gráficos comparativos comprovam esse gargalo físico de forma empírica.

### 3. A Hierarquia de Memória (Shared Memory Tiling)
O kernel `multiplicaKernelTiled` demonstra o impacto de otimizar acessos à VRAM. Ao carregar cooperativamente dados em blocos de 16×16 na SRAM interna (Shared Memory), a necessidade de leituras redundantes na memória global cai por um fator de até 16x, com redução direta no tempo de kernel.

### 4. Cache L3 vs. Núcleos (Ryzen 5 5600 vs. Ryzen 7 5700G)
O Ryzen 5 5600 supera o Ryzen 7 5700G na execução sequencial pura graças ao dobro de Cache L3 (32 MB vs. 16 MB). Porém, sob paralelismo massivo com OpenMP, o Ryzen 7 5700G vence com seus 8 núcleos físicos contra 6, evidenciando como diferentes dimensões arquiteturais se tornam dominantes em diferentes regimes de execução.

### 5. Vetorização SIMD (AVX2) vs OpenMP
A implementação manual com AVX2 demonstra ganhos significativos em relação ao OpenMP padrão, processando 8 floats por instrução com FMA (`_mm256_fmadd_ps`), evidenciando a importância da exploração explícita da ISA do processador Zen 3.
