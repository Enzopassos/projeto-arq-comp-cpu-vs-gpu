import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_file = "benchmarks.csv"
    if not os.path.exists(csv_file):
        print(f"[ERRO] O arquivo '{csv_file}' nao foi encontrado. Execute o benchmark primeiro!")
        return

    print(">>> Lendo dados de benchmarks.csv...")
    df = pd.read_csv(csv_file)
    
    df = df[df['Valido'] == 1].copy()

    # Prevenir divisao por zero ou valores invalidos
    df = df[df['CPU_Seq_ms'] > 0].copy()

    # Calcular Speedups
    df['Speedup_OMP'] = df['CPU_Seq_ms'] / df['CPU_OMP_ms']
    df['Speedup_GPU_Naive'] = df['CPU_Seq_ms'] / df['GPU_Naive_Total_ms']
    df['Speedup_GPU_Tiled'] = df['CPU_Seq_ms'] / df['GPU_Tiled_Total_ms']

    c_seq = "#2C3E50"   
    c_omp = "#3498DB"   
    c_naive = "#E67E22"  
    c_tiled = "#2ECC71"  
    c_pcie = "#E74C3C"   
    c_kernel = "#1ABC9C"

    # Configurar estilo dos graficos
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.edgecolor'] = '#BDC3C7'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#ECF0F1'
    plt.rcParams['grid.linewidth'] = 0.6

    print(">>> Gerando Grafico 1: Tempos de Execucao vs N...")
    plt.figure(figsize=(9, 6.5))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.plot(df['N'], df['CPU_Seq_ms'], marker='o', color=c_seq, linewidth=2.5, label='CPU Sequencial')
    plt.plot(df['N'], df['CPU_OMP_ms'], marker='s', color=c_omp, linewidth=2.5, label='CPU OpenMP')
    plt.plot(df['N'], df['GPU_Naive_Total_ms'], marker='^', color=c_naive, linewidth=2, linestyle='--', label='GPU Naive (Kernel + PCIe)')
    plt.plot(df['N'], df['GPU_Tiled_Total_ms'], marker='D', color=c_tiled, linewidth=2.5, label='GPU Tiled (Kernel + PCIe)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Tempo de Execucao (ms) - Escala Logaritmica')
    plt.title('Comparativo de Tempo de Execucao (CPU vs. GPU)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(df['N'], labels=[str(n) for n in df['N']])
    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7')
    plt.tight_layout()
    plt.savefig('tempo_execucao.png', dpi=300)
    plt.close()

    print(">>> Gerando Grafico 2: Speedup vs N...")
    plt.figure(figsize=(9, 6.5))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.plot(df['N'], df['Speedup_OMP'], marker='s', color=c_omp, linewidth=2.5, label='Speedup CPU OpenMP')
    plt.plot(df['N'], df['Speedup_GPU_Naive'], marker='^', color=c_naive, linewidth=2, linestyle='--', label='Speedup GPU Naive')
    plt.plot(df['N'], df['Speedup_GPU_Tiled'], marker='D', color=c_tiled, linewidth=2.5, label='Speedup GPU Tiled')

    # Linha de referencia de baseline
    plt.axhline(y=1.0, color='#7F8C8D', linestyle=':', linewidth=1.5, label='Linha de Base (CPU Seq)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Fator de Speedup (x vezes mais rapido) - Escala Logaritmica')
    plt.title('Speedup Relativo em Relacao a Execucao CPU Sequencial', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(df['N'], labels=[str(n) for n in df['N']])
    
    # Mostrar os valores exatos de Speedup final nos rotulos do grafico
    final_row = df.iloc[-1]
    plt.annotate(f"{final_row['Speedup_GPU_Tiled']:.1f}x", 
                 xy=(final_row['N'], final_row['Speedup_GPU_Tiled']), 
                 xytext=(-45, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color=c_tiled), fontsize=10, fontweight='bold', color=c_tiled)

    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7')
    plt.tight_layout()
    plt.savefig('speedup.png', dpi=300)
    plt.close()

    print(">>> Gerando Grafico 3: Decomposicao de Tempo da GPU (Gargalo PCIe)...")
    
    # Criar DataFrame focado em percentuais de tempo para GPU Tiled
    df['PCIe_Naive_ms'] = df['GPU_Naive_H2D_ms'] + df['GPU_Naive_D2H_ms']
    df['PCIe_Tiled_ms'] = df['GPU_Tiled_H2D_ms'] + df['GPU_Tiled_D2H_ms']

    # Percentuais do tempo total gasto com PCIe vs Computação pura
    df['Pct_PCIe_Naive'] = (df['PCIe_Naive_ms'] / df['GPU_Naive_Total_ms']) * 100
    df['Pct_Kernel_Naive'] = (df['GPU_Naive_Kernel_ms'] / df['GPU_Naive_Total_ms']) * 100
    df['Pct_PCIe_Tiled'] = (df['PCIe_Tiled_ms'] / df['GPU_Tiled_Total_ms']) * 100
    df['Pct_Kernel_Tiled'] = (df['GPU_Tiled_Kernel_ms'] / df['GPU_Tiled_Total_ms']) * 100

    x = np.arange(len(df['N']))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    bars_kernel = ax.bar(x, df['Pct_Kernel_Tiled'], width, label='Computacao do Kernel (GPU)', color=c_kernel)
    bars_pcie = ax.bar(x, df['Pct_PCIe_Tiled'], width, bottom=df['Pct_Kernel_Tiled'], label='Transferencia PCIe (H2D + D2H)', color=c_pcie)

    ax.set_ylabel('Distribuicao Percentual do Tempo de Execucao (%)')
    ax.set_xlabel('Tamanho da Matriz (N x N)')
    ax.set_title('Impacto do Gargalo do PCIe na GPU Tiled vs. Tamanho N', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in df['N']])
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc='lower right', facecolor='white', edgecolor='#BDC3C7')

    # Adicionar rotulos de porcentagem dentro das barras
    for idx, (pct_kernel, pct_pcie) in enumerate(zip(df['Pct_Kernel_Tiled'], df['Pct_PCIe_Tiled'])):
        if pct_kernel > 5:
            ax.text(idx, pct_kernel / 2, f"{pct_kernel:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if pct_pcie > 5:
            ax.text(idx, pct_kernel + (pct_pcie / 2), f"{pct_pcie:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('gargalo_pcie.png', dpi=300)
    plt.close()

    print(">>> Todos os graficos academicos foram gerados com sucesso!")
    print("    - Arquivos criados:")
    print("      * tempo_execucao.png (Comparativo geral de tempo)")
    print("      * speedup.png (Aceleracao relativa vs. C++ Sequencial)")
    print("      * gargalo_pcie.png (Analise do gargalo de barramento PCIe)")
    print("=========================================================================")

if __name__ == "__main__":
    main()
