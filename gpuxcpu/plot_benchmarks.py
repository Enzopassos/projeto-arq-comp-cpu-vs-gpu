import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_label(filepath):
    base = os.path.basename(filepath)
    if base == "benchmarks.csv":
        return "PC Local"
    if base.startswith("benchmarks_") and base.endswith(".csv"):
        label = base[len("benchmarks_"):-4]
        label = label.replace("AMD_Ryzen_", "Ryzen ")
        label = label.replace("_Processor", "")
        label = label.replace("_with_Radeon_Graphics", "")
        label = label.replace("NVIDIA_GeForce_", "")
        label = label.replace("_6_Core", "")
        label = label.replace("_8_Core", "")
        label = label.replace("_12_Core", "")
        label = label.replace("_", " ")
        return label
    return base

def plot_single(filepath):
    print(f">>> Lendo dados de {filepath}...")
    df = pd.read_csv(filepath)
    df = df[df['Valido'] == 1].copy()
    df = df[df['CPU_Seq_ms'] > 0].copy()

    # Calcular Speedups
    df['Speedup_OMP'] = df['CPU_Seq_ms'] / df['CPU_OMP_ms']
    if 'CPU_Manual_ms' in df.columns:
        df['Speedup_Manual'] = df['CPU_Seq_ms'] / df['CPU_Manual_ms']
    df['Speedup_GPU_Naive'] = df['CPU_Seq_ms'] / df['GPU_Naive_Total_ms']
    df['Speedup_GPU_Tiled'] = df['CPU_Seq_ms'] / df['GPU_Tiled_Total_ms']

    c_seq = "#2C3E50"   
    c_omp = "#3498DB"   
    c_manual = "#9B59B6"
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

    label = get_label(filepath)
    title_suffix = f" ({label})" if label != "PC Local" else ""

    print(">>> Gerando Grafico 1: Tempos de Execucao vs N...")
    plt.figure(figsize=(9, 6.5))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.plot(df['N'], df['CPU_Seq_ms'], marker='o', color=c_seq, linewidth=2.5, label='CPU Sequencial')
    plt.plot(df['N'], df['CPU_OMP_ms'], marker='s', color=c_omp, linewidth=2.5, label='CPU OpenMP')
    if 'CPU_Manual_ms' in df.columns:
        plt.plot(df['N'], df['CPU_Manual_ms'], marker='p', color=c_manual, linewidth=2.5, label='CPU Manual (AVX2)')
    plt.plot(df['N'], df['GPU_Naive_Total_ms'], marker='^', color=c_naive, linewidth=2, linestyle='--', label='GPU Naive (Kernel + PCIe)')
    plt.plot(df['N'], df['GPU_Tiled_Total_ms'], marker='D', color=c_tiled, linewidth=2.5, label='GPU Tiled (Kernel + PCIe)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Tempo de Execucao (ms) - Escala Logaritmica')
    plt.title(f'Comparativo de Tempo de Execucao{title_suffix}', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(df['N'], labels=[str(n) for n in df['N']])
    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7')
    plt.tight_layout()
    plt.savefig('tempo_execucao.png', dpi=300)
    plt.close()

    print(">>> Gerando Grafico 2: Speedup vs N...")
    plt.figure(figsize=(9, 6.5))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.plot(df['N'], df['Speedup_OMP'], marker='s', color=c_omp, linewidth=2.5, label='Speedup CPU OpenMP')
    if 'Speedup_Manual' in df.columns:
        plt.plot(df['N'], df['Speedup_Manual'], marker='p', color=c_manual, linewidth=2.5, label='Speedup CPU AVX2')
    plt.plot(df['N'], df['Speedup_GPU_Naive'], marker='^', color=c_naive, linewidth=2, linestyle='--', label='Speedup GPU Naive')
    plt.plot(df['N'], df['Speedup_GPU_Tiled'], marker='D', color=c_tiled, linewidth=2.5, label='Speedup GPU Tiled')

    plt.axhline(y=1.0, color='#7F8C8D', linestyle=':', linewidth=1.5, label='Linha de Base (CPU Seq)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Fator de Speedup (x vezes mais rapido) - Escala Logaritmica')
    plt.title(f'Speedup Relativo em Relacao a CPU Sequencial{title_suffix}', fontsize=13, fontweight='bold', pad=15)
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
    df['PCIe_Naive_ms'] = df['GPU_Naive_H2D_ms'] + df['GPU_Naive_D2H_ms']
    df['PCIe_Tiled_ms'] = df['GPU_Tiled_H2D_ms'] + df['GPU_Tiled_D2H_ms']

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
    ax.set_title(f'Impacto do Gargalo do PCIe na GPU Tiled{title_suffix}', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in df['N']])
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc='lower right', facecolor='white', edgecolor='#BDC3C7')

    for idx, (pct_kernel, pct_pcie) in enumerate(zip(df['Pct_Kernel_Tiled'], df['Pct_PCIe_Tiled'])):
        if pct_kernel > 5:
            ax.text(idx, pct_kernel / 2, f"{pct_kernel:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if pct_pcie > 5:
            ax.text(idx, pct_kernel + (pct_pcie / 2), f"{pct_pcie:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('gargalo_pcie.png', dpi=300)
    plt.close()

    print(">>> Graficos individuais gerados com sucesso!")
    print("    - tempo_execucao.png")
    print("    - speedup.png")
    print("    - gargalo_pcie.png")


def plot_multiple(datasets):
    print(f">>> Iniciando geracao de graficos comparativos para {len(datasets)} maquinas...")
    
    # Configurar estilo
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.edgecolor'] = '#BDC3C7'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#ECF0F1'
    plt.rcParams['grid.linewidth'] = 0.6

    # Estilos de linha e marcadores para diferenciar computadores
    linestyles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Cores harmoniosas
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F1C40F', '#1ABC9C']

    N_vals = datasets[0][1]['N']

    # --- GRAFICO 1: COMPARATIVO CPU (Seq vs OMP vs AVX2) ---
    print(">>> Gerando comparativo_tempo_cpu.png...")
    plt.figure(figsize=(10, 7))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    for idx, (label, df) in enumerate(datasets):
        style = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        
        plt.plot(df['N'], df['CPU_Seq_ms'], marker=marker, linestyle=style, linewidth=2,
                 label=f'{label} - Seq')
        plt.plot(df['N'], df['CPU_OMP_ms'], marker=marker, linestyle=style, linewidth=2,
                 label=f'{label} - OpenMP')
        if 'CPU_Manual_ms' in df.columns:
            plt.plot(df['N'], df['CPU_Manual_ms'], marker=marker, linestyle=style, linewidth=1.5,
                     label=f'{label} - AVX2')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Tempo de Execucao (ms) - Escala Logaritmica')
    plt.title('Comparativo de Performance de CPU: PC 1 vs PC 2', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(N_vals, labels=[str(n) for n in N_vals])
    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comparativo_tempo_cpu.png', dpi=300)
    plt.close()

    # --- GRAFICO 2: COMPARATIVO GPU (Naive vs Tiled) ---
    print(">>> Gerando comparativo_tempo_gpu.png...")
    plt.figure(figsize=(10, 7))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    for idx, (label, df) in enumerate(datasets):
        style = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        
        plt.plot(df['N'], df['GPU_Naive_Total_ms'], marker=marker, linestyle=style, linewidth=2,
                 label=f'{label} - GPU Naive (Total)')
        plt.plot(df['N'], df['GPU_Tiled_Total_ms'], marker=marker, linestyle=style, linewidth=2.5,
                 label=f'{label} - GPU Tiled (Total)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Tempo de Execucao (ms) - Escala Logaritmica')
    plt.title('Comparativo de Performance de GPU (Total): PC 1 vs PC 2', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(N_vals, labels=[str(n) for n in N_vals])
    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comparativo_tempo_gpu.png', dpi=300)
    plt.close()

    # --- GRAFICO 3: COMPARATIVO GARGALO PCIE (Tempo e Banda Efetiva) ---
    print(">>> Gerando comparativo_gargalo_pcie.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # 3.1: Tempo de transferência PCIe (ms)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    for idx, (label, df) in enumerate(datasets):
        style = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        ax1.plot(df['N'], df['PCIe_Tiled_ms'], marker=marker, linestyle=style, linewidth=2.5, color=color,
                 label=f'{label} (PCIe H2D+D2H)')
                 
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    ax1.set_ylabel('Tempo de Transferencia PCIe (ms) - Log')
    ax1.set_title('Tempo Total Gasto na Transferencia de Dados', fontsize=11, fontweight='bold')
    ax1.set_xticks(N_vals)
    ax1.set_xticklabels([str(n) for n in N_vals])
    ax1.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7')

    # 3.2: Banda Efetiva (GB/s)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    for idx, (label, df) in enumerate(datasets):
        style = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        ax2.plot(df['N'], df['Bandwidth_Tiled_GBs'], marker=marker, linestyle=style, linewidth=2.5, color=color,
                 label=f'{label} (Banda Efetiva)')
                 
    ax2.set_xscale('log')
    ax2.set_xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    ax2.set_ylabel('Largura de Banda Efetiva (GB/s)')
    ax2.set_title('Largura de Banda Efetiva Real vs Limites Teoricos', fontsize=11, fontweight='bold')
    ax2.set_xticks(N_vals)
    ax2.set_xticklabels([str(n) for n in N_vals])

    # Adicionar limites teóricos específicos para PC 1 e PC 2
    for label, _ in datasets:
        label_lower = label.lower()
        if "3080" in label_lower or "pc1" in label_lower or "5600" in label_lower:
            ax2.axhline(y=31.5, color='#3498DB', linestyle=':', alpha=0.8, linewidth=1.5,
                        label='Teorico PCIe 4.0 x16 (31.5 GB/s)')
        elif "4060" in label_lower or "pc2" in label_lower or "5700" in label_lower:
            ax2.axhline(y=7.88, color='#E74C3C', linestyle=':', alpha=0.8, linewidth=1.5,
                        label='Teorico PCIe 3.0 x8 (7.88 GB/s)')
            
    ax2.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7')

    plt.suptitle('Analise Comparativa de Barramento e Gargalo PCIe (GPU Tiled)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('comparativo_gargalo_pcie.png', dpi=300)
    plt.close()

    # --- GRAFICO 4: COMPARATIVO SPEEDUP ---
    print(">>> Gerando comparativo_speedup.png...")
    plt.figure(figsize=(10, 7))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    for idx, (label, df) in enumerate(datasets):
        style = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        color_tiled = colors[idx % len(colors)]
        
        plt.plot(df['N'], df['Speedup_GPU_Tiled'], marker=marker, linestyle=style, linewidth=2.5, color=color_tiled,
                 label=f'{label} - GPU Tiled')
                 
    plt.axhline(y=1.0, color='#7F8C8D', linestyle=':', linewidth=1.5, label='Linha de Base (CPU Seq)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamanho da Matriz (N x N) - Escala Logaritmica')
    plt.ylabel('Fator de Speedup (x vezes mais rapido) - Escala Logaritmica')
    plt.title('Comparativo de Speedup (GPU Tiled vs CPU Sequencial)', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(N_vals, labels=[str(n) for n in N_vals])
    plt.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comparativo_speedup.png', dpi=300)
    plt.close()

    print(">>> Graficos comparativos gerados com sucesso!")
    print("    - comparativo_tempo_cpu.png")
    print("    - comparativo_tempo_gpu.png")
    print("    - comparativo_gargalo_pcie.png")
    print("    - comparativo_speedup.png")


def main():
    # Procurar arquivos de benchmarks
    csv_files = glob.glob("benchmarks_*.csv")
    
    # Adicionar o benchmarks.csv genérico se ele existir
    legacy_file = "benchmarks.csv"
    if os.path.exists(legacy_file):
        if not csv_files:
            csv_files.append(legacy_file)
        elif legacy_file not in csv_files:
            csv_files.append(legacy_file)

    if not csv_files:
        print("[ERRO] Nenhum arquivo de benchmarks (benchmarks_*.csv ou benchmarks.csv) foi encontrado.")
        print("Execute o benchmark C++ primeiro!")
        return

    # Se houver apenas 1 arquivo de benchmark, plota o modo individual clássico
    if len(csv_files) == 1:
        plot_single(csv_files[0])
    else:
        # Carregar múltiplos datasets
        datasets = []
        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
                # Validar colunas necessárias
                required = ['N', 'CPU_Seq_ms', 'CPU_OMP_ms', 'GPU_Naive_Total_ms', 'GPU_Tiled_Total_ms']
                if not all(col in df.columns for col in required):
                    print(f"[AVISO] Ignorando '{filepath}' porque nao contem todas as colunas de benchmark.")
                    continue
                
                df = df[df['Valido'] == 1].copy()
                df = df[df['CPU_Seq_ms'] > 0].copy()
                
                # Calcular métricas básicas
                df['Speedup_OMP'] = df['CPU_Seq_ms'] / df['CPU_OMP_ms']
                df['Speedup_GPU_Tiled'] = df['CPU_Seq_ms'] / df['GPU_Tiled_Total_ms']
                
                df['PCIe_Tiled_ms'] = df['GPU_Tiled_H2D_ms'] + df['GPU_Tiled_D2H_ms']
                # Evitar divisão por zero caso o tempo seja nulo ou muito pequeno
                pcie_safe = df['PCIe_Tiled_ms'].clip(lower=1e-5)
                # Cálculo de largura de banda de PCIe em GB/s (3 * N^2 * 4 bytes de transferência)
                df['Bandwidth_Tiled_GBs'] = (12.0 * df['N'] * df['N']) / (pcie_safe * 1e6)
                
                label = get_label(filepath)
                datasets.append((label, df))
            except Exception as e:
                print(f"[ERRO] Falha ao ler '{filepath}': {e}")

        if len(datasets) < 2:
            if len(datasets) == 1:
                plot_single(csv_files[0])
            else:
                print("[ERRO] Nenhum arquivo de dados valido foi encontrado para comparacao.")
        else:
            # Ordenar por nome de label para consistência
            datasets.sort(key=lambda x: x[0])
            plot_multiple(datasets)

if __name__ == "__main__":
    main()
