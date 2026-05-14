# Análise de Desempenho em Arquiteturas Paralelas: CPU vs GPU

**Disciplina:** Arquitetura de Computadores  
**Autores:** Arthur Iwankiu Castro e Enzo da Silva Passos  
**Instituição:** Universidade Católica de Santos (UniSantos)

## 📌 Sobre o Projeto

Este projeto apresenta um estudo comparativo de performance entre execução sequencial e paralela na multiplicação de matrizes de grande escala. O objetivo é analisar o tempo de execução e o _speedup_ entre diferentes arquiteturas, identificando o gargalo de comunicação via barramento PCIe.

**Hardwares Analisados:**

- **PC 1:** CPU AMD Ryzen 5 5600 + GPU NVIDIA RTX 3080 (Ampere)
- **PC 2:** CPU AMD Ryzen 7 5700G + GPU NVIDIA RTX 4060 (Ada Lovelace)

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** C / C++
- **Paralelismo em CPU:** OpenMP
- **Paralelismo em GPU:** NVIDIA CUDA C++
- **Compiladores:** GCC (com suporte a `-fopenmp`) e NVCC (NVIDIA CUDA Compiler)

## 📂 Estrutura do Repositório

_(A ser preenchido durante o desenvolvimento)_

- `/src` - Códigos fonte (Sequencial, OpenMP, CUDA)
- `/data` - Planilhas com as coletas de tempo
- `/docs` - Gráficos de Speedup gerados

## 🚀 Como Compilar e Executar

_(As instruções serão adicionadas na fase de desenvolvimento)_

**1. Versão Sequencial:**
\`\`\`bash

# Comando de compilação em breve

\`\`\`

**2. Versão OpenMP:**
\`\`\`bash

# Comando de compilação em breve

\`\`\`

**3. Versão CUDA:**
\`\`\`bash

# Comando de compilação em breve

\`\`\`

## 📊 Resultados e Conclusão

_(Os gráficos de Speedup e a análise sobre o custo de transferência de dados [Host -> Device] serão adicionados aqui na fase final do projeto)._
