# Previsão de Séries Temporais Financeiras (Jane Street)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Ambiente-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./license)

Projeto acadêmico desenvolvido para a disciplina de **Redes Neurais (3ª Unidade)**, focado na modelagem, comparação e análise de interpretabilidade de arquiteturas recorrentes (**GRU, LSTM, CNN–LSTM e LSTM com Mecanismo de Atenção**) aplicadas à previsão de retornos financeiros multivariados de alta frequência utilizando o dataset **Jane Street Real-Time Market Data Forecasting**.

Este documento organiza o projeto em:
- **Contexto, Arquitetura de Repositório e Ambiente**;
- **Core**: pipeline base;
- **Testes Comparativos**: ablações controladas em blocos;
- **Resultados, Figuras, Interpretabilidade (XAI) e Trabalhos Futuros**.

---

## Sumário

1. [Introdução ao Problema e ao Dataset](#1-introdução-ao-problema-e-ao-dataset)
2. [Organização do Repositório](#2-organização-do-repositório)
3. [Tecnologias e Configuração do Ambiente](#3-tecnologias-e-configuração-do-ambiente)
4. [Core](#4-core-pipeline-base)
5. [Testes Comparativos](#5-testes-comparativos-estudos-de-ablação)
6. [Resultados Obtidos e Visualizações](#6-resultados-obtidos-e-visualizações)
7. [Interpretabilidade da Atenção](#7-interpretabilidade-da-atenção-xai)
8. [Trabalhos Futuros](#8-trabalhos-futuros)
9. [Autores](#9-autores)
10. [Licença](#10-licença)
11. [Referências](#11-referências)

---

## 1. Introdução ao Problema e ao Dataset

A previsão de retornos em mercados financeiros de alta frequência é um desafio clássico de séries temporais multivariadas, caracterizado por baixa relação sinal-ruído, não estacionariedade e mudanças abruptas de regime (*regime shifts*). O objetivo deste projeto é desenvolver e comparar arquiteturas de redes neurais recorrentes (**GRU, LSTM, CNN–LSTM e LSTM com Mecanismo de Atenção**) capazes de capturar dependências temporais nesse cenário.

O conjunto de dados utilizado é baseado na competição **Jane Street Real-Time Market Data Forecasting** (via espelho público no Kaggle), composto por:

- **Estrutura temporal:** Registros indexados por dia de negociação (`date_id`), instante intradiário (`time_id`) e ativo (`symbol_id`).
- **Variáveis de entrada (`features.csv` / `train.parquet`):** 79 *features* anonimizadas e derivadas de sinais de mercado (`feature_00` a `feature_78`), com presença frequente de valores ausentes (`NaN`).
- **Variáveis alvo e peso (`responders.csv` / `train.parquet`):** 9 respostas de mercado em diferentes horizontes (`responder_0` a `responder_8`), sendo `responder_6` o alvo principal, acompanhado da coluna `weight`, que define a relevância de cada amostra na avaliação.

---

## 2. Organização do Repositório

O projeto é estruturado para integração nativa com o **Google Colab**, evitando perda de artefatos entre sessões de GPU:

```text
projeto/
├── data/
│   ├── raw/
│   │   ├── train.parquet       # Série temporal principal (features, alvos e pesos)
│   │   ├── lags.parquet        # Defasagens auxiliares fornecidas na base
│   │   ├── features.csv        # Metadados e tags das 79 features
│   │   ├── responders.csv      # Metadados dos 9 alvos (responders)
│   │   └── Data info.txt       # Documentação descritiva original da base
│   └── jane street/            # Diretório destino gerado pelo loader.ipynb
├── graphs/                     # Curvas de aprendizado, comparativos de R² e mapas de atenção
├── models/                     # Checkpoints (.keras / .h5) e scalers ajustados no treino
├── loader.ipynb                # Notebook de autenticação Kaggle API e download para o Drive
├── train_core.ipynb            # Notebook principal: pré-processamento, CV temporal, modelos e ablações
├── README.md                   # Documentação técnica do projeto
├── license                     # Licença MIT
└── .gitignore                  # Exclusão de arquivos pesados (*.parquet, checkpoints locais, cache)
```

---

## 3. Tecnologias e Configuração do Ambiente

### Stack Tecnológica
- **Linguagem:** Python 3.10+
- **Deep Learning:** TensorFlow 2.x / Keras (`GRU`, `LSTM`, `Conv1D`, `Attention`, `sample_weight`)
- **Manipulação de Dados e Seleção de Atributos:** Pandas, NumPy, PyArrow / Polars, Scikit-Learn (`StandardScaler`, `RandomForestRegressor`), XGBoost
- **Visualização e XAI:** Matplotlib, Seaborn
- **Infraestrutura:** Google Colab (Runtime com GPU T4/L4 + High-RAM) conectado ao Google Drive

### Passo a Passo para Execução

1. **Clonar ou sincronizar o repositório** na pasta `MyDrive/projeto3/` do Google Drive.
2. **Ingestão dos Dados (`loader.ipynb`):**
3. **Treinamento e Experimentos (`train_core.ipynb`):**
   - Abra `train_core.ipynb` com acelerador de hardware **GPU** ativado.
   - Execute sequencialmente o pipeline **Core** (que salva os pesos em `models/` e os gráficos em `graphs/`) e, em seguida, os blocos de **Testes Comparativos**.

---

## 4. Core

### Ingestão e Eficiência de Memória

- **Downcasting de Tipos:** Converter `float64` para `float32` (testar `float16` se estável) para reduzir o uso de RAM no Colab e permitir sequências maiores.
- **Filtro de Período Estável (`date_id >= 700`):** Usar dados a partir do dia 700, onde a estrutura diária se estabiliza em 968 `time_id`s, reduzindo *domain shift* e tempo de treino.
- **Seleção de Features:** Top 30–50 via *Feature Importance* (RF/XGBoost), calculada estritamente no treino para podar ruído temporal antes de alimentar as redes recorrentes.

### Pré-processamento Baseline

- **Padronização sem Vazamento:** `StandardScaler` ajustado (`fit`) exclusivamente no treino e aplicado (`transform`) na validação e teste.
- **Imputação Neutra (`NaN = 0`):** Preencher nulos com `0.0` após a padronização (escolha com maior respaldo empírico para sinais anonimizados).

### Estratégia de Validação Rigorosa

- **Proibição de Split Aleatório:** Divisão estritamente cronológica por `date_id` para evitar *data leakage* do futuro.
- **Validação Cruzada Temporal (Time-Series CV):** 2 *folds* com janelas de validação de `200 date_ids` cada.
- **Simulação de Cenário Privado (Gap):** Treino no primeiro *fold* e teste nos últimos `200 date_ids` com um **gap (embargo) de 200 dias** entre treino e teste, simulando mudança de regime de mercado.

### Treinamento, Pesos e Métricas

| Componente | Decisão | Observação |
| :--- | :--- | :--- |
| **Função de Perda** | `loss='mse'` + `sample_weight=weight` | Minimizar MSE ponderado é matematicamente equivalente a maximizar o Weighted $R^2$ no conjunto completo; amostras com `weight == 0` são ignoradas automaticamente pelo Keras. |
| **Otimizador / Batch** | `Adam(lr=1e-3)` + `ReduceLROnPlateau` | `batch_size` de 4096 ou 8192 (conforme VRAM disponível). |
| **Early Stopping** | `patience=5` monitorando **Weighted $R^2$** | Alinhado diretamente à métrica principal. |
| **Métrica Principal** | **Zero-Mean Weighted $R^2$** | Calculada sobre o conjunto inteiro de validação/teste (**nunca** média simples por *batch*). |
| **Métricas Secundárias** | **RMSE** e **MAE** (ponderados) | Garantem comparabilidade com a literatura acadêmica. |

### Arquitetura e Hiperparâmetros Baseline

```text
Input (T, F)
   │
   ▼
GRU(units=64, return_sequences=False)
   │
   ▼
Dense(64, activation='relu') → Dropout(0.2)
   │
   ▼
Dense(1, activation='linear')
```

- **Estrutura da Sequência:** Um dia de negociação por sequência (`date_id`, `T = 968`) ou janelas `T` (`64`, `128`, `256`) contidas dentro do mesmo dia.
- **Persistência:** Os melhores checkpoints de cada fold são exportados automaticamente para `models/core_gru_fold{k}.keras`.

---

## 5. Testes Comparativos (Estudos de Ablação)

Alterar **um fator por vez** sobre o **Core** e medir o impacto no *Weighted $R^2$* do CV temporal, mantendo o mesmo pipeline de dados e features (top 30–50, `NaN=0`, `date_id >= 700`):

| Modelo | Descrição | Hipótese a Testar |
| :--- | :--- | :--- |
| **GRU** | 1 camada GRU + 2 Dense (Dropout 0.2) | Baseline de referência. |
| **LSTM** | Substitui GRU por LSTM | Expectativa: `GRU ≥ LSTM` em dados ruidosos. |
| **CNN–LSTM** | `Conv1D` → `MaxPool` → `LSTM` | Extração de padrões locais antes da recorrência. |
| **LSTM + Attention** | `LSTM(return_sequences=True)` + Self-Attention | Foco em `time_id`s mais relevantes (evitar *cross-symbol attention*). |

### Bloco A — Imputação de Dados Ausentes

| Papel | Configuração | Descrição |
| :--- | :--- | :--- |
| **Padrão (Core)** | **NaN = 0** | Padronização + zero |
| **Teste** | **Forward Fill (FFIL)** | `ffill` por `symbol_id`, ordenado por `date_id`/`time_id`, depois padronização, depois zero para nulos iniciais |

### Bloco B — Engenharia de Features Incremental

Adiciona grupos de features **uma etapa por vez**, medindo o ganho incremental no CV:

| Etapa | Papel | Features adicionadas |
| :--- | :--- | :--- |
| **B0** | **Padrão** | Features do Core (top 30–50, sem features derivadas) |
| **B1** | **Teste** | + **Médias de mercado (cross-sectional):** média de cada feature agrupada por `date_id` e `time_id` (snapshot transversal) |
| **B2** | **Teste** | + **Estatísticas rolantes por `symbol_id`:** média e std em janelas curtas (5/10 passos) e longas (até 1000 `time_id`s) |
| **B3** | **Teste** | + **Interações temporais:** ex.: `feature_06 * time_id` |
| **B4** | **Teste** | + **Agregados estatísticos por linha:** `mean`, `std`, `range` across features |

**Justificativa:** o 8º colocado relatou **+0.002** com todas as novas features; o `AKHIL-149` relatou **+29%** de R² sobre a baseline com interações e agregados. A ordem B1→B4 segue "do mais informativo ao mais exploratório", permitindo que o artigo mostre **exatamente onde o ganho aparece**.

> **Distinção importante:** B1 (**cross-sectional**) e B2 (**rolling**) são features **complementares, não substitutas**. B1 dá contexto transversal do mercado naquele instante; B2 dá dinâmica temporal individual de cada ativo. A solução de topo usou **ambas**.

### Bloco C — Estrutura da Sequência

| Papel | Configuração | Descrição |
| :--- | :--- | :--- |
| **Padrão (Core)** | **Um dia por sequência** | `T = 968` (`time_id`s por dia) |
| **Teste** | **Janelas deslizantes curtas** | `T ∈ {64, 128, 256}` dentro do dia |

### Bloco D — Alvos Auxiliares

| Papel | Configuração | Descrição |
| :--- | :--- | :--- |
| **Padrão (Core)** | **Apenas `responder_6`** | Regressão single-task |
| **Teste** | **`responder_6` + `7` + `8`** | Multi-task, ponderado pela loss |

### Bloco E — Robustez da Loss

| Papel | Configuração | Descrição |
| :--- | :--- | :--- |
| **Padrão (Core)** | **MSE + `sample_weight`** | Equivalente matemático ao Weighted R² |
| **Teste** | **Huber + `sample_weight`** | Menos sensível a outliers |

---

## 6. Resultados Obtidos e Visualizações

*(Preencher após a execução completa de `train_core.ipynb` e exportação dos gráficos para `graphs/`)*

### 6.1 Comparativo Geral das Arquiteturas (Validação Temporal)

| Modelo | Weighted R² (Fold 1) | Weighted R² (Fold 2) | Weighted R² (Gap / Privado) | RMSE | MAE | Parâmetros | Tempo (min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GRU (Core)** | — | — | — | — | — | — | — |
| **LSTM** | — | — | — | — | — | — | — |
| **CNN–LSTM** | — | — | — | — | — | — | — |
| **LSTM + Attention** | — | — | — | — | — | — | — |

### 6.2 Síntese dos Estudos de Ablação (Blocos A–E)

| Bloco | Experimento | Weighted R² (Val) | Delta vs. Core ($\Delta R^2$) | Decisão / Observação |
| :--- | :--- | :---: | :---: | :--- |
| **Core** | Baseline (`GRU`, `NaN=0`, `T=968`, `MSE`) | — | `0.0000` | Referência |
| **Bloco A** | Imputação `Forward Fill (ffill)` | — | — | — |
| **Bloco B** | Feature Engineering (`B1` a `B4` cumulativo) | — | — | — |
| **Bloco C** | Janelas Deslizantes (`T ∈ {64, 128, 256}`) | — | — | — |
| **Bloco D** | Multi-Task (`responder_6 + 7 + 8`) | — | — | — |
| **Bloco E** | `Huber Loss` + `sample_weight` | — | — | — |

### 6.3 Figuras Geradas (`graphs/`)

Os seguintes artefatos visuais são gerados automaticamente ao final da execução para compor o artigo:

- `graphs/feature_importance_top50.png` — Ranking das 50 *features* selecionadas no conjunto de treino (RF/XGBoost).
- `graphs/learning_curves_models.png` — Evolução de *Loss* (MSE ponderado) e *Weighted $R^2$* ao longo das épocas por arquitetura.
- `graphs/ablation_delta_r2.png` — Gráfico de barras comparando o ganho marginal ($\Delta R^2$) de cada bloco de ablação (A–E).
- `graphs/pred_vs_real_sample.png` — Aderência temporal entre `responder_6` real e predito ao longo de um dia de teste (`date_id`).
- `graphs/attention_weights_intraday.png` — Distribuição média dos pesos de atenção por `time_id` intradiário.

---

## 7. Interpretabilidade da Atenção (XAI)

- Extrair e analisar os pesos da camada de *Self-Attention* do modelo **LSTM + Attention** ao longo dos 968 instantes diários (`time_id`), identificando quais períodos do pregão (abertura, meio do dia ou fechamento) concentram maior relevância preditiva para o `responder_6`.

---

## 8. Trabalhos Futuros

- **Online Learning (Aprendizagem Contínua):** Atualizar os pesos incrementalmente durante o teste (no pódio da competição, elevou o Weighted $R^2$ de `0.01413` congelado para `0.01942`, ganho de `+0.00529`).
- **Ensembling Híbrido e Massivo:** Combinar redes recorrentes com modelos tabulares (XGBoost/LightGBM) em escala (10+ modelos diversos).
- **Otimização Sistemática de Hiperparâmetros:** Busca automatizada (Optuna/Hyperband) para `units`, `dropout`, `T`, `batch_size` e `lr`.
- **Transferibilidade:** Testar o mesmo pipeline em outros datasets financeiros multivariados (ex.: *Limit Order Book*, *Optiver* ou cripto).

---

## 9. Autores

| Autor | Perfil |
| :--- | :--- |
| **Anderson Carlos da Silva Morais** | [GitHub](https://github.com/AndersonCSM) |
| **Pablo Geraldo Linhares do Nascimento Leite** | [GitHub](https://github.com/pablonascimento1) |

*Instituição: Universidade Federal Rural do Semi-Árido (UFERSA) — Engenharia de Computação.*

---

## 10. Licença

Este projeto está licenciado sob a **Licença MIT** — consulte o arquivo [`license`](./license) na raiz do repositório para obter mais detalhes.

---

## 11. Referências

- **Competição Original:** Jane Street Real-Time Market Data Forecasting (Kaggle). Disponível em: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting
- **Espelho Público dos Dados:** Mohamed Sameh — *Jane Street Dataset* (Kaggle Datasets). Disponível em: https://www.kaggle.com/datasets/mohamedsameh0410/jane-street-dataset/data
- **Referências Técnicas da Competição:**
  - *Group 3 Time Series (Part 1)* — https://www.kaggle.com/code/hangyuzhou0604/group3-timeseries-part1
  - *XGBoost for R²* — https://www.kaggle.com/code/angerr/xgboost-for-r
  - *Immutables JSRMF* — https://www.kaggle.com/code/harshavardhankonisa/immutables-jsrmf
  - *Jane Street Time Series Analysis, EDA & Ensemble* — https://www.kaggle.com/code/marylumin/jane-street-time-series-analysis-eda-ensemble
