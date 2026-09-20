# Predição da Vida Útil Restante (RUL) de Motores Turbofan utilizando Rede Neural MLP

Este projeto foi desenvolvido como parte da disciplina de **Redes Neurais Artificiais**, na Universidade Federal Rural do Semi-Árido (UFERSA).

---

## Introdução e Contexto

A manutenção preventiva é uma das estratégias mais empregadas para garantir o funcionamento adequado de dispositivos de alto valor e risco, como as turbinas de aeronaves na aviação comercial. A falha não prevista desses motores afeta diretamente a segurança humana e causa prejuízos milionários.

Neste projeto, uma **Rede Neural Multilayer Perceptron (MLP)** puramente orientada a dados foi desenvolvida para predizer a **Vida Útil Restante (RUL - Remaining Useful Life)** de motores turbofan operando em condições reais de voo, analisando sensores físicos e telemetria para antecipar a falha estrutural.

---

## Objetivo

Desenvolver uma MLP capaz de estimar, em ciclos de voo, quanto tempo de operação um motor aeronáutico ainda possui antes de falhar, baseando-se no histórico de degradação capturado por sensores termodinâmicos durante as fases de subida, cruzeiro e descida.

Adicionalmente, o projeto garante o rigor metodológico para evitar o *Data Leakage* (vazamento de dados) e implementa técnicas avançadas de **Explainable AI (SHAP)** para validar se a predição da rede está coerente com as leis da física.

---

## Tipo do Problema

O problema é de **Regressão**, utilizando a variável escalar `RUL` (medida em ciclos de voo) como saída.

A rede neural utiliza uma camada de saída com um único neurônio linear para produzir o valor contínuo da predição.

---

## Dataset

Foi utilizado o rigoroso conjunto de dados **N-CMAPSS** (*New Commercial Modular Aero-Propulsion System Simulation*), fornecido pela NASA.

🔗 [NASA Prognostics Data Repository - N-CMAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

O arquivo específico extraído para este projeto modela a degradação na Turbina de Alta Pressão (HPT) e Baixa Pressão (LPT) utilizando a matriz DS02-006:

```
N-CMAPSS_DS02-006.h5
```

O projeto divide estritamente as frotas para avaliação justa:
- **Treinamento:** Motores 1 ao 10 (`_dev`)
- **Teste:** Motores 11 ao 20 (`_test`)

---

## Características Utilizadas

Após descartar deliberadamente sensores virtuais e parâmetros de saúde inacessíveis no mundo real, a modelagem foi construída exclusivamente sobre **18 grandezas físicas brutas**:

- **4 Condições Operacionais:** Altitude, Mach, TRA, Temperatura Total na Entrada (T2).
- **14 Sensores Físicos:** Temperaturas (T24, T30, T48, T50), Pressões (P15, P2, P21, P24, Ps30, P40, P50), Fluxo de Combustível (Wf) e Velocidades Físicas (Nf, Nc).

**Engenharia de Variáveis Temporais:**
Para contornar a extrema volatilidade e ruído do voo, aplicou-se um **Janelamento Estatístico** (Sliding Window de 20 ciclos). Para cada um dos 18 sensores, extraiu-se:
- Leitura Bruta (`_raw`)
- Média Móvel (`_mean`)
- Variância Móvel (`_var`)

O vetor de entrada da rede saltou para **54 características finais**.

**Variável de saída:**
- `RUL` (Remaining Useful Life)

---

## Metodologia

### Pré-processamento e Amostragem
Devido à amostragem de altíssima frequência (Hz) do N-CMAPSS, os dados possuem altíssima colinearidade. Para otimizar o custo computacional, foi realizada uma subamostragem aleatória estratificada que reteve apenas **30% das instâncias de cada motor** durante o treinamento, o que também atuou como um agente regularizador.

Os dados foram padronizados usando o método **Z-Score** (`StandardScaler`), ajustado exclusivamente sobre os dados de treinamento (Motores 1-10) e aplicado na validação (Motores 11-20). A matriz final foi reduzida à precisão `float32`.

---

## Modelagem

A rede **Multilayer Perceptron (MLP)** foi refinada utilizando o framework Keras Tuner, que comprovou matematicamente a seguinte arquitetura ótima profunda:

```
54 features (Raw, Mean, Var)
   ↓
Dense – 256 neurônios – ReLU (+ BatchNormalization)
   ↓
Dense – 128 neurônios – ReLU (+ BatchNormalization)
   ↓
Dense – 64 neurônios – ReLU (+ BatchNormalization)
   ↓
Dense – 1 neurônio – Linear
   ↓
RUL Predito (Ciclos)
```

### Configurações

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam (LR: 0.001) |
| Função de perda (Loss) | Erro Quadrático Médio (MSE) |
| Métrica Secundária | MAE |
| Batch size | 4096 |
| Regularização Adicional | ReduceLROnPlateau (fator 0.5) e EarlyStopping (6 épocas) |
| Dropout | Não utilizado na arquitetura final |

---

## Avaliação do Modelo

Para avaliar a regressão, as seguintes métricas foram implementadas no conjunto de Motores de Teste (totalmente desconhecidos pela IA na fase de treino):
- $R^2$ (Coeficiente de Determinação)
- MAE (Erro Absoluto Médio)
- RMSE (Raiz do Erro Quadrático Médio)
- MSE (Erro Quadrático Médio)

---

## Resultados

O modelo puramente orientado a dados apresentou o seguinte desempenho base:

| Métrica | Valor |
|---|---:|
| $R^2$ | **86,42%** |
| MAE | **5,16 ciclos** |
| RMSE | **6,99 ciclos** |
| MSE | **48,85 ciclos** |

O coeficiente de 86,42% é altamente relevante por ter sido obtido sob um cenário rigoroso de teste e livre de *data leakage*.

### Inteligência Artificial Explicável (SHAP)
Foi aplicado o framework **SHAP (Game Theory)**. Os resultados evidenciaram coerência termodinâmica impecável:
- O impacto da **Temperatura na saída da Turbina de Baixa Pressão (`T50_mean`)** dominou a rede, confirmando os princípios físicos de fadiga térmica em rotores aeronáuticos.
- O predomínio maciço das features do tipo `_mean` comprova que a rede neural valorizou a "tendência temporal" e rejeitou o ruído instantâneo das manobras de voo.

---

## Tecnologias Utilizadas

- Python
- TensorFlow / Keras (com Keras Tuner)
- SHAP (Explainable AI)
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Google Colaboratory (GPU Tesla T4)

---

## Como Executar

O projeto foi dividido em notebooks modulares para facilitar a reprodutibilidade técnica.

**1. Dependências principais:**
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn shap keras-tuner h5py
```

**2. Estrutura de Pastas Esperada:**
* Crie uma pasta `data/` na raiz do projeto (no mesmo nível da pasta que contém os scripts) e insira o dataset baixado da NASA (`N-CMAPSS_DS02-006.h5`).

**3. Ordem de Execução dos Notebooks:**
1. `data_graph.ipynb`: Análise exploratória da degradação térmica.
2. `model_tuner.ipynb`: Busca sistemática de hiperparâmetros e topologia da rede.
3. `model_training.ipynb`: Pré-processamento (Sliding Windows) e treinamento da rede.
4. `model_avaliation.ipynb`: Cálculo do $R^2$, MAE, RMSE e geração das curvas reais de degradação.
5. `model_explain.ipynb`: Aplicação do SHAP para auditar a rede globalmente e localmente.

---

## Limitações e Trabalhos Futuros

- **Teto da Arquitetura MLP:** O janelamento estatístico comprime o tempo, mas não permite à MLP compreender as relações sequenciais puras. Isso impõe um "teto operacional" no $R^2$.
- Recomenda-se para trabalhos futuros o uso de Limite Fixo de RUL (*Piecewise Linear RUL*), Engenharia de derivadas de sensores e a substituição da MLP por Redes Neurais Recorrentes (LSTM/GRU) ou Transformers de Séries Temporais.

---

## Autor
<table align="center">
  <tr> 
    <td align="center">
      <a href="https://github.com/AndersonCSM">
        <img src="https://avatars.githubusercontent.com/u/9919?v=4" width="120px;" alt="Foto do Autor"/><br>
        <sub>
          <b>Anderson</b>
         </sub>
      </a>
    </td>
  </tr>
</table>

## Licença

Este projeto está disponibilizado sob a **Licença MIT**.
O dataset N-CMAPSS é propriedade da NASA e está sujeito aos respectivos termos de uso da agência governamental.
