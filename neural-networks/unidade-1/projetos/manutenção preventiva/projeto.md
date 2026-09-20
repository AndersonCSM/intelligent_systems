# Projeto de Análise e Manutenção Preventiva (N-CMAPSS)

Este projeto foca na predição da vida útil restante (RUL - *Remaining Useful Life*) de motores aeronáuticos (Turbofans) baseados no dataset sintético N-CMAPSS da NASA, utilizando a arquitetura de **Perceptron Multicamadas (MLP)**.

> [!NOTE]
> Os códigos completos e executáveis estão nos notebooks:
> - **[Exploração de Dados (data_explore.ipynb)](file:///home/anderson/github_projects/intelligent_systems/neural-networks/unidade-1/projetos/manutenção preventiva/data_explore.ipynb)**
> - **[Treinamento do Modelo (model_training.ipynb)](file:///home/anderson/github_projects/intelligent_systems/neural-networks/unidade-1/projetos/manutenção preventiva/model_training.ipynb)**

## 1. O que é o N-CMAPSS?
O N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation) é um simulador da NASA que gera trajetórias de degradação ("run-to-failure") de motores sob condições operacionais **reais de voo**. Durante a vida útil do motor, as peças se degradam até a falha, e o nosso objetivo é prever quando essa falha vai ocorrer (RUL).

Temos três tipos de matrizes de dados principais neste projeto:
- **`W` (Flight Conditions):** Condições operacionais de voo (Altitude, Mach Number, Throttle Resolver Angle, Temperatura ambiente).
- **`X_s` (Physical Sensors):** Sensores físicos reais do motor (temperaturas e pressões). **Estas são as Features principais para o modelo**.
- **`Y` (Target):** RUL, em ciclos de voo.

### 1.1. Conjunto de Dados

- **DS01 e DS02**: Cenários de falhas mais "simples" ou isoladas. Geralmente focam na degradação da eficiência de uma única peça (como a Turbina de Alta Pressão - HPT). O nome `DS02-006`, por exemplo, costuma indicar que existem 6 motores sendo monitorados nesse arquivo. São perfeitos para criar e validar a arquitetura inicial da rede neural (MLP). Eles exigem menos memória e o modelo converge mais rápido, já que o padrão de falha é mais limpo. O próprio notebook de exemplo da NASA foca no DS02.

- **DS03, DS04 e DS05**: O simulador introduz falhas múltiplas e simultâneas. Por exemplo, a Turbina de Alta Pressão (HPT) e o Compressor (HPC) podem começar a degradar ao mesmo tempo, além de misturar perfis de voo diferentes (voos curtos vs. longos). Úteis para testar se a rede neural consegue separar e prever sinais de falhas misturados.

- **DS08**: São os datasets mais complexos e caóticos. Combinam degradação em quase todos os subsistemas ao mesmo tempo, em dezenas de motores voando em rotas completamente diferentes. Recomendado apenas para pesquisa avançada ou validação de robustez extrema do modelo.

> **Decisão:** O projeto seguirá analisando exclusivamente o cenário **DS02**.

## 2. Análise Exploratória dos Dados (EDA)

A análise exploratória completa foi realizada no notebook `data_explore.ipynb`. 

### Destaques da Exploração:
- **Condições de Voo Variáveis**: Observamos que a altitude (`alt`) e o número de Mach (`Mach`) mudam drasticamente de um ciclo para outro, o que significa que o modelo precisará lidar com um ambiente altamente dinâmico.
- **Distribuição dos Sensores**: Através de gráficos KDE, notamos que a densidade das leituras de temperatura (ex: `T24`, `T30`) e pressão (`P15`, `P30`) mudam de forma significativa entre os primeiros ciclos de voo e os ciclos próximos à falha.
- **Variável Alvo (RUL)**: O RUL decai linearmente a cada ciclo. No início da vida, o RUL é alto (ex: 70-80 ciclos dependendo da unidade) e, ao final, chega a zero, sinalizando a falha catastrófica iminente.

## 3. Treinamento do Modelo (MLP)

O pipeline de machine learning e treinamento de rede neural está documentado no notebook `model_training.ipynb`. O fluxo seguiu estritamente as melhores práticas de *Intelligent Systems*:

### 3.1. Pré-processamento e Validação de Premissas
- Juntamos as condições de voo (`W`) e as leituras físicas (`X_s`) como entradas.
- **Scaling Obrigatório**: Uma vez que estamos lidando com sensores de escalas de grandeza diferentes (ex: Temperaturas em °R na casa dos milhares vs. Mach na casa dos decimais), a etapa de *Normalization* foi aplicada com `StandardScaler`. 
- **Separação Treino/Teste**: O scaling foi ajustado (fit) apenas nos dados de treino para evitar vazamento de dados (*data leakage*), sendo depois aplicado no teste.

### 3.2. Arquitetura da Rede Neural
Como estamos prevendo um valor contínuo (RUL), este é um problema de **Regressão**.
Optamos por uma Rede Neural Densa (MLP) usando Keras com a seguinte configuração:
- Camada de entrada recebendo a dimensão combinada de `W` e `X_s`.
- 3 Camadas Ocultas (Dense) com ativação `ReLU` (64, 64 e 32 neurônios).
- Uso de `Dropout(0.2)` nas primeiras camadas para atuar como regularização e mitigar o *overfitting*.
- Camada de Saída com 1 neurônio e ativação `linear`.

### 3.3. Avaliação de Performance
- O modelo foi compilado com o otimizador `Adam` e a função de perda de Erro Quadrático Médio (`MSE`), além da métrica secundária `MAE` (Erro Absoluto Médio).
- Os plots de avaliação no notebook demonstram a curva de aprendizado (Treino vs Validação) para identificar convergência.
- O resultado prático comparando as predições com o **RUL Real** ilustra a capacidade do modelo de alertar a necessidade de manutenção antes da falha.

## Conclusões
O pipeline montado atinge o objetivo proposto, permitindo a ingestão dos complexos arquivos HDF5 do N-CMAPSS e treinando uma MLP capaz de compreender a relação entre o desgaste monitorado nos sensores e o momento da falha do turbofan. 