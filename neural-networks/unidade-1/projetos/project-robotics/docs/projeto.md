# Projeto MLP + Webots — Função de Custo para Navegação Reativa (Versão Aprimorada)

**Objetivo:** Treinar uma MLP (Perceptron Multicamadas) para regredir uma função de custo contínua que representa o risco de colisão de um robô e-puck no Webots. O robô utilizará essa função para amostrar diferentes direções e escolher aquela que minimiza o custo, resultando em um movimento suave, orgânico e livre, sem depender de regras manuais (`if/else`) ou classificadores discretos.

---

## 1. O Conceito

| Componente | Descrição |
| :--- | :--- |
| **Entrada (Input)** | Leitura dos 8 sensores infravermelhos (IR) do e-puck: `ps0` a `ps7`. |
| **Saída (Output)** | Um valor escalar contínuo (0 a 1) representando o **Índice de Risco** ou **Custo**. <br> `0.0` = Espaço completamente livre e seguro. <br> `1.0` = Risco iminente de colisão (obstáculo muito próximo ou tocando). |

A MLP aprenderá o mapeamento não-linear entre o padrão de luz refletida nos sensores e a distância real para os obstáculos ao redor, funcionando como um **Campo Potencial Artificial Aprendido**.

---

## 2. Melhoria Crítica: Ajuste do Alcance dos Sensores (Lookup Table)

**Problema:** No Webots, o sensor IR do e-puck tem alcance padrão de **~4 cm**. Se o robô estiver a 20 cm da parede, os sensores retornam `0.0`. A MLP aprenderia que "distância longa = custo zero", mas só reagiria a 4 cm da parede, causando curvas bruscas e tardias.

**Solução:** Edite o arquivo `.proto` do e-puck (ou defina no código) para estender a `lookupTable` do sensor para **20 cm**, adicionando um ponto intermediário:

```json
// lookupTable padrão (muito curta):
// [0.0, 0.0, 0.0], [0.04, 1000.0, 0.0], [0.05, 4096.0, 0.0]

// lookupTable estendida (recomendada):
"lookupTable": [
    [0.0,    0.0,     0.0],
    [0.02,   500.0,   0.0],   // Ponto intermediário para suavizar
    [0.20,   4096.0,  0.0]    // Alcance máximo estendido para 20cm
]
```
Isso garante que os sensores respondam em distâncias relevantes para a navegação reativa.

---

## 3. Coleta de Dados (Estratégia Unificada)

### 3.1. Coleta por Grade Supervisionada (Webots + Supervisor)

1. Use o nó Supervisor para teleportar o e-puck para posições pré-definidas (x, z) dentro da arena.
2. Em vez de uma grade quadrada uniforme, utilize uma **Grade Logarítmica** ou **Amostragem com Rejeição**:
   - Gere coordenadas aleatórias uniformes, mas descarte aquelas onde a distância mínima até a parede/obstáculo seja maior que 0.3m. Isso força o dataset a ter uma densidade maior de amostras na região onde os sensores IR realmente respondem (0 a 20 cm).
3. Em cada posição, faça o robô girar em 8 ângulos (ex: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°).
4. Para cada combinação (posição + ângulo), registre:
   - Os 8 valores dos sensores.
   - A distância euclidiana mínima real até o obstáculo/parede mais próximo (calculada via geometria da arena ou usando a API de Supervisor para ler boundingObject).
   - **Observação Crítica (Deslocamento do Sensor):** O cálculo da distância deve subtrair o raio do robô (ex: `0.037m` para o e-puck) da distância até o centro. Isso garante que a `distance_min` gerada no dataset represente o espaço livre a partir da *borda* do robô (onde os sensores estão montados). Fazer isso na fase de coleta assegura que `distance = 0` equivalha a uma colisão física perfeita.

### 3.2. Estrutura Final do CSV (Ajustada)

| Colunas | Nome | Tipo | Descrição |
| :--- | :--- | :--- | :--- |
| 1 a 8 | `ps0` a `ps7` | `float` (0~4096) | Leitura bruta dos sensores de proximidade. |
| 9 | `ps0_norm` a `ps7_norm` | `float` (0~1) | Valores dos sensores normalizados (divididos por 4096). Guardar ambos para testes. |
| 10 | `distance_min` | `float` | (Metadado) Distância real em metros até o obstáculo mais próximo. |
| 11 | `cost` (Target) | `float` (0~1) | **A SAÍDA DA MLP**.<br> Cálculo: `cost = exp(-alpha * distance_min)`, com `alpha = 5` (ajustável).<br> Se `distance = 0` (colisão), `cost = 1.0`. Se `distance = 0.5m`, `cost ≈ 0.08`. |
| 12 | `collision` (Flag) | `int` (0 ou 1) | Apenas para metadado/filtragem. Se 1, o robô está sobreposto à geometria. Descartar essas linhas no treino ou forçar `cost = 1.0`. Nunca usar como feature de entrada. |

> **Nota:** A flag collision não é necessária para o treinamento, pois a MLP aprenderá automaticamente que "sensores saturados no máximo" correspondem a `cost = 1.0`.

---

## 4. Pré-processamento dos Sensores (Feature Engineering)

Jogar o valor bruto (0~4096) diretamente na MLP faz com que ela tenha dificuldade para aprender distâncias longas, pois a curva de resposta do IR é não-linear (inversa ao quadrado).

**Melhoria:** Normalize e transforme os dados de entrada para um espaço que facilite a regressão.

**Opção A (Recomendada para iniciantes):**
Use apenas o valor normalizado entre 0 e 1.

```python
sensor_norm = sensor_raw / 4096.0  # Range: 0.0 a 1.0
```

**Opção B (Linearização aproximada):**
Converta a leitura para uma estimativa de distância (em metros) antes de alimentar a MLP. Isso torna o problema mais linear.

```python
# Fórmula empírica para aproximar a distância (ajuste fino necessário)
# Baseado na curva típica: distância = 0.1 * (1 - norm) / (norm + 0.01)
distance_est = 0.1 * (1 - sensor_norm) / (sensor_norm + 0.01)

# Agora, use distance_est (capado em 0.2m) como entrada. 
```

**Decisão final para o projeto:** Salvar os valores normalizados (0~1) no CSV e utilizá-los como entrada da MLP.

---

## 5. Treinamento da MLP (Regressão)

### 5.1. Arquitetura Aprimorada

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # Entrada: 8 sensores normalizados
    layers.Input(shape=(8,)),
    
    # 1ª Camada Oculta
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),  # Regularização para evitar overfitting
    
    # 2ª Camada Oculta
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    # Camada de Saída (Regressão)
    layers.Dense(1, activation='linear')  # Saída linear (não Sigmoid)
])
```

**Justificativas:**
- **Dropout:** O dataset terá cerca de 3200 amostras (grade 20x20 * 8 ângulos). Dropout é essencial para evitar overfitting.
- **Ativação Linear na Saída:** Como o `cost = exp(-alpha * distance)` gera valores assintóticos (nunca exatamente 0 ou 1), uma ativação Sigmoid forçaria a saída a se aproximar de extremos artificiais. A camada Linear permite valores contínuos e suaves. Na inferência, use `np.clip(prediction, 0, 1)` para segurança.

### 5.2. Função de Perda
Utilize Erro Quadrático Médio (MSE) ou Huber Loss (mais robusto a outliers).
No Keras: `loss='huber'` ou `loss='mse'`.

### 5.3. Validação
- Separe 20% dos dados para teste.
- Plote um gráfico de Custo Real x Custo Predito. A diagonal perfeita indica bom aprendizado.
- Calcule o R² Score para medir a qualidade da regressão.

---

## 6. Navegação: "Giro Virtual" (Substituindo o Giro Físico)

Virar o robô fisicamente no Webots para cada ângulo candidato (a cada 100ms) é computacionalmente caro e provoca travamentos/hesitações.

**Solução:** Implemente uma rotação virtual do vetor de sensores via interpolação circular.

### 6.1. Algoritmo de Interpolação
Os 8 sensores estão espaçados em ângulos fixos no chassi do robô.
Para simular uma rotação de theta graus (ex: +15°), desloque os índices do array circularmente e interpole os valores.

```python
import numpy as np
from scipy.interpolate import interp1d

def virtual_rotate(sensor_values, angle_deg):
    """
    sensor_values: array de 8 elementos (ps0 a ps7)
    angle_deg: ângulo a ser rotacionado (positivo = anti-horário)
    Retorna: novo array de 8 elementos rotacionados.
    """
    # Ângulos originais dos sensores no e-puck (aproximados em graus)
    orig_angles = np.array([72.8, 44.1, 0.0, -61.5, -118.8, -180.0, 135.8, 107.2]) # Ajuste fino necessário
    
    # Ordenar os ângulos para interpolação circular
    # ... (implementação de interpolação seno/cosseno ou linear com wrapping)
    # Exemplo conceitual: deslocar o array circularmente e interpolar
    pass
```
*Ou, simplificando:* Para cada ângulo candidato, calcule o novo ângulo de cada sensor e interpole os valores lidos usando uma função de interpolação linear (como `numpy.interp`) considerando a periodicidade de 360°.

### 6.2. Pipeline de Decisão (Controller)

1. Leia os 8 sensores atuais (array A).
2. Defina 7 ângulos candidatos: [-60°, -40°, -20°, 0°, +20°, +40°, +60°].
3. Para cada ângulo, gere o vetor de sensores rotacionado virtualmente (A_rot).
4. Alimente cada A_rot na MLP para obter o custo predito.
5. Escolha o ângulo com menor custo.
6. Aplique o movimento.

---

## 7. Controle Motor (Suavização e Filtragem)

Para evitar zigue-zagues bruscos, não aplique o ângulo de menor custo diretamente. Utilize um filtro de média móvel exponencial (passa-baixa).

```python
# Variáveis globais (ou persistentes no controller)
steering_filtered = 0.0
alpha = 0.3  # Fator de suavização (quanto menor, mais suave)

# Loop principal
while wb_robot_step():
    target_angle = get_best_angle()  # Ângulo de menor custo (-60 a +60)
    
    # Filtro exponencial
    steering_filtered = alpha * target_angle + (1 - alpha) * steering_filtered
    
    # Converter para velocidades das rodas
    base_speed = 2.0  # rad/s
    left_speed = base_speed - steering_filtered * 1.2
    right_speed = base_speed + steering_filtered * 1.2
    
    # Aplicar velocidades
    wb_motor_set_velocity(left_motor, left_speed)
    wb_motor_set_velocity(right_motor, right_speed)
```

---

## 8. Validação em Cenário Desconhecido (O Diferencial)

Para provar que o agente realmente aprendeu o conceito abstrato de "distância ao obstáculo" (e não apenas decorou o mapa de treino), teste em um ambiente completamente diferente.

- **Treino:** Arena retangular simples (ex: 2m x 2m).
- **Teste:** Mapa em formato de "L", corredor estreito, ou labirinto com obstáculos internos.

Se a MLP generalizou bem, o robô deverá navegar no novo mapa sem colisões, utilizando apenas as leituras dos sensores infravermelhos.

---

## 9. Resumo do Fluxo de Trabalho (Checklist)

| Etapa | Ação |
| :--- | :--- |
| 1. Setup | Aumentar lookupTable dos IRs no Webots para ~20cm. |
| 2. Coleta | Gerar grade de posições (com viés para paredes), teleportar o robô, girar em 8 ângulos, salvar sensores + distância real. |
| 3. Pré-processamento | Normalizar sensores (0~1). Calcular `cost = exp(-5 * distance)`. Salvar CSV. |
| 4. Treino | MLP com 8 entradas, 32/16 neurônios (ReLU), Dropout(0.2), saída Linear. Loss = Huber/MSE. |
| 5. Navegação | Implementar rotação virtual (interpolação) para testar ângulos sem mover o robô. |
| 6. Controle | Aplicar filtro passa-baixa no ângulo de direção para suavizar a trajetória. |
| 7. Validação | Executar o robô em um mapa totalmente novo para validar a generalização. |

---

## 10. Conclusão

A abordagem de regredir uma função de custo contínua com uma MLP e utilizar um otimizador por amostragem de direções (giro virtual) transforma o robô em um agente verdadeiramente autônomo. Diferente de classificadores que engessam o comportamento em ações discretas, este método permite:

- Movimentos fluidos e naturais.
- Robustez a ruídos dos sensores.
- Generalização para ambientes nunca vistos, pois a MLP aprende a métrica subjacente de "proximidade física".