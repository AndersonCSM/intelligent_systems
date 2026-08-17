## 3. Avaliação em Classificação Binária

No processo de classificação binária, devemos treinar um modelo para separar um conjunto de dados em duas categorias exclusivas, geralmente representadas como 0 ou 1, **Verdadeiro ou Falso, ou Positivo ou Negativo.**

O modelo recebe características de entrada ($x$) e tenta prever a qual classe ($y$) aquele dado pertence. Para isso, ele geralmente calcula uma probabilidade. Se a probabilidade for maior que um determinado limiar (comumente 0.5), o dado é classificado como Positivo; caso contrário, Negativo.

### 3.1. Matriz de Confusão

A Matriz de Confusão é uma ferramenta de visualização de desempenho para algoritmos de classificação (especialmente binária). Em um cenário de classificação binária (onde temos as classes Positivo e Negativo), a matriz é uma tabela $2 \times 2$:

<table style="width:60%; text-align:center; border: 2px solid #808080; border-collapse: collapse">
  <tr>
    <th rowspan="2">Real</th>
    <th colspan="2">Predito</th>
  </tr>
  <tr>
    <th>Positivo</th>
    <th>Negativo</th>
  </tr>
  <tr>
    <td><b>Positivo</b></td>
    <td style="background-color: #e6ffed; color: black">Verdadeiro Positivo (VP)</td>
    <td style="background-color: #ffeef0; color: black">Falso Negativo (FN)</td>
  </tr>
  <tr>
    <td><b>Negativo</b></td>
    <td style="background-color: #ffeef0; color: black">Falso Positivo (FP)</td>
    <td style="background-color: #e6ffed; color: black">Verdadeiro Negativo (VN)</td>
  </tr>
</table>

Neste contexto, define-se:

* **Verdadeiro Positivo (VP):** O valor real era positivo e o modelo previu positivo. (Acerto)
* **Verdadeiro Negativo (VN):** O valor real era negativo e o modelo previu negativo. (Acerto)
* **Falso Positivo (FP):** O valor real era negativo, mas o modelo previu positivo. (Erro Tipo I - "Alarme Falso")
* **Falso Negativo (FN):** O valor real era positivo, mas o modelo previu negativo. (Erro Tipo II - "Omissão")

#### Acurácia

Indica a performance geral do modelo, ou seja, quanto ele acertou do total.

$$
\textrm{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}
$$

#### Precisão

Mede a proporção de verdadeiros positivos entre todas as predições positivas. Responde à pergunta: "De todos que o modelo classificou como Positivo, quantos eram realmente Positivos?". Penaliza os falsos positivos. Use quando um alarme falso é caro — por exemplo, filtrar e-mails legítimos como spam.

$$
\textrm{Precisão} = \frac{VP}{VP + FP}
$$

#### Recall

Mede a proporção de verdadeiros positivos e dados realmente positivos. Responde à pergunta: "De todos que eram realmente Positivos, quantos o modelo conseguiu detectar?". Penaliza os falsos negativos. Use quando deixar escapar um positivo é perigoso — por exemplo, não detectar um tumor maligno.

$$
\textrm{Recall} = \frac{VP}{VP + FN}
$$

#### F-1 Score

É a média harmônica entre Precisão e Recall. Ela é muito útil quando você precisa de um equilíbrio entre as duas métricas e tem classes desbalanceadas.

$$
F_1 = 2 \frac{\textrm{Precisão} \cdot \textrm{Recall}}{\textrm{Precisão} + \textrm{Recall}}
$$

### 3.2. Curva ROC

**A curva ROC (do inglês *Receiver Operating Characteristic*, ou Característica de Operação do Receptor)** é uma ferramenta gráfica essencial para avaliar o desempenho de um modelo de classificação binária.

A curva é plotada em um plano bidimensional definido por dois índices fundamentais:
* Eixo Y: Recall.
* Eixo X: Representa a proporção de casos negativos que foram incorretamente classificados como positivos.

$$
TFP = \frac{FP}{FP + VN}
$$

Em um classificador perfeito, a curva passaria pelo canto superior esquerdo (coordenada 0,1), onde a sensibilidade é 100% e a taxa de falsos positivos é 0%.

Para resumir a curva ROC em um único valor numérico, utilizamos o AUC (*Area Under the Curve*). Quanto mais próximo de 1,0 for o valor do AUC, melhor é o nosso classificador.