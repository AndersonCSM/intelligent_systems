# Classificação de Patologias em Folhas utilizando Visão Computacional, Transfer Learning e Explicabilidade (Grad-CAM)

Este projeto foi desenvolvido como parte da disciplina de **Redes Neurais Artificiais**, na Universidade Federal Rural do Semi-Árido (UFERSA).

---

## Introdução e Contexto

A agricultura de precisão e o monitoramento automatizado de lavouras desempenham um papel vital na segurança alimentar global e na redução do desperdício de defensivos agrícolas. Na fruticultura, especificamente na cultura da maçã (*Malus domestica*), doenças foliares fúngicas como a **ferrugem** (*cedar apple rust*) e a **sarna** (*apple scab*) são responsáveis por perdas drásticas de produtividade se não identificadas em estágio inicial.

O desafio reside na complexidade visual das imagens em ambiente aberto: variações severas de iluminação natural, sombras, folhas sobrepostas e a ocorrência simultânea de múltiplas patologias na mesma folha tornam o diagnóstico visual humano subjetivo e demorado.

Neste projeto, desenvolveu-se uma solução computacional de ponta a ponta que unifica técnicas clássicas de **Visão Computacional (OpenCV)** para realce morfológico, pipelines de alta performance assíncronos em **`tf.data`**, **Transfer Learning** sobre múltiplos backbones convolucionais modernos, otimização por **Grid Search**, fusão em **Ensemble** e auditoria de decisões com **Explainable AI (Grad-CAM)**.

---

## Objetivo

Desenvolver, comparar e auditar modelos de Redes Neurais Convolucionais profundas capazes de diagnosticar o estado fitopatológico de folhas de macieira com alta acurácia e generalização, garantindo:

1. **Preservação de Rigor Anti-Vazamento:** Divisão estratificada de treino e validação prévia a qualquer transformação.
2. **Avaliação do Pré-processamento Clássico:** Teste do realce adaptativo de contraste local (**CLAHE**) no espaço HSV.
3. **Benchmarking Sistemático (*Grid Search*):** Comparação entre diferentes famílias arquiteturais (`DenseNet121`, `EfficientNetB0`, `ResNet50V2`, `MobileNetV2`).
4. **Maximização de Generalização (*Ensemble*):** Combinação das predições das melhores redes para anular variâncias individuais.
5. **Auditoria Visual (*XAI*):** Mapeamento térmico via **Grad-CAM** para certificar que a rede toma decisões fundamentadas nas lesões das folhas e não em ruídos de fundo.

---

## Tipo do Problema

O problema é de **Classificação Multiclasse** em 4 classes mutuamente exclusivas:
- `healthy` (Folha saudável)
- `multiple_diseases` (Folha com presença simultânea de múltiplas patologias)
- `rust` (Folha infectada com ferrugem)
- `scab` (Folha infectada com sarna)

A camada de saída das redes utiliza **4 neurônios com ativação Softmax**, otimizados pela função de perda **Categorical Crossentropy**.

---

## Dataset

Foi utilizado o conjunto de dados oficial da competição internacional **Plant Pathology 2020 – FGVC7** (*Fine-Grained Visual Categorization*), organizada no âmbito da conferência **CVPR 2020** e hospedada no Kaggle.

🔗 [Plant Pathology 2020 - FGVC7 (Kaggle)](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)

### Distribuição Amostral:
* **Conjunto de Treino Anotado:** 1.821 imagens em alta resolução.
* **Conjunto de Teste para Submissão:** 1.821 imagens não rotuladas.

| Classe | Frequência Aproximada | Proporção (%) | Característica Visual |
| :--- | :---: | :---: | :--- |
| **rust** | ~622 | 34.2% | Pústulas e manchas amarelo-alaranjadas |
| **scab** | ~592 | 32.5% | Manchas necróticas marrom-escuras / aveludadas |
| **healthy** | ~516 | 28.3% | Superfície verde uniforme sem lesões |
| **multiple_diseases** | **~91** | **5.0%** | Coocorrência de ferrugem e sarna *(classe minoritária)* |

> **Nota Metodológica sobre o Desbalanceamento:** A extrema raridade da classe `multiple_diseases` (~5%) exige mecanismos ativos como divisão estratificada (`StratifiedKFold`), pesos de classe (`class_weight`) e monitoramento do **Macro F1-Score** e de matrizes de confusão normalizadas para evitar que a rede se torne cega à classe minoritária.

---

## Metodologia e Pipeline de Dados

```
[Imagens Brutas]
       ↓
[Pré-processamento OpenCV (CLAHE no canal V - HSV)]
       ↓
[Pipeline tf.data: Leitura assíncrona + Data Augmentation dinâmico + Cache + Prefetch]
       ↓
[Base Convolucional Pré-treinada (ImageNet congelada)]
       ↓
[GlobalAveragePooling2D] → [Dropout(0.3)] → [Dense(4, Softmax)]
       ↓
[Treinamento com EarlyStopping e ReduceLROnPlateau]
       ↓
[Grid Search & Fine-Tuning dos Campeões]
       ↓
[Ensemble de Predições] → [Geração de submission.csv]
       ↓
[Explicabilidade Visual com Grad-CAM]
```

### 1. Pré-processamento com OpenCV
* **Análise Morfológica e Filtros (na EDA):** `cv2.GaussianBlur` para atenuação de ruído e `cv2.Canny` para destacar bordas das pústulas.
* **Realce de Contraste Local (CLAHE):** Aplicação de *Contrast Limited Adaptive Histogram Equalization* no canal de luminância (V) do espaço HSV, convertendo de volta para RGB. Essa abordagem mantém a imagem com **3 canais (RGB)** compatíveis com pesos ImageNet e realça lesões de baixo contraste sob sombras naturais.

### 2. Pipeline Assíncrono com `tf.data`
* **Leitura sob Demanda:** Redimensionamento dinâmico para tamanho fixo ($224 \times 224$ ou superior) via decodificação eficiente `tf.io.read_file`.
* **Data Augmentation Dinâmico em Lote:** Espelhamentos horizontais e verticais aleatórios, pequenas rotações e variações de iluminação para multiplicar artificialmente o volume de amostras de doenças raras.
* **Escalabilidade Dinâmica de Batch:** 
  $$\text{BATCH\_SIZE} = \text{PER\_REPLICA\_BATCH} \times \text{strategy.num\_replicas\_in_sync}$$
  Otimizado para saturação da GPU (Tesla T4 no Google Colab) sem estouro de memória, com suporte nativo a `tf.data.AUTOTUNE`.

---

## Arquiteturas Avaliadas e Modelagem

Os backbones foram selecionados estrategicamente para cobrir as principais famílias da visão computacional moderna (disponibilizados em [pre_training.py](pre_training.py)):

| Família Arquitetural | Modelo Avaliado | Diferencial Técnico |
| :--- | :--- | :--- |
| **Conexões Densas** | `DenseNet121` | Reutilização intensiva de mapas de características e mitigação do desvanecimento de gradientes |
| **Escalonamento Composto** | `EfficientNetB0` | Equilíbrio ótimo entre profundidade, largura e resolução de imagem |
| **Conexões Residuais V2** | `ResNet50V2` | Pré-ativação nas camadas de salto para fluxo mais direto de gradientes |
| **Computação Compacta** | `MobileNetV2` | Convoluções separáveis em profundidade com gargalos invertidos lineares |

### Cabeçalho de Classificação Enxuto (Head):
```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)
```

### Mecanismos de Treinamento e Callbacks:
* **`EarlyStopping`:** Monitorando `val_loss` (patience = 6 a 10 épocas, `restore_best_weights=True`) para interromper no ponto exato de máxima generalização.
* **`ReduceLROnPlateau`:** Monitorando `val_loss` (`factor=0.2`, `patience=2-3`, `min_lr=1e-6`) para desacelerar o passo do otimizador Adam ao atingir platôs.
* **`ModelCheckpoint`:** Salvamento automático dos pesos `.keras` que atingirem o maior `val_roc_auc`.

---

## Espaço de Busca (*Grid Search*) e Ensemble

O arquivo `03_grid_search_ensemble.ipynb` executa uma busca sistemática explorando o seguinte espaço de hiperparâmetros:

| Hiperparâmetro | Faixa de Valores | Justificativa |
| :--- | :---: | :--- |
| **Backbones** | `DenseNet121`, `EfficientNetB0`, `ResNet50V2`, `MobileNetV2` | Comparação entre famílias de conectividade distinta |
| **Learning Rate Inicial** | `[1e-3, 5e-4, 1e-4]` | Ponto de partida ótimo para o otimizador Adam |
| **Épocas Máximas** | `[100, 200, 400]` | Teto seguro com parada antecipada por EarlyStopping |
| **Validation Split** | `[0.15, 0.20, 0.25]` | Avaliação da sensibilidade ao volume de treino/validação |
| **Dropout** | `[0.0, 0.1, 0.3]` | Avaliação da regularização contra sobreajuste |
| **Batch Size (base)** | `[32, 64]` | Maximização do throughput da GPU no Google Colab |

### Construção do Ensemble Final:
As probabilidades dos 2 ou 3 modelos de topo do grid são combinadas:
$$P_{\text{ensemble}} = \sum_{i=1}^{K} w_i \cdot P_i \quad \text{onde} \quad \sum w_i = 1$$

O Ensemble final gera o arquivo oficial [data/submission.csv](data/submission.csv) contendo as probabilidades para o conjunto de teste de 1.821 imagens.

---

## Inteligência Artificial Explicável (XAI - Grad-CAM)

Para validar a confiabilidade do diagnóstico automatizado, o notebook `04_xai_gradcam.ipynb` implementa o algoritmo **Grad-CAM (Gradient-weighted Class Activation Mapping)**:

1. **Cálculo de Gradientes:** Extrai-se o gradiente da pontuação da classe predita em relação aos mapas de ativação da última camada convolucional do modelo campeão.
2. **Ponderação e ReLU:** Aplica-se média global dos gradientes para obter a importância de cada canal, seguida de retificação ReLU para evidenciar apenas as feições com contribuição positiva para a classe.
3. **Sobreposição de Mapa de Calor (*Heatmap*):** Gera-se um mapa colorido normalizado sobreposto à fotografia original da folha.

**Auditoria Qualitativa:**
* Verifica-se se o modelo focou nas manchas alaranjadas de ferrugem ou lesões de sarna.
* Confirma-se que a rede ignorou o fundo (solo, hastes metálicas ou dedos do operador) durante a tomada de decisão.

---

## Estrutura do Repositório

```
neural-networks/unidade-2/projeto/
│
├── data/                               # Diretório de dados (Kaggle Plant Pathology 2020)
│   ├── images/                         # Imagens em alta resolução (.jpg)
│   ├── train.csv                       # Metadados e rótulos de treino
│   ├── test.csv                        # Lista de IDs para teste
│   ├── sample_submission.csv           # Formato de submissão da competição
│   └── submission.csv                  # Submissão gerada pelo Ensemble final
│
├── graphs/                             # Figuras e gráficos em alta resolução (300 DPI)
│   ├── loss_curve_{modelo}.png         # Curvas de perda treino vs validação
│   ├── accuracy_auc_curve_{modelo}.png # Curvas de acurácia e ROC-AUC
│   ├── confusion_matrix_{modelo}.png   # Matrizes absoluta e normalizada
│   ├── roc_curve_multiclass.png        # Curvas ROC One-vs-Rest por classe
│   └── models_benchmark_comparison.png # Gráfico comparativo geral
│
├── models/                             # Checkpoints e históricos de treinamento
│   ├── best_{modelo}.keras             # Pesos ótimos salvos pelo ModelCheckpoint
│   ├── history_{modelo}.csv            # Histórico época a época de cada execução
│   ├── grid_search_summary.csv         # Tabela consolidada de métricas do grid
│   └── ensemble_meta.json              # Configurações e pesos do Ensemble
│
├── references/                         # Cadernos de referência do Kaggle
│   ├── image-pre-processing-and-densenet.ipynb
│   └── plant-pathology-2020-eda-models.ipynb
│
├── 01_eda.ipynb                        # Etapa 1: Análise Exploratória e testes OpenCV
├── 02_pipeline_single.ipynb            # Etapa 2: Prova de Conceito e Baseline Single
├── 03_grid_search_ensemble.ipynb       # Etapa 3: Grid Search, Fine-Tuning e Ensemble
├── 04_xai_gradcam.ipynb                # Etapa 4: Explicabilidade com Grad-CAM
│
├── pre_training.py                     # Módulo Python com o catálogo de modelos Keras
├── strategy.md                         # Documento de planejamento e decisões técnicas
└── README.md                           # Documentação oficial do projeto
```

---

## Tecnologias e Bibliotecas Utilizadas

- **Linguagem:** Python (v3.11+)
- **Deep Learning:** TensorFlow 2.x / Keras (com `tf.keras.applications` e `tf.data`)
- **Visão Computacional Clássica:** OpenCV (`cv2`)
- **Manipulação de Dados:** Pandas, NumPy
- **Métricas e Pré-processamento:** Scikit-Learn
- **Visualização Científica:** Matplotlib, Seaborn, Plotly
- **Ambiente de Aceleração:** Google Colaboratory (GPU Tesla T4 / A100)

---

## Como Executar o Projeto

### 1. Instalação de Dependências
No seu ambiente virtual ou terminal:
```bash
pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib seaborn plotly
```

### 2. Execução no Google Colab
Abra os notebooks desejados no Google Colab, certifique-se de selecionar o runtime com **GPU** (*Ambiente de execução $\to$ Alterar tipo de ambiente de execução $\to$ GPU T4*) e monte o Google Drive na primeira célula:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Sequência Recomendada de Execução
1. **`01_eda.ipynb`:** Inspeção da distribuição de classes, canais de cor e demonstração visual de filtros.
2. **`02_pipeline_single.ipynb`:** Validação do fluxo ponta a ponta com o baseline da `DenseNet121`.
3. **`03_grid_search_ensemble.ipynb`:** Execução da busca sistemática de hiperparâmetros, exportação de tabelas/gráficos e fusão em Ensemble.
4. **`04_xai_gradcam.ipynb`:** Carregamento do modelo campeão e geração dos mapas de calor com Grad-CAM.

---

## Referências e Créditos

Este trabalho foi fundamentado e estruturado a partir das técnicas e boas práticas desenvolvidas em dois projetos de referência da comunidade para a competição *Plant Pathology 2020 – FGVC7*:

1. **[Image pre processing and DenseNet](references/image-pre-processing-and-densenet.ipynb)**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de análise visual e filtragem com OpenCV, incluindo filtros de suavização (`cv2.GaussianBlur`), detecção de bordas morfológicas (`cv2.Canny`) e a aplicação de realce adaptativo de contraste local (**CLAHE**) no espaço de cor HSV como técnica de pré-processamento aliada à `DenseNet121`.

2. **[Plant Pathology 2020 - EDA, Models](references/plant-pathology-2020-eda-models.ipynb)**
   * **Plataforma / Contexto:** Kaggle (*Plant Pathology 2020 - FGVC7*)
   * **Contribuição aproveitada no projeto:** Metodologia de Análise Exploratória (EDA) aprofundada com distribuição estatística dos canais de cor RGB, pipeline assíncrono de alto rendimento em `tf.data.Dataset`, estratégias de *Transfer Learning* sobre múltiplos backbones modernos (`DenseNet`, `EfficientNet`, `NoisyStudent`) e a arquitetura de fusão de probabilidades em *Ensemble*.

3. **Competição Oficial e Dataset:**
   * Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). *Plant Pathology 2020 – FGVC7 Challenge Dataset*. CVPR 2020 Workshop on Fine-Grained Visual Categorization. [Kaggle Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).

4. **Metodologia de Explicabilidade (Grad-CAM):**
   * Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV), pp. 618-626.

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

---

## Licença

Este projeto está disponibilizado sob a **Licença MIT**.  
O dataset *Plant Pathology 2020 – FGVC7* é de propriedade do concurso CVPR 2020 e está sujeito aos termos de uso da competição Kaggle.
