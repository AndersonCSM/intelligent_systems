# Estratégia do Projeto - Classificação de Patologias em Folhas (Plant Pathology 2020)

Para garantir uma entrega legível, modular e de padrão acadêmico/Kaggle, o projeto foi estruturado em quatro etapas complementares. A solução unifica as técnicas clássicas de visão computacional (OpenCV) para realce de características com a alta performance e escalabilidade de pipelines modernos em `tf.data`, *Transfer Learning*, *ReduceLROnPlateau*, *Grid Search*, *Ensemble* e Explicabilidade (*XAI*).

---

## Estrutura de Arquivos e Módulos

| Etapa | Arquivo / Módulo | Objetivo Principal |
| :--- | :--- | :--- |
| **01** | `01_eda.ipynb` | Análise exploratória visual, canais de cor, balanceamento e inspeção de filtros |
| **02** | `02_pipeline_single.ipynb` <br> `02_pipeline_improved.ipynb` <br> `02_pipeline_improved_v2.ipynb` | PoC e Estudos de Ablação: Baseline natural vs. CLAHE (V1) vs. Segmentação + Balanço de Cores (V2) |
| **Ref** | `grid_search.md` | Documentação formal da busca em grade (espaço cartesiano de 648 combinações e análise metodológica) |
| **03** | `03_benchmark_models.ipynb` | Benchmarking sistemático dos backbones selecionados (`DenseNet`, `EfficientNetV2`, `ResNet`, `ConvNeXt`, `Xception`) |
| **04** | `04_ensemble.ipynb` | Construção, ponderação (ROC-AUC) e validação do Ensemble final + Geração de `submission.csv` |
| **05** | `05_xai_gradcam.ipynb` | Explicabilidade visual com Grad-CAM / Grad-CAM++ sobre os melhores modelos |

---

## 1. Análise Exploratória de Dados (EDA) - `01_eda.ipynb`

- **Visualização Amostral:** Inspeção lado a lado de folhas saudáveis (*healthy*) vs. infectadas (*rust*, *scab*, *multiple_diseases*).
- **Análise dos Canais de Cor:** Distribuição estatística das intensidades nos canais RGB e HSV por classe.
- **Diagnóstico de Desbalanceamento:**
  - Gráficos de barras, pizza e *parallel coordinates* identificando a raridade da classe minoritária `multiple_diseases` (~5% do total).
  - Definição da estratégia de mitigação (divisão estratificada e pesos de classe `class_weight`).
- **Inspeção de Pré-processamento e Aumento de Dados:**
  - Demonstração visual de filtros OpenCV: `cv2.GaussianBlur` (suavização de ruído), `cv2.Canny` (detecção de bordas das lesões) e `CLAHE` (realce adaptativo de contraste local).
  - Demonstração visual de transformações: rotações, *flip* horizontal/vertical, variações de brilho e zoom.

---

## 2. Pipeline Single (Baseline / PoC) - `02_pipeline_single.ipynb`

*Objetivo:* Estabelecer um fluxo fim a fim funcional e reprodutível, validando o pipeline de dados, estabilidade de gradientes e regularizações antes dos treinamentos pesados.

- **Importação e Carregamento:** Leitura dos metadados (`train.csv`), verificação de integridade e caminhos de imagens.
- **Divisão Estratificada (Train / Validation):** Garantia de proporção idêntica de classes em ambos os conjuntos (`train_test_split(..., stratify=y)`).
- **Pré-processamento com OpenCV (Imagens 3 Canais):**
  - Aplicação de **CLAHE** no espaço de cor HSV (canal V) para acentuar o contraste das manchas e lesões sem perder a informação de cor (mantendo 3 canais RGB para os backbones pré-treinados).
  - *Nota:* O filtro `cv2.Canny` permanece como análise morfológica na EDA, pois modelos ImageNet exigem tensores RGB.
- **Pipeline de Dados Eficiente (`tf.data.Dataset`):**
  - Carregamento sob demanda com `tf.io.read_file` e decodificação rápida.
  - *Data Augmentation* dinâmico em lote no GPU/CPU.
  - **Batch Size Dinâmico:** Definido como `BATCH_SIZE = PER_REPLICA_BATCH * strategy.num_replicas_in_sync` (ajustando dinamicamente o lote para Colab GPU ou TPU).
  - Encadeamento otimizado: `.shuffle(buffer_size)`, `.batch(BATCH_SIZE)`, `.repeat()` e `.prefetch(tf.data.AUTOTUNE)`.
- **Arquitetura Baseline com Regularização:**
  - Carga de 1 modelo pré-treinado (ex: `DenseNet121` com pesos `imagenet`, base congelada).
  - **Topo de Classificação Customizado:**
    - `GlobalAveragePooling2D()`
    - `Dropout(0.3)` (regularização essencial contra overfitting)
    - `Dense(4, activation='softmax')`
- **Mecanismos de Treinamento e Callbacks:**
  - **`EarlyStopping`:** Monitorando `val_loss` (patience = 7 a 10 épocas, `restore_best_weights=True`) para interromper o treino no ponto ideal.
  - **`ReduceLROnPlateau`:** Monitorando `val_loss` (`factor=0.2`, `patience=3`, `min_lr=1e-6`) para reduzir adaptativamente a taxa de aprendizado quando o ganho estagnar.
  - **`class_weight`:** Incorporado no `model.fit()` para compensar o desbalanceamento das classes.
- **Métricas e Avaliação:**
  - Curvas de aprendizado (*Loss*, *Categorical Accuracy* e *ROC-AUC* por época).
  - Matriz de confusão e relatório de classificação (*Precision*, *Recall*, *F1-Score* ponderado e macro).

---

## 3. Benchmarking de Arquiteturas Pré-Treinadas - `03_benchmark_models.ipynb`

*Objetivo:* Avaliar sistematicamente famílias arquiteturais pré-treinadas distintas sob um regime unificado de hiperparâmetros calibrados na PoC, comparando capacidade representacional, tempo de treinamento e métricas de validação.

> **Nota sobre o Grid Search:** A busca exaustiva de hiperparâmetros (648 combinações) foi teoricamente modelada e devidamente documentada no arquivo independente **[grid_search.md](file:///home/anderson/github_projects/intelligent_systems/neural-networks/unidade-2/projeto/grid_search.md)**. Conforme justificado metodologicamente, como o baseline da PoC já atingiu convergência de excelência (>95% acurácia e >0.97 ROC-AUC), os hiperparâmetros ótimos foram fixados para priorizar a diversidade arquitetural.

### 3.1 Matriz de Triagem Arquitetural (Análise e Justificativa de Descarte)

Dentre as 9 famílias de arquiteturas disponíveis no Keras Applications, selecionou-se **apenas uma versão representativa** por família para análise e triagem técnica fundamentada:

| Família | Versão Avaliada | Decisão | Justificativa Técnica para Descarte / Seleção |
| :--- | :--- | :---: | :--- |
| **DenseNet** | `DenseNet121` | **JÁ EXECUTADO** | Conexões densas concatenam mapas de todas as camadas. Validada na PoC com excelente convergência e baixo consumo de parâmetros. **Componente garantido no Ensemble**. |
| **VGG** | `VGG16` | **DESCARTAR** | Arquitetura ultrapassada (2014) com ~138M de parâmetros. Sem conexões residuais (*skip connections*); em 512x512 consome excessiva VRAM e sofre degradação de gradiente. Desempenho inferior a modelos modernos com 10% do tamanho. |
| **NASNet** | `NASNetMobile` | **DESCARTAR** | Células convolucionais heterogêneas geradas por busca automática (RL). Pouco otimizada para execução em GPU (Tensor Cores / CUDA), resultando em treino lento por época e comportamento instável com precisão mista. |
| **Inception** | `InceptionV3` | **DESCARTAR** | Os blocos multi-escala clássicos (1x1, 3x3, 5x5) foram historicamente superados pela **Xception**, que substituiu os módulos por convoluções separáveis em profundidade. Manter ambos seria redundância teórica. |
| **MobileNet** | `MobileNetV2` | **DESCARTAR** | Projetada estritamente para *Edge Computing* / dispositivos móveis (blocos residuais invertidos). Embora rápida, sua capacidade representacional para distinguir microlesões e pústulas foliares sutis em 512x512 é limitada em comparação às redes completas. |
| **EfficientNetV2** | `EfficientNetV2B0` | **SELECIONADA** | Evolução da V1 descartada. Substitui as primeiras camadas por **Fused-MBConv**, acelerando o treino na GPU, além de incorporar atenção de canal (*Squeeze-and-Excitation*). Alta complementaridade com a DenseNet. |
| **ResNet** | `ResNet50V2` | **SELECIONADA** | Padrão-ouro em conexões residuais aditivas ($x + \mathcal{F}(x)$). A versão V2 utiliza pré-ativação (BN e ReLU antes das convoluções), garantindo fluxo direto de gradientes. Erros ortogonais aos da DenseNet (ideal para Ensemble). |
| **ConvNeXt** | `ConvNeXtTiny` | **SELECIONADA** | Representa o estado da arte convolucional (2022). Incorpora conceitos de *Vision Transformers* (kernels grandes 7x7, ativação GELU, LayerNorm), capturando contexto espacial amplo das folhas de macieira. |
| **Xception** | `Xception` | **SELECIONADA** | Convoluções separáveis em profundidade com desacoplamento espacial/canal. Excelente capacidade de extração com parametrização eficiente. |

### 3.2 Protocolo de Execução do Benchmarking

1. **Modelos a Treinar na Etapa 3:**
   - `EfficientNetV2B0`
   - `ResNet50V2`
   - `ConvNeXtTiny`
   - `Xception`
   *(Nota: `DenseNet121` já está treinada com pesos e métricas persistidos na PoC).*
2. **Condições Unificadas:**
   - Resolução: `512x512` | Batch Size: `16` | Taxa inicial: `1e-4` (Adam) | Precisão: `mixed_float16`.
   - Callbacks: `EarlyStopping(patience=8)`, `ReduceLROnPlateau(patience=3, factor=0.2)`, `ModelCheckpoint`.
3. **Quadro Consolidado de Desempenho:**
   - Tabela comparativa com: `Model_Name`, `Parâmetros`, `Tempo/Época`, `Val_Loss`, `Val_Accuracy`, `Macro_ROC_AUC` e `Macro_F1`.

---

## 4. Ensemble de Modelos (Fusão e Ponderação de Predições) - `04_ensemble.ipynb`

*Objetivo:* Combinar as probabilidades previstas pelos múltiplos modelos de famílias distintas treinados no benchmarking, explorando a decorrelação de erros para atingir generalização superior a qualquer classificador individual (requisito central do projeto).

### 4.1 Estratégias de Combinação

1. **Média Simples (*Soft Voting*):**
   $$\hat{y}_i = \frac{1}{M} \sum_{m=1}^{M} P_m(y_i)$$
2. **Média Ponderada por Desempenho (*Weighted Blending* por ROC-AUC):**
   $$w_m = \frac{\text{AUC}_m - 0.5}{\sum_{k=1}^M (\text{AUC}_k - 0.5)}, \quad \hat{y}_i = \sum_{m=1}^{M} w_m P_m(y_i)$$
   - Atribui peso proporcionalmente maior aos modelos com maior capacidade de ordenação probabilística no conjunto de validação.

### 4.2 Métricas de Ganho do Ensemble
- Comparação direta: Ganho delta ($\Delta \text{AUC}$, $\Delta \text{F1}$, $\Delta \text{Acc}$) do Ensemble frente ao melhor modelo individual.
- Matriz de confusão combinada e curva ROC do Ensemble.

### 4.3 Geração da Submissão Oficial
- Inferência do Ensemble sobre as 1.821 imagens não rotuladas de `data/test.csv`.
- Exportação do arquivo final `data/submission.csv` no padrão da competição Kaggle (`[image_id, healthy, multiple_diseases, rust, scab]`).

---

## 5. Explicabilidade (XAI) - `05_xai_gradcam.ipynb`

*Objetivo:* Interpretar visualmente como as redes tomam as decisões, auditando se as ativações convolucionais coincidem de fato com as lesões reais das folhas e não com ruídos de fundo.

- **Carregamento dos Melhores Modelos:** Checkpoint campeão individual e modelos do Ensemble.
- **Implementação do Grad-CAM / Grad-CAM++:**
  - Extração de gradientes da classe predita em relação aos mapas de ativação da última camada convolucional.
  - Ponderação dos mapas de características com retificação positiva (ReLU).
- **Geração e Sobreposição de Heatmaps:**
  - Mapas de calor sobrepostos às fotos originais das folhas em alta resolução.
- **Auditoria Qualitativa por Classe:**
  - Inspeção visual das regiões ativadas para `healthy`, `rust`, `scab` e `multiple_diseases`.
  - Diagnóstico de eventuais casos limítrofes ou erros de predição.

---

## 6. Template e Padronização de Exportação de Artefatos (Artigo, Avaliação e Reprodutibilidade)

Para garantir que todos os elementos necessários para a escrita do artigo/relatório e para a avaliação das bancas estejam prontos e organizados, o pipeline salvará automaticamente:

#### 1. Pesos e Modelos Compilados (`models/`)
* `models/best_{model_name}.keras`: Checkpoint com os melhores pesos de cada backbone salvo pelo `ModelCheckpoint` (via `val_roc_auc`).
* `models/ensemble_meta.json`: Metadados, hiperparâmetros ótimos e pesos da ponderação atribuída a cada modelo no Ensemble.

#### 2. Dados e Relatórios Tabulares (`models/`)
* `models/grid_search_summary.csv`: Tabela consolidada com todas as combinações executadas (`model_name`, `initial_lr`, `val_split`, `dropout`, `epochs_trained`, `val_loss`, `val_accuracy`, `val_roc_auc`, `val_macro_f1`, `tempo_total_s`).
* `models/history_{model_name}.csv`: Histórico época a época de treino e validação (`loss`, `val_loss`, `categorical_accuracy`, `val_categorical_accuracy`, `roc_auc`, `val_roc_auc`, `lr`).
* `models/classification_report_{model_name}.csv`: Precisão, recall e F1-Score detalhados por classe (`healthy`, `multiple_diseases`, `rust`, `scab`).

#### 3. Figuras em Alta Resolução (`graphs/` a 300 DPI)
* `graphs/loss_curve_{model_name}.png`: Curva de Loss de Treino vs. Validação identificando o ponto de atuação do *EarlyStopping*.
* `graphs/accuracy_auc_curve_{model_name}.png`: Evolução conjunta da Acurácia e do ROC-AUC.
* `graphs/confusion_matrix_{model_name}.png`: Painel duplo com Matriz de Confusão em contagens brutas e Matriz Normalizada por linha (essencial para auditar o acerto da classe minoritária).
* `graphs/roc_curve_multiclass_{model_name}.png`: Curvas ROC One-vs-Rest para cada uma das 4 classes com as respectivas áreas sob a curva (AUC).
* `graphs/models_benchmark_comparison.png`: Gráfico de barras consolidado comparando ROC-AUC, Acurácia e Macro F1 entre todos os modelos do grid e o Ensemble final.

#### 4. Submissão Oficial
* `data/submission.csv`: Arquivo final com IDs de teste e as 4 probabilidades previstas pelo Ensemble.

---

## Decisões Técnicas e Boas Práticas Adotadas

1. **Garantia Anti-Vazamento:** Divisão de treino/validação antes de qualquer cálculo estatístico ou aumento de dados.
2. **Preservação de 3 Canais para Transfer Learning:** Aplicação de filtros que mantêm o formato (H, W, 3) compatível com pesos ImageNet.
3. **Regularização Enxuta:** Uso focado de `Dropout` + `EarlyStopping` na cabeça de classificação, sem camadas desnecessárias de normalização.
4. **Otimização Reativa de Taxa de Aprendizado:** Uso do `ReduceLROnPlateau` para decair a taxa dinamicamente apenas quando a validação estagnar, permitindo que cada arquitetura convirja em seu próprio ritmo.
5. **Métricas Oficiais da Competição:** Uso de **Categorical Crossentropy**, **Acurácia Categórica** e **ROC-AUC (One-vs-Rest)** para avaliação global.
6. **Eficiência Computacional e Batch Dinâmico:** Uso de `tf.data.AUTOTUNE` e escalonamento de batch por `strategy.num_replicas_in_sync`, garantindo saturação total da GPU (Tesla T4 no Colab) ou TPU sem estouro de memória.
7. **Persistência em Nuvem (Google Colab / Drive):** Montagem do Google Drive (`drive.mount('/content/drive')`) para salvar pesos em `models/` e figuras em `graphs/`, protegendo contra desconexões acidentais da sessão do Colab.
