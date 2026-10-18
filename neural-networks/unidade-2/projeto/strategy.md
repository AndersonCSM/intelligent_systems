# Estratégia do Projeto - Classificação de Patologias em Folhas (Plant Pathology 2020)

Para garantir uma entrega legível, modular e de padrão acadêmico/Kaggle, o projeto foi estruturado em quatro etapas complementares. A solução unifica as técnicas clássicas de visão computacional (OpenCV) para realce de características com a alta performance e escalabilidade de pipelines modernos em `tf.data`, *Transfer Learning*, *ReduceLROnPlateau*, *Grid Search*, *Ensemble* e Explicabilidade (*XAI*).

---

## Estrutura de Arquivos e Módulos

| Etapa | Arquivo / Notebook | Objetivo Principal |
| :--- | :--- | :--- |
| **01** | `01_eda.ipynb` | Análise exploratória visual, canais de cor, balanceamento e inspeção de filtros |
| **02** | `02_pipeline_single.ipynb` | Prova de Conceito (PoC), validação do pipeline `tf.data`, callbacks e baseline inicial |
| **03** | `03_grid_search_ensemble.ipynb` | Benchmarking de múltiplos backbones, busca sistemática de hiperparâmetros e Ensemble |
| **04** | `04_xai_gradcam.ipynb` | Explicabilidade visual com Grad-CAM / Grad-CAM++ sobre os melhores modelos |

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

## 3. Grid Search & Ensemble - `03_grid_search_ensemble.ipynb`

*Objetivo:* Comparar sistematicamente diferentes arquiteturas pré-treinadas e explorar um grid de hiperparâmetros para encontrar a combinação de maior generalização, finalizando com a fusão em Ensemble.

### Espaço de Busca (*Hyperparameter Grid*)

| Hiperparâmetro | Valores / Espaço a Variar | Justificativa Técnica |
| :--- | :--- | :--- |
| **Modelos (Backbones)** | `DenseNet121`, `EfficientNetB0`, `ResNet50V2`, `MobileNetV2` | Avaliar famílias distintas (densas, eficientes, residuais e compactas) |
| **`learning_rate` inicial** | `[1e-3, 5e-4, 1e-4]` | Avaliar taxa de partida ideal para convergência do Adam |
| **`epochs`** | `[100, 200, 400]` *(com EarlyStopping ativo)* | Teto seguro de treinamento com interrupção automática |
| **`validation_split`** | `[0.15, 0.20, 0.25]` *(estratificado)* | Sensibilidade do modelo ao volume de dados de treino vs validação |
| **`Dropout`** | `[0.0, 0.1, 0.3]` | Avaliar o grau ótimo de desativação para evitar memorização |
| **`batch_size`** *(base)* | `[32, 64]` | Escalado com `num_replicas_in_sync`, ideal para a GPU Tesla T4 (Colab) mantendo bom número de passos/época |

### Fluxo de Execução do Grid Search:
1. **Iteração Sistemática:** Laço automatizado iterando sobre as combinações definidas no grid.
2. **Callbacks Unificados:**
   - `EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)`
   - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)`
   - `ModelCheckpoint(filepath=..., monitor='val_roc_auc', mode='max', save_best_only=True)`
3. **Quadro Comparativo de Resultados:**
   - Tabela consolidada com `Model_Name`, `Initial_LR`, `Val_Split`, `Dropout`, `Val_Loss`, `Val_Accuracy`, `Val_ROC_AUC` e tempo por época.
4. **Fine-Tuning dos Campeões:**
   - Descongelamento parcial das últimas camadas convolucionais dos 2 melhores modelos com taxa de aprendizado ultra-baixa (`1e-5`).
5. **Construção do Ensemble:**
   - Fusão das predições dos 2 ou 3 modelos de topo através de:
     - Média simples de probabilidades.
     - Média ponderada baseada no ROC-AUC individual de validação.
   - Avaliação do ganho de métricas do Ensemble frente a qualquer modelo individual.
6. **Geração do Arquivo de Submissão (`submission.csv`):**
   - Inferência com o Ensemble sobre as 1.821 imagens não rotuladas de `test.csv`.
   - Exportação de `submission.csv` no padrão oficial com as colunas `[image_id, healthy, multiple_diseases, rust, scab]`.

### Template e Padronização de Exportação de Artefatos (Artigo, Avaliação e Reprodutibilidade)

Para garantir que todos os elementos necessários para a escrita do artigo/relatório e para a avaliação das bancas estejam prontos e organizados, o Grid Search salvará automaticamente:

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

## 4. Explicabilidade (XAI) - `04_xai_gradcam.ipynb`

*Objetivo:* Interpretar visualmente como a rede toma as decisões, garantindo que o modelo está de fato olhando para as lesões das folhas e não para o fundo.

- **Carregamento do Melhor Modelo:** Restauração do checkpoint campeão do Grid Search / Fine-Tuning.
- **Implementação do Grad-CAM / Grad-CAM++:**
  - Extração dos gradientes da classe de maior probabilidade em relação aos mapas de ativação da última camada convolucional.
  - Ponderação dos mapas de características e aplicação da função ReLU para destacar apenas influências positivas.
- **Geração de Mapas de Calor (Heatmaps):**
  - Sobreposição do mapa de calor colorido sobre a imagem original da folha.
- **Análise Qualitativa por Classe:**
  - Avaliar se as regiões destacadas pelo Grad-CAM coincidem com as pústulas de ferrugem (*rust*) e manchas necróticas de sarna (*scab*).
  - Diagnóstico de eventuais erros de classificação (falsos positivos/negativos).

---

## Decisões Técnicas e Boas Práticas Adotadas

1. **Garantia Anti-Vazamento:** Divisão de treino/validação antes de qualquer cálculo estatístico ou aumento de dados.
2. **Preservação de 3 Canais para Transfer Learning:** Aplicação de filtros que mantêm o formato (H, W, 3) compatível com pesos ImageNet.
3. **Regularização Enxuta:** Uso focado de `Dropout` + `EarlyStopping` na cabeça de classificação, sem camadas desnecessárias de normalização.
4. **Otimização Reativa de Taxa de Aprendizado:** Uso do `ReduceLROnPlateau` para decair a taxa dinamicamente apenas quando a validação estagnar, permitindo que cada arquitetura convirja em seu próprio ritmo.
5. **Métricas Oficiais da Competição:** Uso de **Categorical Crossentropy**, **Acurácia Categórica** e **ROC-AUC (One-vs-Rest)** para avaliação global.
6. **Eficiência Computacional e Batch Dinâmico:** Uso de `tf.data.AUTOTUNE` e escalonamento de batch por `strategy.num_replicas_in_sync`, garantindo saturação total da GPU (Tesla T4 no Colab) ou TPU sem estouro de memória.
7. **Persistência em Nuvem (Google Colab / Drive):** Montagem do Google Drive (`drive.mount('/content/drive')`) para salvar pesos em `models/` e figuras em `graphs/`, protegendo contra desconexões acidentais da sessão do Colab.
