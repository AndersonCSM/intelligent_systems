# Documentação Metodológica: Grid Search (Busca Sistemática em Grade)
### Projeto Plant Pathology 2020 (FGVC7) - Redes Neurais

Este documento consolida a especificação teórica, o espaço de busca, os protocolos de experimentação e a fundamentação metodológica do **Grid Search**, registrando formalmente o planejamento da busca de hiperparâmetros no projeto.

---

## 1. Fundamentação Teórica e Conceito

O **Grid Search (Busca em Grade)** é o método exaustivo padrão para otimização de hiperparâmetros em aprendizado de máquina. Ele consiste na definição prévia de um conjunto discreto de valores para cada hiperparâmetro de interesse, avaliando o desempenho do modelo no produto cartesiano dessas combinações:

$$\Omega = \Lambda_1 \times \Lambda_2 \times \dots \times \Lambda_K$$

Onde cada $\lambda_k \in \Lambda_k$ representa uma escolha de configuração (ex: taxa de aprendizado, taxa de dropout, tamanho de lote).

### Quando o Grid Search é Indispensável:
1. **Baselines com Desempenho Fraco ou Instável:** Quando a acurácia inicial do modelo é baixa (< 70–80%) ou ocorrem problemas de explosão/desaparecimento de gradiente (*gradient explosion/vanishing*).
2. **Incerteza na Região de Convergência do Otimizador:** Quando não se sabe previamente qual ordem de grandeza da taxa de aprendizado ($10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}$) permite convergência estável do Adam/SGD no problema específico.
3. **Calibração de Regularização Contra Overfitting:** Quando a capacidade do modelo excede o volume do conjunto de dados e torna-se imperativo quantificar a sensibilidade do modelo a diferentes níveis de *Dropout* e *Weight Decay*.

---

## 2. Espaço de Busca Cartesiano (*Hyperparameter Grid*)

O grid formulado para o projeto contempla variações arquiteturais, de otimização, de regularização e de partição de dados:

| Hiperparâmetro | Notação | Valores / Espaço a Variar | Justificativa Técnica |
| :--- | :---: | :--- | :--- |
| **Modelos (Backbones)** | $\mathcal{M}$ | `DenseNet121`, `EfficientNetB0`, `ResNet50V2`, `MobileNetV2` | Avaliar famílias estruturalmente distintas: densas (concatenação), escaladas (MBConv), residuais (soma aditiva) e compactas (*inverted residuals*). |
| **Taxa de Aprendizado Inicial** | $\eta$ | `[1e-3, 5e-4, 1e-4]` | Identificar a taxa de partida ideal para convergência do otimizador Adam sem oscilação excessiva nos primeiros passos. |
| **Épocas Teto** | $E_{\text{max}}$ | `[100, 200, 400]` *(com EarlyStopping)* | Prover um horizonte de treino longo o suficiente para convergência plena, interrompido via callback. |
| **Divisão de Validação** | $v_{\text{split}}$ | `[0.15, 0.20, 0.25]` *(estratificado)* | Mensurar a sensibilidade do aprendizado ao volume relativo de dados de treino vs. validação. |
| **Taxa de Dropout** | $p_{\text{drop}}$ | `[0.0, 0.1, 0.3]` | Avaliar o grau ótimo de desativação aleatória de neurônios no topo da rede para mitigar memorização. |
| **Batch Size Base** | $B$ | `[32, 64]` | Escalonado com `num_replicas_in_sync`, balanceando estabilidade dos gradientes estocásticos e tempo de época na GPU. |

### Dimensão do Espaço e Custo Computacional:
* **Total de Combinações Teóricas:**
  $$|\Omega| = 4 \times 3 \times 3 \times 3 \times 3 \times 2 = 648 \text{ configurações}$$
* **Tempo Estimado:** Considerando uma média de 12 minutos por treino em GPU Tesla T4 (mesmo com *EarlyStopping*), a execução exaustiva demandaria aproximadamente **129 horas ininterruptas de GPU** (~5,4 dias), extrapolando largamente as cotas padrão do Google Colab.

---

## 3. Mecanismos de Treinamento e Callbacks Unificados do Grid

Para assegurar comparabilidade justa entre todas as combinações executadas, o laço do Grid Search opera sob protocolos padronizados:

```python
callbacks_grid = [
    # Interrompe o treino quando o modelo estagnar na validação
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduz adaptativamente a taxa de aprendizado para ajuste fino
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    # Salva o melhor checkpoint com base na métrica oficial do desafio
    callbacks.ModelCheckpoint(
        filepath=f"models/best_{model_name}_cfg{cfg_id}.keras",
        monitor='val_roc_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # Auditoria tabular contínua da evolução
    callbacks.CSVLogger(f"models/history_{model_name}_cfg{cfg_id}.csv")
]
```

---

## 4. Estrutura de Artefatos e Persistência do Grid Search

O protocolo prevê o salvamento dos seguintes artefatos para cada execução do grid:

1. **`models/grid_search_summary.csv`:** Tabela consolidada com todas as combinações avaliadas contendo:
   * `config_id`, `model_name`, `initial_lr`, `val_split`, `dropout`, `batch_size`
   * `epochs_trained`, `val_loss`, `val_accuracy`, `val_roc_auc`, `val_macro_f1`, `tempo_total_s`
2. **`models/history_{model_name}_cfg{id}.csv`:** Histórico de convergência época a época.
3. **`models/classification_report_{model_name}_cfg{id}.csv`:** Precisão, Recall e F1 por classe.
4. **`graphs/`:** Curvas de Loss, Acurácia e ROC-AUC a 300 DPI para as melhores configurações.

---

## 5. Racional da Decisão Metodológica no Projeto

### Por que o Grid Search Exaustivo foi Substituído pelo Benchmarking Direcionado?

Durante a execução da Prova de Conceito (PoC) documentada em `02_pipeline_single.ipynb` e `02_pipeline_improved.ipynb`, obtivemos as seguintes constatações empíricas cruciais:

1. **Convergência Imediata de Alto Padrão:**
   * Ao aplicar a resolução completa de **512x512** (preservando microdetalhes das lesões), base descongelada (*fine-tuning*) com taxa de `1e-4`, otimizador Adam e pré-processamento adaptativo CLAHE, o modelo baseline já superou **95% de acurácia** e **0.97 de Macro ROC-AUC**.
2. **Retorno Marginal Decrescente:**
   * Na literatura de Transfer Learning para classificação de imagens médicas e agrícolas, uma vez calibrados os parâmetros de primeira ordem (resolução, taxa de partida na ordem de $10^{-4}$ e *ReduceLROnPlateau*), o ajuste fino de hiperparâmetros menores traz oscilações marginais da ordem de **0.5% a 2%**, às custas de dezenas de horas de processamento.
3. **Risco de Sobreajuste de Validação (*Data Snooping*):**
   * Avaliar centenas de combinações em um conjunto de validação de ~365 imagens (20% de 1.821) gera uma seleção enviesada para ruídos da amostra de validação em vez de generalização real.
4. **Limitação Operacional de Cotas em Nuvem (Google Colab):**
   * A execução de 648 ou mesmo 30+ treinos completos exaure os créditos computacionais (*compute units*) e causa desconexões por inatividade.

### Decisão Estratégica:
* **Hiperparâmetros Fixados no Ponto Ótimo:**
  * Resolução: `512x512`
  * Taxa de Aprendizado: `1e-4` (com `ReduceLROnPlateau` fator 0.2, paciência 3)
  * Regularização: `Dropout(0.3)` + `EarlyStopping` (paciência 8)
  * Batch Size: `16` por réplica com precisão mista `mixed_float16`
  * Validação: `20%` estratificada
* **Pivô Metodológico:** A energia computacional foi redirecionada para a comparação de **Famílias Arquiteturais Distintas** (DenseNet, EfficientNetV2, ResNet, ConvNeXt, Xception) para alimentar a etapa de **Ensemble**, onde a diversidade de representação gera ganho comprovado de generalização.
