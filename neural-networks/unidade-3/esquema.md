# Esquema e Arquitetura do Projeto — Sistemas Inteligentes (Unidade 3)

Este documento define a **estrutura de diretórios, fluxo de execução modular e protocolo experimental** padrão adotados nos projetos do repositório, servindo como base arquitetural para o novo projeto em `intelligent-systems/unidade-3/projeto/`.

---

## 1. Estrutura de Diretórios

```text
intelligent-systems/unidade-3/projeto/
├── data/
│   ├── raw/                    # Dados brutos originais (imutáveis)
│   └── processed/              # Splits tratados (treino/validação/teste) prontos para modelagem
├── graphs/                     # Gráficos de EDA, curvas de treino, matriz de confusão, resíduos e XAI
├── models/                     # Modelos treinados (.keras, .pth, .joblib) e scalers ajustados no treino
├── 00_loader.ipynb             # (Opcional) Ingestão, download via API/Kaggle e validação inicial
├── 01_eda.ipynb                # Análise exploratória, qualidade dos dados, correlações e diagnóstico
├── 02_preparacao_core.ipynb    # Limpeza, feature selection, split sem vazamento e scaling
├── 03_core_modelos.ipynb       # Modelo baseline + testes comparativos / ablações controladas
├── 04_avaliacao_xai.ipynb      # Avaliação final no teste, análise de erros e interpretabilidade
├── esquema.md                  # Este documento (arquitetura e plano estrutural)
├── strategy.md                 # Registro de decisões técnicas, justificativas e protocolo experimental
├── README.md                   # Documentação técnica final do projeto (problema, método, resultados)
└── .gitignore                  # Exclusão de datasets pesados, checkpoints locais e caches
```

---

## 2. Fluxo de Execução Modular (Pipeline em Etapas)

A arquitetura separa responsabilidades em notebooks numerados para evitar reprocessamento pesado, garantir reprodutibilidade e facilitar comparações justas entre modelos:

| Etapa | Notebook / Artefato | Responsabilidade Principal | Saídas Esperadas |
| :---: | :--- | :--- | :--- |
| **00** | `00_loader.ipynb` | Download automatizado ou carregamento bruto e checagem de integridade/tipos. | Arquivos em `data/raw/` |
| **01** | `01_eda.ipynb` | Perfil estatístico (`info`, `describe`), valores ausentes, duplicatas, outliers, balanceamento de classes e matriz de correlação. | Diagnóstico em `strategy.md` + gráficos em `graphs/` |
| **02** | `02_preparacao_core.ipynb` | Seleção de atributos, divisão estratificada/temporal (`train`/`val`/`test`) com `random_state` fixo e normalização (`fit` apenas no treino). | Tensores/CSVs limpos em `data/processed/` + `scaler` em `models/` |
| **03** | `03_core_modelos.ipynb` | Treinamento do modelo **Baseline** seguido de **Testes Comparativos / Ablações** sob o mesmo protocolo de hiperparâmetros e validação cruzada. | Pesos/modelos em `models/` + curvas em `graphs/` |
| **04** | `04_avaliacao_xai.ipynb` | Avaliação única e final no conjunto de **Teste**, matriz de confusão / análise de resíduos e explicabilidade (Feature Importance, SHAP ou XAI). | Métricas finais, figuras em `graphs/` e síntese no `README.md` |

---

## 3. Alinhamento com o Workflow de Sistemas Inteligentes

Todo desenvolvimento nesta pasta segue o **Workflow de 10 Passos** definido em [`intelligent-systems/workflow.md`](../../workflow.md) e [`intelligent-systems/guide.md`](../../guide.md):

1. **Exploração e Qualidade dos Dados:** Evidências quantitativas e visuais antes de qualquer decisão de modelagem (`01_eda.ipynb`).
2. **Validação de Premissas:** Verificar linearidade, independência, multicolinearidade, estacionariedade ou desbalanceamento.
3. **Seleção de Features:** Redução de ruído e colinearidade baseada em evidência estritamente no treino (`02_preparacao_core.ipynb`).
4. **Normalização / Scaling:** `StandardScaler`, `MinMaxScaler` ou `RobustScaler` ajustados **exclusivamente** no conjunto de treino.
5. **Divisão dos Dados:** Separação estrita treino/validação/teste (estratificada para classificação ou temporal para séries).
6. **Ajuste e Comparação de Modelos:** Baseline simples primeiro, seguido de modelos mais complexos ou regularizados com validação cruzada (`03_core_modelos.ipynb`).
7. **Avaliação Consistente:** Uso uniforme das métricas adequadas ao problema (ex.: Accuracy, Precision, Recall, F1, ROC-AUC, Matriz de Confusão para classificação; MSE, RMSE, MAE, $R^2$ para regressão).
8. **Análise de Importância / Coeficientes:** Interpretação do impacto das variáveis no modelo (`04_avaliacao_xai.ipynb`).
9. **Análise de Erros e Resíduos:** Investigação de falsos positivos/negativos, QQ-plots ou viés residual.
10. **Relatório e Documentação:** Consolidação clara de resultados, limitações e trabalhos futuros em `README.md` e `strategy.md`.

---

## 4. Diretrizes Técnicas e Reprodutibilidade

- **Ambiente Local:** Conda `intelligent_systems` (Python 3.11) ou Google Colab quando houver necessidade de GPU dedicada.
- **Prevenção de Data Leakage:** Nenhuma estatística global (média, desvio padrão, seleção de features ou balanceamento sintético como SMOTE) pode tocar os dados de validação/teste antes do `split`.
- **Controle de Experimentos:** Comparações entre modelos devem manter fixos os dados de entrada, a semente aleatória (`seed`) e as métricas de avaliação.
- **Versionamento Limpo:** Não versionar datasets brutos grandes, arquivos `.parquet`/`.csv` pesados ou checkpoints temporários (configurar `.gitignore` local).

---

## 5. Checklist de Inicialização (Próximo Chat)

- [ ] Definir o **tema, problema e dataset** do projeto da Unidade 3.
- [ ] Criar `.gitignore` e `README.md` inicial em `intelligent-systems/unidade-3/projeto/`.
- [ ] Configurar `data/raw/` e rodar `00_loader.ipynb` (se necessário) + `01_eda.ipynb`.
- [ ] Registrar as primeiras decisões técnicas em `strategy.md`.
