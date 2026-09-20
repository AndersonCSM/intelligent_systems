# Projeto: Manutenção Preventiva - N-CMAPSS

Este projeto foca na predição da vida útil restante (RUL - *Remaining Useful Life*) de motores aeronáuticos (Turbofans) baseados no dataset sintético **N-CMAPSS** da NASA, utilizando técnicas de Machine Learning, especificamente Redes Neurais Artificiais (MLP - Multi-Layer Perceptron).

## Estrutura do Projeto

- `data_explore.ipynb`: Notebook focado na análise exploratória dos dados (EDA), distribuição dos sensores e entendimento das condições de voo e variáveis do motor.
- `model_training.ipynb`: Pipeline completo de Machine Learning (tratamento, treinamento de MLP e avaliação de performance usando MSE/RMSE).

## Como carregar os dados (Dataset N-CMAPSS)

O dataset N-CMAPSS é muito grande para ser hospedado no GitHub, com arquivos em formato HDF5 (`.h5`) pesando mais de 2 GB cada.

Para executar este projeto localmente, siga estes passos:

1. **Baixar o Dataset:** 
   O dataset público está disponível e pode ser baixado em plataformas de pesquisa abertas (como a [página oficial da NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) ou no repositório equivalente no Mendeley/Kaggle). Procure por "N-CMAPSS".
2. **Localização dos arquivos:** 
   Salve os arquivos `.h5` diretamente no diretório `data/` deste projeto.
   
   A estrutura deve ficar assim:
   ```text
   projetos/manutenção preventiva/
   ├── data/
   │   ├── N-CMAPSS_DS01-005.h5
   │   ├── N-CMAPSS_DS02-006.h5  <-- (Foco inicial dos notebooks)
   │   └── ...
   ├── data_explore.ipynb
   └── model_training.ipynb
   ```
3. **Instalar Dependências:**
   O projeto utiliza a biblioteca `h5py` para ler o formato HDF5, além das clássicas (Pandas, Numpy, Scikit-Learn e TensorFlow). Certifique-se de que estão instaladas em seu ambiente virtual:
   ```bash
   pip install h5py numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```
