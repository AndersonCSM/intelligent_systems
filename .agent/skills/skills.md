# Skills do Agente

Este documento lista as habilidades essenciais que o assistente deve aplicar para desenvolver e otimizar as funcionalidades do repositório `intelligent_systems`, abrangendo as práticas da disciplina e o projeto aplicado de robótica.

## 1. Análise de Algoritmos em IA e Machine Learning
- **Descrição**: Capacidade de analisar e refatorar scripts de Inteligência Artificial, Aprendizado de Máquina e Aprendizado por Reforço.
- **Ações**:
  - Otimizar modelos de regressão, classificação (KNN, SVM) e Redes Neurais.
  - Aplicar boas práticas de desenvolvimento (`Clean Code`) na estruturação de notebooks (`.ipynb`) no diretório `/course`.
  - Refinar a lógica de recompensas e hiperparâmetros de modelos RL aplicados em robótica.

## 2. Integração e Controle em Robótica (Webots)
- **Descrição**: Suporte específico para as atividades do `/project-robotics`.
- **Ações**:
  - Desenvolver, depurar e refatorar controladores externos escritos em Python (ou C/C++) para ambientes Webots.
  - Ajustar o pipeline de dados de sensores (Câmera, LiDAR, ultrassônicos) e navegação autônoma.
  
## 3. Documentação e Comentários
- **Descrição**: Garantir que as funcionalidades teóricas e as implantações práticas fiquem rigorosamente documentadas.
- **Ações**:
  - Adicionar docstrings ricas (preferencialmente detalhando parâmetros e retornos).
  - Atualizar os manuais (como `README.md`, `environment.md`, `workflow.md`) sempre que novas funcionalidades, dependências ou arquiteturas forem introduzidas.
  - Explicar passo a passo soluções matemáticas implementadas.

## 4. Automação e Gerenciamento de Dependências
- **Descrição**: Otimizar a manutenção do repositório e de seus ecossistemas de desenvolvimento.
- **Ações**:
  - Gerenciar arquivos como `requirements.txt`, mantendo clareza nas bibliotecas exigidas (ex: scikit-learn, pytorch, numpy) para cada subsistema.
  - Desenvolver e polir scripts de execução de controladores, automação de testes ou formatação de código (ex: `black`, `flake8`).

## 5. Resolução de Problemas (Debugging Avançado)
- **Descrição**: Isolar, diagnosticar e corrigir falhas de execução e inconsistências de software.
- **Ações**: 
  - Analisar stack traces em Python e falhas de runtime da simulação do Webots.
  - Rastrear chamadas assíncronas ou falhas de IPC.
  - Propor soluções robustas que evitem comprometer o restante do ambiente já devidamente configurado.

## 6. Política de Modificação de Código
- **Descrição**: Nenhuma alteração em código-fonte, notebooks ou arquivos de implementação deve ser feita sem autorização explícita do usuário.
- **Ações**:
  - Antes de editar qualquer arquivo, confirmar a permissão do usuário quando a solicitação não deixar isso inequívoco.
  - Limitar respostas a explicações, diagnósticos e sugestões quando não houver autorização para alteração.
  - Considerar como código-fonte as células de notebooks, scripts e demais arquivos executáveis do repositório.
