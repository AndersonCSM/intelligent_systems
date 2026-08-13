# To-Do List: Projeto Final - Navegação Adaptativa com RL

## Fase 1: Configuração Inicial e Infraestrutura
- [x] **Instalação do Ambiente:** Instalar Webots e configurar dependências Python (`gymnasium`, `stable-baselines3`, `pyyaml`, `numpy`).
- [x] **Configuração do Webots:** Criar uma arena simples (paredes, obstáculos) e inserir o robô e-puck.
- [x] **Controlador Externo:** Alterar o campo `controller` do e-puck no Webots para `<extern>`.
- [x] **Criar `config.yaml`:** Definir parâmetros base (velocidade máxima, limites de colisão, total de *timesteps* de treinamento).

## Fase 2: Lógica Matemática e Recompensas
- [x] **Criar `rewards.py`:** Implementar a função `calculate_evasion_reward()`.
- [x] **Testar a Matemática:** Passar valores estáticos (mock) para a função e garantir que as penalidades estão escalando corretamente (ex: testar cenário onde $S_{max} > 0.95$ e garantir que a recompensa zere ou seja fortemente negativa).

## Fase 3: Construção do Ambiente (A Ponte Webots-Python)
- [x] **Criar `robot_env.py`:** Estruturar a classe `EPuckEnv` herdando de `gym.Env`.
- [x] **Definir Espaços:** Configurar o `observation_space` (8 sensores normalizados) e `action_space` (2 motores, de -1.0 a 1.0).
- [x] **Implementar `step()`:**
    - Ler sensores do Webots.
    - Calcular a recompensa (chamando `rewards.py`).
    - Enviar velocidades para os motores do Webots.
    - Retornar estado, recompensa e sinal de `done`.
- [x] **Implementar `reset()`:** Utilizar a API de Supervisor do Webots para reposicionar o robô no centro da arena e zerar suas velocidades.

## Fase 4: Orquestração e Treinamento
- [x] **Criar `train.py`:** Importar o ambiente e instanciar o modelo `PPO` da Stable-Baselines3.
- [x] **Treinamento de Validação:** Rodar o treinamento com apenas 1.000 épocas para garantir que o fluxo (Python -> Webots -> Python) não está travando.
- [x] **Treinamento Oficial:** Rodar um treinamento longo (ex: 50.000 a 100.000 *timesteps*).
- [x] **Salvar Pesos:** Garantir que o script gere o arquivo `.zip` (ex: `modelo_epuck_mlp.zip`) ao finalizar.

## Fase 5: Teste, Validação e Ajustes Finais
- [x] **Criar `test_model.py`** Carregar o modelo `.zip` gerado e executar um loop contínuo sem treinamento.
- [x] **Análise de Comportamento:** Assistir ao robô no Webots. Ele está rodopiando? Está muito medroso? 
- [x] **Tuning (se necessário):** Caso o comportamento não esteja ideal, ajustar as penalidades no `config.yaml` e treinar novamente.
- [x] **Coleta de Métricas:** Registrar quanto tempo o robô sobrevive sem bater após o treinamento comparado às primeiras épocas.

## Fase 6: Documentação Acadêmica (Relatório)
- [ ] **Introdução:** Contextualizar a disciplina de Sistemas Inteligentes e a limitação de heurísticas fixas em ambientes dinâmicos.
- [ ] **Metodologia:** Descrever a arquitetura (Webots + RL), a modelagem do MDP (Estado, Ação, Recompensa) e detalhar a matemática da função de evasão.
- [ ] **Resultados:** Comparar o comportamento puramente exploratório inicial com a navegação fluida alcançada pelas matrizes treinadas.
- [ ] **Referências:** Inserir a bibliografia formatada (Braitenberg, Sutton & Barto, Floreano, PPO).