# Projeto Final Sistemas inteligentes - Aprendizado de Máquina na Robótica


## Escopo do Projeto Final: Navegação Adaptativa via Aprendizado por Reforço

### Contexto e Motivação
A robótica clássica utiliza algoritmos puramente matemáticos, como os **Coeficientes de Braitenberg**, para evasão de obstáculos. Embora eficientes para geometrias específicas, eles são estáticos e incapazes de lidar com ambientes dinâmicos, desgaste de hardware ou multitarefas (ex: desviar de parede E ir para a luz). O projeto substitui essa heurística fixa por uma **Rede Neural MLP (Multilayer Perceptron)**.

### Metodologia de Aprendizado
Como não existe um "gabarito" em tempo real para navegação, o uso de Aprendizado Supervisionado foi descartado. A abordagem escolhida é o **Aprendizado por Reforço (Reinforcement Learning)**, utilizando o algoritmo **PPO** (via biblioteca *Stable-Baselines3*).

* **Ambiente (Simulador):** Webots com o modelo de robô e-puck.
* **Espaço de Observação (Entradas da MLP):** Vetor com 8 valores normalizados (0 a 1) vindos dos sensores de distância infravermelhos.
* **Espaço de Ação (Saídas da MLP):** Vetor contínuo (-1.0 a 1.0) controlando a velocidade do motor esquerdo e direito.

### A Função de Recompensa (Reward Function)
Para evitar o *Reward Hacking* (ex: o robô ficar girando no próprio eixo para não bater), definiu-se uma função de recompensa **multiplicativa**:

$$R = V_{frente} \cdot (1 - \sqrt{\Delta v}) \cdot (1 - S_{max}^2)$$

* **$V_{frente}$**: Incentiva o movimento frontal contínuo.
* **$\Delta v$**: Penaliza a diferença de velocidade entre as rodas (evita rotação excessiva no próprio eixo).
* **$S_{max}^2$**: Aplica uma penalidade exponencial com base no sensor que detecta o obstáculo mais próximo (força o desvio suave).
* **Condição de Falha:** Se $S_{max} > 0.95$, o episódio encerra com recompensa massiva negativa (ex: -100) e o robô é resetado.

### Arquitetura de Software
O código será estruturado focando em modularidade e boas práticas de engenharia de software, dividindo responsabilidades:

1.  **`config.yaml`**: Central de hiperparâmetros (tempos, limites, velocidades máximas). Mantém o código limpo e os experimentos rastreáveis.
2.  **`rewards.py`**: Isola as equações matemáticas da Função de Recompensa.
3.  **`robot_env.py`**: Classe que herda de `gym.Env`. Atua como a "ponte", enviando comandos para a API do Webots e lendo os sensores (o Ambiente).
4.  **`train.py`**: O orquestrador. Carrega o ambiente, inicializa a rede PPO, executa o loop de treinamento e salva os pesos (`.zip`).
5.  **`test_model.py` (ou `.ipynb`)**: Utilizado para carregar os pesos da rede já treinada e analisar visualmente ou graficamente o comportamento do robô.

## Referências
* **Braitenberg, V. (1984).** *Vehicles: Experiments in Synthetic Psychology.* MIT Press.
* **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* **Amodei, D., et al. (2016).** *Concrete Problems in AI Safety.* arXiv preprint arXiv:1606.06565.
* **Nolfi, S., & Floreano, D. (2000).** *Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines.* MIT Press.
* **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347.
* **Raffin, A., et al. (2021).** *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research, 22(268), 1-8.
* **Towers, M., et al. (2023).** *Gymnasium: A Standard Interface for Reinforcement Learning Environments.*
* **Michel, O. (2004).** *Cyberbotics Ltd. Webots™: Professional Mobile Robot Simulation.* International Journal of Advanced Robotic Systems, 1(1), 39-42.
