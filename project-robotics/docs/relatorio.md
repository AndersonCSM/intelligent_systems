# Relatório Técnico: Navegação Adaptativa via Aprendizado por Reforço

## 1. Introdução
No contexto da disciplina de Sistemas Inteligentes, o problema de navegação autônoma e evasão de obstáculos é historicamente abordado por heurísticas fixas, como os clássicos Coeficientes de Braitenberg. Embora eficientes para geometrias específicas e ambientes controlados, essas abordagens estáticas apresentam limitações significativas em ambientes dinâmicos ou na presença de ruído nos sensores e desgaste de hardware. Este projeto propõe a substituição dessa heurística por um modelo de Aprendizado de Máquina, especificamente o Aprendizado por Reforço (RL), permitindo que o robô (e-puck) desenvolva de forma autônoma uma política de navegação baseada em tentativa e erro, adaptando-se às condições do ambiente simulado no Webots.

## 2. Metodologia

A arquitetura do sistema baseia-se na integração do simulador Webots com a biblioteca `Gymnasium` em Python, configurando o robô e-puck para operar com um controlador `<extern>`. O treinamento é orquestrado pelo algoritmo Proximal Policy Optimization (PPO), implementado na biblioteca `Stable-Baselines3`.

A modelagem do Processo de Decisão de Markov (MDP) foi definida da seguinte forma:
- **Espaço de Estado (Observação):** Um vetor contínuo contendo 8 valores normalizados $S \in [0, 1]^8$, correspondentes às leituras dos sensores infravermelhos de distância do e-puck.
- **Espaço de Ação:** Um vetor contínuo de duas dimensões $A \in [-1.0, 1.0]^2$, onde cada valor é escalado e enviado diretamente como velocidade em rad/s para o motor esquerdo e direito, respectivamente.
- **Função de Recompensa:** Projetada para evitar o *Reward Hacking* e incentivar a progressão fluida. Baseada na clássica função de *fitness* de Floreano et al., a recompensa em cada passo de tempo (quando o robô se move para frente) é calculada como:

$$R = V_{frente} \cdot (1 - \Delta v) \cdot (1 - S_{max})$$

Onde $V_{frente}$ é a velocidade média translacional, $\Delta v$ é a diferença absoluta entre as velocidades das rodas (penalizando a rotação no próprio eixo) e $S_{max}$ é a leitura máxima dentre os sensores de proximidade (penalizando a proximidade a obstáculos). Caso o $S_{max}$ ultrapasse um limiar crítico (0.95), indicando colisão, o episódio é imediatamente encerrado com uma penalidade massiva, evitando o comportamento anômalo de "suicídio" da rede.

## 3. Resultados
Durante os estágios iniciais de treinamento, a política exploratória resultou em movimentos erráticos, como a rotação incessante no próprio eixo ou colisões diretas com as paredes devido à incapacidade inicial de correlacionar os valores altos dos sensores com a penalidade terminal. Após ajustes na função de recompensa (para garantir gradientes estritamente positivos durante a sobrevivência útil), as matrizes treinadas com o PPO convergiram para uma navegação fluida. O robô adquiriu a capacidade de transitar suavemente pelo centro da arena e realizar curvas precisas ao se aproximar dos obstáculos, demonstrando superioridade de adaptação em relação a controladores reativos estáticos.

## 4. Referências
- Braitenberg, V. (1984). *Vehicles: Experiments in Synthetic Psychology.* MIT Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Nolfi, S., & Floreano, D. (2000). *Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines.* MIT Press.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347.
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research.
- Towers, M., et al. (2023). *Gymnasium: A Standard Interface for Reinforcement Learning Environments.*
