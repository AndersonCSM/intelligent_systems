# Plano de Ação: Conclusão das Fases 2 e 3 (Navegação Adaptativa com RL)

Este documento descreve os passos técnicos necessários para concluir as pendências identificadas na Fase 2 e na Fase 3 do projeto. Siga estas etapas para implementar as correções você mesmo.

## Requisito Importante
Para conseguir mover o e-puck magicamente de volta ao centro da arena ao fim de cada episódio (tarefa da Fase 3), o script Python deve atuar como um *Supervisor* do Webots.
**Verifique o arquivo `mundo1.wbt`**: Certifique-se de que o nó `E-puck` possui a flag `supervisor TRUE`. Se não possuir, adicione essa linha dentro das chaves de definição do `E-puck`.

---

## Passo 1: Fase 2 - Lógica Matemática e Recompensas

Crie um script automatizado que passe valores estáticos para a função matemática e valide as penalidades.

**Criar arquivo: `controllers/controller_project/test_rewards.py`**
*   Crie um script (ex: usando `unittest`).
*   Importe `calculate_evasion_reward` do arquivo `rewards.py`.
*   Valide os seguintes cenários:
    1.  **Andar para frente sem obstáculos:** Sensores = `0.0`, velocidades left/right = `1.0`. A recompensa deve ser positiva e `done=False`.
    2.  **Girar no próprio eixo:** Sensores = `0.0`, velocidades left = `1.0`, right = `-1.0`. A recompensa deve ser igual ou muito próxima de `0.0`.
    3.  **Colisão (Falha):** Um dos sensores com leitura `> 0.95`. A função deve retornar `reward=-100.0` e `done=True`.

---

## Passo 2: Fase 3 - Ponte Webots-Python (Configuração do Robô)

Alinhe a nomenclatura e prepare a classe para atuar como Supervisor.

**Modificar arquivo: `controllers/controller_project/robot.py`**
1.  Na importação no topo, troque `from controller import Robot, ...` para `from controller import Supervisor, ...`.
2.  Altere a inicialização na classe: substitua `self.robot = Robot()` por `self.robot = Supervisor()`.
3.  **Dica de Nomenclatura:** Renomeie a classe de `EPuck` para `EPuckLocomotion` para manter a consistência com o que já foi importado no arquivo `robot_env.py`.
4.  Crie um novo método `reset_position(self)` na classe. Esse método deverá usar a API do Supervisor para buscar o próprio nó (ex: `self.robot.getSelf()`) e redefinir os campos `translation` (ex: `[0, 0, 0]`) e `rotation` para restaurar a posição de origem. Use também `self.robot.simulationResetPhysics()`.

---

## Passo 3: Fase 3 - Ponte Webots-Python (Ambiente Gym)

Conecte o ambiente Gym à nova função de teletransporte.

**Modificar arquivo: `controllers/controller_project/robot_env.py`**
1.  Dentro da função `reset()`, adicione a chamada para a nova função que você criou no passo anterior (ex: `self.locomotion.reset_position()`).
2.  Isso garantirá que, ao início de cada episódio, o Webots teletransporte fisicamente o robô de volta para a posição inicial e resete as velocidades.

---

## Passo 4: Validação
1.  Rode o seu novo script `test_rewards.py` no terminal do ambiente virtual para confirmar a Fase 2.
2.  Execute um loop de testes manual do Gymnasium sem RL ainda, e verifique visualmente na tela do Webots se o e-puck volta imediatamente para o centro após X passos ou após bater na parede.
3.  Por fim, marque os itens pendentes da Fase 2 e 3 como concluídos no arquivo `todo.md`.
