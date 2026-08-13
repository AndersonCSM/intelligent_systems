# Função de movimentação melhorada
Cada sensor irá calcular calcular um custo associado a leitura da posição, velocidade e movimento do robô.

Entradas do sensor:
- 8 sensores ultrassônicos distribuidos de acordo com o robô e-puck
- 2 motores para definir a velocidade

Saída:
- 2 motores.

Critérios da função de custo 
- O robô possui um threshold de distância minima de 0.5 cm para reagir, esse valor é um hiperparametro do usuário
- O robô deve considerar se o sensor está ativo ou não: valor [0 XOR 1]
- O robô deve considerar a leitura do sensor: 0 (no objects detected) até 4095 (object near the sensor)
- O robô deve considerar a velocidade velocidade dos motores da esquerda e da direita
- O robô deve considerar o movimento atual que se está fazendo: ENUM de opções dentro do conjunto de movimentos
- Cada sensor deve possuir um peso associando na influência do movimento, similar a braitengerg
    - Sensores frontais maior peso, sensores laterais menor peso para evitar que o robô passe próximo a um obstáculo e pela leitura do sensor lateral começe a fazer curva desnecessária.
- O robô deve considerar uma constante K para natureza do robô, ideal como 1.
- O robô deve considerar uma função externa que será uma métrica de objetivo, hiperparametro como 1

Custo local: define o custo associado a cada movimento
- custo frente: apenas os sensores frontais são considerados
- custo mover  esquerda: os sensores da esquerda são considerados
- custo mover  direita: os sensores da direita são considerados
- custo turn back: todos os sensores são considerados

Conjunto de movimentos:
- mover frente: O robô move para frente enquanto não atingir o threshold de custo que indica obstáculo.
    - movimento padrão do robô
- mover esquerda com x graus: O robô irá girar para a esquerda no próprio eixo até a função de custo está aceitável e retornar o movimento de mover frente.
    - Se o threshold da direita for atingido, significa que ele precisa virar para a esquerda.
- mover direita com x graus: O robô irá girar para a direita no próprio eixo até a função de custo está aceitável e retornar o movimento de mover frente.
    - Se o threshold da esquerda for atingido, significa que ele precisa virar para a direta.
- turn back (girar 360): o robô deve acionar os motores em sentido inverso para rotacionar 360 graus e retornar a se mover
    - Quando o custo for máximo, ele entrou em um beco sem saida.


Lógica de funcionamento:
1. Robô inicia no centro da arena, com um angulo em z aleatório e se move para frente.
2. Quando uma função de custo local atingir o threshold, o robô observa o movimento que está fazendo
3. Se o movimento do robô continua aumentando o custo local, ele para o movimento atual e começa outro movimento como fazer curva
4. Ele faz movimento de curva de acordo com a posição que vai minimizar a função de custo local que lançou o threshold e de acordo com o movimento atual.
    4.1. Se o custo local é da esquerda, ele irá mover direita com x graus até a função de custo sair do thresold, então volta a mover frente.
    4.2. o movimento atual deve ser considerado pois ele pode está se movendo para frente e o sensores laterais direita/esquerda alarmarem por passar próximo de um obstáculo, mas o movimento dele não irá colidir, a menos que o obstáculo ultrapasse o threshold.


 threshold 

def distancia_minima(sensores_ativos, leituras, R=10.0):
    distancias = []
    for ativo, leitura in zip(sensores_ativos, leituras):
        if ativo:
            d = R * (1 - leitura)
            distancias.append(d)
    if not distancias:
        return float('inf')   # nenhum sensor ativo → sem obstáculo
    return min(distancias)