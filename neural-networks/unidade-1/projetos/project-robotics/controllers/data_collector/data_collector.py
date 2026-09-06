import csv
import math
import random
from controller import Supervisor

def get_min_distance(x, y):
    """
    Calcula a menor distância teórica entre a coordenada (x, y)
    e os limites da arena ou os obstáculos.
    """
    # A arena é 1.5 x 1.5, com o centro em (0,0). Os limites são -0.75 e 0.75.
    wall_dist = min([
        abs(x - (-0.75)), # Parede Esquerda
        abs(x - 0.75),    # Parede Direita
        abs(y - (-0.75)), # Parede Inferior
        abs(y - 0.75)     # Parede Superior
    ])
    
    # Lista de obstáculos baseada no 'mundo1.wbt'
    # Estamos tratando caixas como círculos/cilindros inscritos (raio) para agilizar o cálculo.
    obstacles = [
        {'x': 0.0, 'y': -0.549, 'r': 0.05},     # caixa1
        {'x': 0.251, 'y': 0.112, 'r': 0.05},    # caixa2
        {'x': -0.430, 'y': 0.0, 'r': 0.075},    # box(3)
        {'x': 0.610, 'y': 0.0, 'r': 0.075},     # box(4)
        {'x': 0.0, 'y': 0.250, 'r': 0.05},      # box(1)
        {'x': 0.368, 'y': -0.415, 'r': 0.05},   # OilBarrel
        {'x': -0.114, 'y': -0.314, 'r': 0.09},  # oil barrel(1)
        {'x': 0.055, 'y': 0.511, 'r': 0.05},    # oil barrel(2)
        {'x': -0.392, 'y': 0.358, 'r': 0.09},   # oil barrel(3)
        {'x': 0.428, 'y': 0.387, 'r': 0.05},    # oil barrel(4)
        {'x': -0.473, 'y': -0.265, 'r': 0.111}, # RobocupSoccerBall
    ]
    
    min_obs_dist = float('inf')
    for obs in obstacles:
        # Distância Euclidiana até o centro do obstáculo
        dist_to_center = math.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
        # Desconta o raio (tamanho físico) para ter a distância até a borda
        dist_to_surface = dist_to_center - obs['r']
        if dist_to_surface < min_obs_dist:
            min_obs_dist = dist_to_surface
            
    # Subtraímos o raio do e-puck (3.7 cm) para que a distância 
    # seja a partir da borda do robô, não do centro matemático!
    EPUCK_RADIUS = 0.037
    return min(wall_dist, min_obs_dist) - EPUCK_RADIUS

def run():
    # Inicializa a API do Webots
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    
    # Obtém o nó do robô que está executando este controlador
    robot_node = supervisor.getSelf()
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")
    
    # Inicializa os sensores
    sensors = []
    for i in range(8):
        sensor = supervisor.getDevice(f"ps{i}")
        sensor.enable(timestep)
        sensors.append(sensor)
        
    # Abre o arquivo CSV
    csv_filename = 'dataset.csv'
    csv_file = open(csv_filename, 'w', newline='')
    writer = csv.writer(csv_file)
    
    # Escreve o cabeçalho (Conforme o Markdown Etapa 3.2)
    header = [f'ps{i}' for i in range(8)] + [f'ps{i}_norm' for i in range(8)] + ['distance_min', 'cost', 'collision']
    writer.writerow(header)
    
    NUM_SAMPLES = 2000
    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print(f"Iniciando coleta de dados ({NUM_SAMPLES} amostras de posições válidas)...")
    
    samples_collected = 0
    # O loop principal roda enquanto o simulador não for fechado e não terminarmos as amostras
    while supervisor.step(timestep) != -1 and samples_collected < NUM_SAMPLES:
        
        # 1. Amostragem com Rejeição (Rejection Sampling)
        # Gera uma coordenada X e Y aleatória dentro do espaço útil da arena
        x = random.uniform(-0.65, 0.65)
        y = random.uniform(-0.65, 0.65)
        
        dist_min = get_min_distance(x, y)
        
        # Foco em áreas interessantes: descartamos apenas pontos 
        # muito longe (> 0.3m) ou excessivamente dentro da parede (< -0.02m).
        if dist_min > 0.3 or dist_min < -0.02: 
            continue
            
        print(f"[{samples_collected+1}/{NUM_SAMPLES}] Coletando na posição ({x:.2f}, {y:.2f}), dist={dist_min:.3f}m")
        
        # 2. Teleporte do Robô
        # No Webots (ENU), z é o eixo vertical. Vamos colocar perto de 0 para não afundar no chão.
        trans_field.setSFVec3f([x, y, 0.005])
        
        # 3. Rotação em 8 Ângulos (com Jitter) e Leitura
        for base_angle_deg in ANGLES:
            # Melhoria 1: Ruído aleatório (Jitter) de +/- 15 graus no ângulo
            angle_deg = base_angle_deg + random.uniform(-15.0, 15.0)
            angle_rad = math.radians(angle_deg)
            # O robô rotaciona no eixo Z
            rot_field.setSFRotation([0, 0, 1, angle_rad])
            
            # Resetar a física evita que o robô seja "arremessado" ou preserve momento do teleporte
            robot_node.resetPhysics()
            
            # Avança a simulação um pouquinho para o sensor físico do Webots atualizar a leitura
            for _ in range(5):
                supervisor.step(timestep)
                
            # Lê os 8 sensores originais
            raw_readings = [s.getValue() for s in sensors]
            
            # Melhoria 3: Ruído Gaussiano de ~2% no sensor (desvio padrão de 80)
            readings = []
            for r in raw_readings:
                noisy_r = r + random.gauss(0, 80)
                # Clipa os valores para garantir que fiquem no limite físico do sensor
                noisy_r = max(0.0, min(4096.0, noisy_r))
                readings.append(noisy_r)
                
            readings_norm = [r / 4096.0 for r in readings]
            
            # Calcula o 'Cost' - a variável alvo da MLP
            if dist_min <= 0:
                cost = 1.0
                collision = 1
            else:
                # Função de custo decrescente exponencial (alpha = 5)
                cost = math.exp(-5.0 * dist_min)
                collision = 0
                
            # Salva no CSV
            row = readings + readings_norm + [dist_min, cost, collision]
            writer.writerow(row)
            
        samples_collected += 1
        
    csv_file.close()
    print(f"✅ Coleta finalizada! {NUM_SAMPLES * len(ANGLES)} linhas foram gravadas no '{csv_filename}'.")
    
    # Pausa a simulação quando acabar
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

if __name__ == '__main__':
    run()
