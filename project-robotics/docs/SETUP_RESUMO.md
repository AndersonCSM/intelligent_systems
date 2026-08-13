# SETUP - Project Robotics Python

## 1. Criar o venv

```bash
cd ~/github_projects/intelligent_systems/project-robotics
chmod +x scripts/setup_venv.sh
scripts/setup_venv.sh
```

## 2. Ativar o venv

```bash
source robotics/bin/activate
```

## 3. Instalar Webots R2025a

```bash
# Opcao 1 - Via pip (mais simples)
pip install webots

# Opcao 2 - Usar instalacao local
chmod +x scripts/setup_webots.sh
./scripts/setup_webots.sh
```

## 4. Testar integracao

```bash
python -c "from controller import Robot; print('OK')"
```

## 5. Rodar exemplos

```bash
# Abra Webots e carregue mundo1.wbt
python -m controllers.controller_project.examples
```

## Estrutura da Classe EPuckLocomotion

```bash
EPuckLocomotion
├── Configuracao
│   ├── time_step (64 ms default)
│   ├── max_speed (6.28 rad/s)
│   ├── num_sensors (8)
│   └── sensor_range (512)
│
├── Hardware
│   ├── sensors[0-7] (DistanceSensor)
│   ├── left_motor (Motor)
│   └── right_motor (Motor)
│
├── Controle
│   ├── read_sensors() -> normalizado [0-1]
│   ├── set_motor_speeds(left, right)
│   ├── apply_braitenberg_control(sensors)
│   └── step() -> simulacao
│
└── Braitenberg Matrix (8x2)
    └── Mapeia sensores -> velocidades dos motores
```

## Como Integrar com Seu Treinamento

```python
# train.py
from controllers.controller_project.robot_env import EPuckEnv
from stable_baselines3 import PPO

env = EPuckEnv()  # robot_env.py ja tem estrutura pronta

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("epuck_model")
```
