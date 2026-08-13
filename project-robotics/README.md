# Project Robotics - Python Webots Controller

Este projeto implementa o controle de um robô e-puck no Webots usando Python e aprendizado por reforço (RL).

## Pré-requisitos

- Python 3.8+
- Webots R2025a ou superior (instalado no seu sistema)
- git (para versionamento)

## Setup com venv (Recomendado)

### 1. Acessar o repositório

```bash
cd ~/github_projects/intelligent_systems/project-robotics
```

### 2. Criar e ativar o virtual environment

No Linux/macOS:
```bash
chmod +x scripts/setup_venv.sh
scripts/setup_venv.sh
source robotics/bin/activate
```

No Windows (PowerShell):
```powershell
python -m venv robotics
robotics\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r docs/requirements.txt
```

### 3. Verificar instalação

```bash
python -c "from controller import Robot; print('OK - Webots API')"
python -c "import gymnasium; print('OK - Gymnasium')"
python -c "import numpy; print('OK - NumPy')"
```

## Estrutura do Projeto

```bash
project-robotics/
├── controllers/
│   ├── controller_project/
│   │   ├── config.yaml           # Configuracoes do ambiente
│   │   ├── locomotion.py         # Controle do robo (Python)
│   │   ├── robot_env.py          # Ambiente gym.Env
│   │   ├── rewards.py            # Funcao de recompensa
│   │   ├── train.py              # Script de treinamento
│   │   ├── test_model.py         # Script de teste
│   │   ├── examples.py           # Exemplos praticos
├── docs/
│   ├── COMECE_AQUI.md            # Guia de inicio rapido
│   ├── SETUP_RESUMO.md           # Resumo de setup
│   ├── WEBOTS_SETUP.md           # Guia de setup Webots R2025a
│   └── requirements.txt          # Dependencias Python
├── scripts/
│   ├── setup_venv.sh             # Script de setup do ambiente Python
│   └── setup_webots.sh           # Script de configuracao do Webots
├── worlds/
│   └── mundo1.wbt                # Mundo Webots
├── robotics/                     # Virtual environment
├── README.md                      # Este arquivo
└── .gitignore                     # Configuracao Git
```

## Como Usar

### 1. Controle Básico do Robô

```python
from controllers.controller_project.locomotion import initialize_robot

# Inicializar robo
locomotion = initialize_robot()

# Ler sensores (valores normalizados 0-1)
sensors = locomotion.read_sensors()
print(f"Sensores: {sensors}")

# Controlar motores (velocidade -1.0 a 1.0)
left_speed = 0.5   # 50% da velocidade maxima
right_speed = 0.5
locomotion.set_motor_speeds(left_speed, right_speed)

# Avancar simulacao um passo
locomotion.step()
```

### 2. Algoritmo de Braitenberg (Obstacle Avoidance)

```python
from controllers.controller_project.locomotion import initialize_robot

locomotion = initialize_robot()

for _ in range(1000):
    sensors = locomotion.read_sensors()
    locomotion.apply_braitenberg_control(sensors)
    locomotion.step()
```

### 3. Treinamento com RL (Controlador Externo)

O treinamento por Aprendizado por Reforço utiliza o modo **controlador externo** (`<extern>`) do Webots. Nesse modo, o Webots abre uma porta TCP e aguarda que um script Python externo se conecte para assumir o controle do robô.

#### Passo 1: Configurar o mundo do Webots

No arquivo `worlds/mundo1.wbt`, o nó do `E-puck` deve estar configurado assim:

```
E-puck {
  controller "<extern>"
  supervisor TRUE
}
```

- **`controller "<extern>"`**: Faz o Webots aguardar uma conexão externa em vez de rodar um script automaticamente.
- **`supervisor TRUE`**: Permite que o script Python use a API de Supervisor para reposicionar o robô entre episódios de treinamento.

#### Passo 2: Iniciar a simulação no Webots

Abra o Webots, carregue o mundo `mundo1.wbt` e clique em **Play**. Você verá a mensagem:

```
INFO: 'e-puck' extern controller: Waiting for local or remote connection on port 1234...
```

Isso significa que o simulador está pronto e aguardando o seu script.

#### Passo 3: Executar o treinamento

Em um **novo terminal**, navegue até a pasta do controlador e execute:

```bash
cd controllers/controller_project/
WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:1234" ../../robotics/bin/python train.py
```

> **Nota:** A variável `WEBOTS_CONTROLLER_URL` informa ao script em qual porta o Webots está escutando. O `WEBOTS_HOME` já é configurado automaticamente pelo `train.py`.

### 4. Testar modelo treinado

Após o treinamento gerar o arquivo `.zip` com os pesos, repita o processo acima mas com o script de teste:

```bash
cd controllers/controller_project/
WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:1234" ../../robotics/bin/python test_model.py
```

## Especificacoes do E-puck

| Parametro | Valor |
|-----------|-------|
| Sensores | 8 sensores infravermelhos |
| Motores | 2 motores DC (roda esquerda/direita) |
| Velocidade maxima | 6.28 rad/s |
| Passo de simulacao | 64 ms (padrao) |
| Algoritmo de controle | Braitenberg (Matriz 8x2) |

## Configuracao

Edite `config.yaml` para ajustar:

- Numero maximo de passos por episodio
- Numero de sensores
- Velocidade maxima
- Parametros de treinamento

## Troubleshooting

### `ModuleNotFoundError: No module named 'controller'`

O módulo `controller` faz parte da instalação do Webots e **não é instalado via pip**. O Python precisa saber onde encontrá-lo.

**Solução permanente** (registrar no venv):
```bash
robotics/bin/python -c "import site; open(site.getsitepackages()[0] + '/webots.pth', 'w').write('/usr/local/webots/lib/controller/python\n')"
```

**Solução manual** (por sessão de terminal):
```bash
export PYTHONPATH="/usr/local/webots/lib/controller/python:$PYTHONPATH"
```

### `KeyError: 'WEBOTS_HOME'`

A biblioteca nativa do Webots precisa da variável `WEBOTS_HOME` para localizar os arquivos `.so` (bibliotecas C).

**Solução:** O `train.py` e o `test_model.py` já configuram essa variável automaticamente via `os.environ.setdefault("WEBOTS_HOME", "/usr/local/webots")`. Se por algum motivo o Webots estiver instalado em outro local, defina manualmente:
```bash
export WEBOTS_HOME=/caminho/para/webots
```

### `ModuleNotFoundError: No module named 'controllers'`

Isso acontece quando os imports usam caminhos absolutos (ex: `from controllers.controller_project.robot_env import ...`) mas o script é executado de dentro da pasta `controllers/controller_project/`.

**Solução:** Os imports já foram corrigidos para usar caminhos relativos (ex: `from robot_env import EPuckEnv`). Sempre execute os scripts de dentro da pasta do controlador.

### O robô não reseta a posição entre episódios

Para que o robô seja teleportado de volta ao centro da arena, o nó `E-puck` no arquivo `.wbt` **precisa** ter `supervisor TRUE`. Sem essa flag, a API de Supervisor não funciona e o `reset()` não consegue reposicionar o robô.

### comando para ativar o venv
 source robotics/bin/activate
 WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:1234" python train.py
## Referencias

- [Webots Documentation](https://cyberbotics.com/doc/reference/introduction)
- [Webots Python API](https://cyberbotics.com/doc/reference/python)
- [Webots External Controllers](https://cyberbotics.com/doc/guide/running-extern-robot-controllers)
- [E-puck Robot](https://www.e-puck.org/)
- [Braitenberg Vehicles](https://en.wikipedia.org/wiki/Braitenberg_vehicle)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3 (PPO)](https://stable-baselines3.readthedocs.io/)
