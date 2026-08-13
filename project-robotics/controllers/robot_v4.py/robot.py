
from enum import Enum
from typing import List
from controller import Supervisor, DistanceSensor, Motor

import numpy as np

class Movimento(Enum):
    FRONT = 1
    TURN_LEFT = 2
    TUNR_RIGHT = 3
    FLIP = 4 

class Robot:
    def __init__(self, limit_dist = 3.5, pesos = None, fastmode = False, timestep = 64):
        
        # Hiperparametros
        self.limit_dist = limit_dist
        
        # FSM
        self.moviment = Movimento.FRONT

        # Moving speeds
        self.left_speed = 0.0
        self.right_speed = 0.0

        # pesos dos sensores
        if pesos is None:
            self.pesos = [1. , 1., 1., 1., 1., 1., 1., 1.]
        else:
            self.pesos = pesos
        
        # Configurações do simulador
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep()) if timestep == 64 else int(timestep)

        if fastmode:
            self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        else:
            self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)

        # Robô E-puck
        self.max_speed = 6.28 # rad/s
        self.num_sensors = 8
        self.sensor_range = 4095

        # Inicializar sensores
        self.sensor = []
        self.init_sensors()

        # Inicializa
        self.left_motor = None
        self.right_motor = None
        self.init_motors()



    def init_sensors(self):
        sensor_names = [f"ps{i}" for i in range(self.num_sensors)]

        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            
            if sensor is None:
                raise RuntimeError(f"Could not find sensor '{name}' on e-puck robot")
            
            # Cada sensor precisa ser habilitado antes de fornecer leituras válidas.
            sensor.enable(self.time_step)
            
            self.sensors.append(sensor)

    def init_motors(self):
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")

        if self.left_motor is None or self.right_motor is None:
            raise RuntimeError("Could not find motor devices on e-puck robot")

        # float('inf') desativa o controle por posição e libera o modo velocidade.
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Começa parado para evitar movimento inesperado ao carregar o controller.
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def init_robot(timestep = 64, fastmode = False):
        
        return Robot(timestep=timestep, fastmode=fastmode)

    def calibration_sensors():
        pass

    def get_sensors():
        pass

    def set_motor_speeds(self, left_speed = 0. , right_speed = 0.):
        
        left_velocity = left_speed * self.max_speed
        right_velocity = right_speed * self.max_speed

        self.left_motor.setVelocity(left_velocity)
        self.right_motor.setVelocity(right_velocity)

    def get_custo():
        cost = 0.

        pass

    def get_z():
        pass

    def fsm(self,):
        pass

    def apply_moviment(self):
        pass

    def reset_position(self):
        robot_node = self.robot.getSelf()

        # Mantém o robô sempre nascendo no centro da arena (0,0)
        rand_x = 0.0
        rand_y = 0.0

        # Redefine a posição (z é a altura do e-puck no webots)
        translation_field = robot_node.getField("translation")
        translation_field.setSFVec3f([rand_x, rand_y, 0.0])

        # Gera rotação aleatória ao redor do eixo Z (0 a 2*PI)
        rand_angle = np.random.uniform(0, 2 * np.pi)

        # Redefine a orientação usando a rotação aleatória
        rotation_field = robot_node.getField("rotation")
        rotation_field.setSFRotation([0.0, 0.0, 1.0, rand_angle])

        # Reseta a física para zerar velocidades lineares e angulares residuais
        robot_node.resetPhysics()

    def get_position(self):
        robot_node = self.robot.getSelf()
        position = robot_node.getPosition()
        
        # Webots retorna [x, y, z], mas para arena 2D usamos x e y
        return (position[0], position[1])

    def get_timestep(self):
        return self.timestep








