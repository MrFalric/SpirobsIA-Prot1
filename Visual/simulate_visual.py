import pygame
import numpy as np
import random
import math
import time
import tensorflow as tf
from collections import deque

# ====== PARAMETROS ======
# Definicion de variables estructurales y geometricas del entorno de simulacion
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

# Propiedades mecanicas de los cables y limites de tension aplicables
MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

# Parametros de colision del objetivo y rango limite del brazo robotico
OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

# Dimensiones de la matriz de imagen y cantidad de cuadros apilados para la red convolucional
IMG_SIZE = 84
FRAME_STACK = 4

# ====== COLORES ======
# Valores en formato RGB utilizados para el renderizado de la interfaz grafica
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (80, 80, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
ORANGE = (255, 165, 0)

# Inicializacion del motor grafico Pygame y configuracion de la ventana
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulacion Tentaculo - IA Visual + Tension")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# ====== MODELO ======
# Carga de la red neuronal convolucional multimodal h5 omitiendo la etapa de compilacion inicial
model = tf.keras.models.load_model("tentacle_visual_tension_dqn.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
    loss='mse'
)

print("Modelo visual + tension cargado correctamente")

# ====== ENTORNO ======
class Segment:
    """
    Estructura basica de datos para almacenar la longitud, anchura y orientacion
    angular individual de cada eslabon constituyente del tentaculo.
    """
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    """
    Clase contenedora que gestiona la arquitectura de eslabones y simula
    las reacciones cinematicas provocadas por los cambios de tension en los cables.
    """
    def __init__(self):
        # Distribuye las dimensiones decrecientes de cada modulo desde la base hasta el extremo
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]

        # Valores de inicio de tension para calibrar el equilibrio de fuerzas estaticas
        self.left_tension = 1.0
        self.right_tension = 1.0

    def apply_action(self, action):
        """
        Incrementa la tension de un cable lateral mientras reduce simultaneamente el opuesto,
        manteniendo los resultados estrictamente dentro de los rangos operativos permitidos.
        """
        if action == 0:
            self.left_tension += TENSION_STEP
            self.right_tension -= TENSION_STEP / 2
        elif action == 1:
            self.right_tension += TENSION_STEP
            self.left_tension -= TENSION_STEP / 2

        self.left_tension = np.clip(self.left_tension, 0.5, MAX_TENSION)
        self.right_tension = np.clip(self.right_tension, 0.5, MAX_TENSION)

    def update_angles(self):
        """
        Aplica la diferencia neta de tensiones a las articulaciones consecutivas.
        El efecto rotacional de la fuerza se intensifica en relacion directa con la distancia a la base.
        """
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * 0.01
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        """
        Resuelve la ecuacion de cinematica directa recorriendo la cadena de modulos articulados.
        Entrega una lista ordenada de las coordenadas geometricas absolutas de cada nodo.
        """
        positions = [BASE_POS.copy()]
        current_angle = BASE_ANGLE
        for seg in self.segments:
            current_angle += seg.angle
            dx = seg.length * np.cos(current_angle)
            dy = seg.length * np.sin(current_angle)
            positions.append(positions[-1] + np.array([dx, dy]))
        return positions

    def draw(self, surface):
        """
        Dibuja los contornos poligonales tridimensionales simulados para cada eslabon.
        Traza lineas graficas externas adicionales para senalar visualmente el cable izquierdo (azul) y derecho (rojo).
        """
        positions = self.compute_positions()
        left_points = []
        right_points = []

        for i in range(len(self.segments)):
            p1 = positions[i]
            p2 = positions[i + 1]
            seg = self.segments[i]

            direction = p2 - p1
            norm = direction / np.linalg.norm(direction)
            perp = np.array([-norm[1], norm[0]])
            w = seg.width / 2

            left_points.append(p1 - perp * w)
            right_points.append(p1 + perp * w)

            if i == len(self.segments) - 1:
                left_points.append(p2 - perp * w)
                right_points.append(p2 + perp * w)

            l = seg.length / 2
            points = [
                p1 - perp * w,
                p1 + perp * w,
                p1 + norm * l + perp * (w / 0.6),
                p2 + perp * w,
                p2 - perp * w,
                p1 + norm * l - perp * (w / 0.5),
            ]

            center = (p1 + p2) / 2
            rot_right = np.array([[0, 1], [-1, 0]])
            points = [center + rot_right @ (pt - center) for pt in points]

            pygame.draw.polygon(surface, GREEN, points)

        pygame.draw.lines(surface, BLUE, False, left_points, 2)
        pygame.draw.lines(surface, RED, False, right_points, 2)

        return positions

    def get_tip(self):
        """
        Extrae y devuelve el ultimo vector del arreglo posicional correspondente a la punta del brazo.
        """
        return self.compute_positions()[-1]


# ====== FUNCIONES ======
def spawn_object():
    """
    Selecciona de forma aleatoria coordenadas polares controladas para generar el objeto
    dentro de un area espacial aproximada donde el tentaculo tenga la capacidad fisica de interactuar.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.7)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y])


# ====== RENDER IA ======
def render_image(tentacle, obj):
    """
    Construye una matriz de baja resolucion (84x84) que emula la entrada de una camara digital.
    Escala y rasteriza de forma geometrica los segmentos del tentaculo y el punto del objetivo,
    normalizando los pixeles ocupados a un valor binario flotante de 1.0 para el procesamiento de la IA.
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def scale(p):
        x = int(p[0] / WIDTH * IMG_SIZE)
        y = int(p[1] / HEIGHT * IMG_SIZE)
        x = np.clip(x, 0, IMG_SIZE - 1)
        y = np.clip(y, 0, IMG_SIZE - 1)
        return np.array([x, y])

    positions = tentacle.compute_positions()

    for i in range(len(positions) - 1):
        p1 = scale(positions[i])
        p2 = scale(positions[i + 1])

        xs = np.linspace(p1[0], p2[0], 10).astype(int)
        xs = np.linspace(p1[0], p2[0], 10).astype(int)
        ys = np.linspace(p1[1], p2[1], 10).astype(int)

        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = 1.0

    obj_p = scale(obj)
    x, y = obj_p
    img[max(0,y-2):min(IMG_SIZE,y+2), max(0,x-2):min(IMG_SIZE,x+2)] = 1.0

    return img


def stack_frames(frames, new_frame):
    """
    Mantiene una estructura de datos de tipo cola circular conteniendo las capturas mas recientes.
    Agrupa los cuadros secuenciales a traves del eje de profundidad para proveer informacion temporal
    de velocidad y sentido de movimiento a la red neuronal convolucional.
    """
    frames.append(new_frame)
    if len(frames) < FRAME_STACK:
        while len(frames) < FRAME_STACK:
            frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MAIN ======
def main():
    """
    Algoritmo central de ejecucion continua de la aplicacion grafica interactiva.
    Calcula de forma sincrona los estados del entorno, invoca la inferencia de la red neuronal,
    aplica las directrices fisicas de movimiento y evalua colisiones geometricas por tramos.
    """
    tentacle = CableTentacle()
    target = spawn_object()