import pygame
import numpy as np
import random
import math
import time
import tensorflow as tf
from collections import deque

# ====== PARÁMETROS ======
# Parametros de configuracion fisica para garantizar la sincronizacion exacta con la red entrenada
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

IMG_SIZE = 84
FRAME_STACK = 4

# ====== COLORES ======
# Declaracion de constantes de color en codificacion RGB para Pygame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (80, 80, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
ORANGE = (255, 165, 0)

# Inicializacion del subsistema grafico de video y configuracion de fuentes de texto
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación Tentáculo - IA Visual + Tensión")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# ====== MODELO ======
# Carga de la estructura de capas y pesos guardados en formato H5 deshabilitando compilacion inicial
model = tf.keras.models.load_model("tentacle_visual_tension_dqn.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
    loss='mse'
)

print("Modelo visual + tensión cargado correctamente")

# ====== ENTORNO ======
class Segment:
    def __init__(self, length, width):
        """
        Define la geometría basica de un eslabon articulado.
        """
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    def __init__(self):
        """
        Crea los arreglos dimensionales del brazo y establece las tensiones
        iniciales idénticas a las configuradas durante el proceso de aprendizaje.
        """
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]

        # Ajuste inicial homologado con el script de entrenamiento
        self.left_tension = 1.0
        self.right_tension = 1.0

    def apply_action(self, action):
        """
        Modifica dinamicamente los valores de tension lateral en base a la accion predicha.
        Mantiene los limites operativos mediante funciones de corte.
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
        Ajusta de forma proporcional el angulo de los segmentos mecanicos en relacion
        al desbalance de las fuerzas y aplica limites de rotacion estructural.
        """
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * 0.01
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        """
        Calcula las coordenadas de los nodos de articulacion mediante trigonometria compuesta.
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
        Genera los renderizados geometricos poligonales en pantalla de los elementos verdes
        y dibuja las lineas que identifican el contorno de los tendones artificiales principales.
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
        Retorna la coordenada cartesiana final de la punta extrema del mecanismo.
        """
        return self.compute_positions()[-1]


# ====== FUNCIONES ======
def spawn_object():
    """
    Crea un vector posicion aleatorio para reubicar la meta dentro del campo visual.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.7)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y])


# ====== RENDER IA ======
def render_image(tentacle, obj):
    """
    Construye la matriz reducida de 84x84 pixeles simulando la perspectiva de estado
    que requiere la red convolucional para procesar las imagenes en escala de grises.
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def scale(p):
        """
        Ajusta y traslada puntos del lienzo de visualizacion al tamano de la matriz de la IA.
        """
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
        ys = np.linspace(p1[1], p2[1], 10).astype(int)

        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = 1.0

    obj_p = scale(obj)
    x, y = obj_p
    img[max(0,y-2):min(IMG_SIZE,y+2), max(0,x-2):min(IMG_SIZE,x+2)] = 1.0

    return img


def stack_frames(frames, new_frame):
    """
    Concatena y gestiona el vector dimensional de imagenes consecutivas en el eje de profundidad.
    """
    frames.append(new_frame)
    if len(frames) < FRAME_STACK:
        while len(frames) < FRAME_STACK:
            frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MAIN ======
def main():
    """
    Bucle operativo principal de la simulacion interactiva. Ejecuta predicciones de la red,
    transfiere las matrices e informis de tension al modelo neuronal, valida colisiones fisicas
    contra el objetivo amarillo y actualiza la interfaz grafica en tiempo real a 60 FPS.
    """
    tentacle = CableTentacle()
    target = spawn_object()

    frames = deque(maxlen=FRAME_STACK)
    img = stack_frames(frames, render_image(tentacle, target))

    captured_count = 0
    missed_count = 0
    start_time = time.time()

    running = True
    while running:
        screen.fill(BLACK)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Reinicio automatico e incremento de fallos si se excede el temporizador de 15 segundos
        if time.time() - start_time >= 15:
            missed_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()

        # ====== INPUT MODELO ======
        # Normalizacion del vector de tensiones para coincidir con la escala del entrenamiento
        tension_state = np.array([
            tentacle.left_tension / MAX_TENSION,
            tentacle.right_tension / MAX_TENSION
        ])

        # Consulta directa de inferencia al modelo Deep Q-Network multimodal
        q_values = model.predict(
            [img[np.newaxis], tension_state[np.newaxis]],
            verbose=0
        )

        action = int(np.argmax(q_values[0]))

        tentacle.apply_action(action)
        tentacle.update_angles()

        positions = tentacle.draw(screen)
        pygame.draw.circle(screen, YELLOW, target.astype(int), OBJECT_RADIUS)

        # ====== COLISIÓN ORIGINAL ======
        # Algoritmo de proyeccion geometrica por segmentos para detectar toques del cuerpo del robot
        collision = False
        for i in range(len(positions) - 1):
            a = positions[i]
            b = positions[i + 1]
            seg = b - a
            L = np.linalg.norm(seg)
            if L == 0:
                continue

            seg_dir = seg / L
            proj = np.clip(np.dot(target - a, seg_dir), 0, L)
            closest = a + proj * seg_dir
            d = np.linalg.norm(target - closest)

            if d < OBJECT_RADIUS + tentacle.segments[i].width / 2:
                collision = True
                break

        # Logica de procesamiento en caso de interseccion exitosa contra la meta
        if collision:
            pygame.draw.circle(screen, ORANGE, closest.astype(int), 6)

            # Simulacion de maxima fuerza elastica al hacer contacto, igual que entrenamiento
            tentacle.left_tension = MAX_TENSION
            tentacle.right_tension = MAX_TENSION

            captured_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()

        # ====== NUEVO FRAME ======
        # Captura y apilamiento del cuadro del estado posterior
        img = stack_frames(frames, render_image(tentacle, target))

        # ====== UI ======
        # Renderizado de texto informativo y estadisticas de control en la ventana principal
        time_left = max(0, 15 - int(time.time() - start_time))

        info = [
            f"Capturados: {captured_count}",
            f"No capturados: {missed_count}",
            f"Tiempo restante: {time_left}s",
            f"Accion: {['Izquierda', 'Derecha'][action]}",
        ]

        for i, txt in enumerate(info):
            screen.blit(font.render(txt, True, WHITE), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()