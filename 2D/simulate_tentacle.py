import pygame
import numpy as np
import pickle
import random
import math
import time

# Configuracion de dimensiones de la simulacion y propiedades de los segmentos
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

# Parametros fisicos y limites de tension en los cables de control
MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

# Tamano del objetivo y alcance efectivo maximo calculado
OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

# Definicion de colores para la interfaz grafica en formato RGB
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (80, 80, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
ORANGE = (255, 165, 0)

# Inicializacion de los modulos de Pygame para el renderizado de la ventana y fuentes
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación Tentáculo - IA")
font = pygame.font.SysFont("Arial", 20)

class Segment:
    def __init__(self, length, width):
        """
        Representa un eslabon individual del tentaculo rigido articulado.
        Almacena su longitud, grosor y el angulo relativo respecto a su predecesor.
        """
        self.length = length
        self.width = width
        self.angle = 0.0

class CableTentacle:
    def __init__(self):
        """
        Inicializa la estructura del tentaculo con una reduccion gradual
        de longitud y anchura desde la base hasta la punta. Almacena las tensiones internas.
        """
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]
        self.left_tension = 0.0
        self.right_tension = 0.0

    def apply_action(self, action):
        """
        Modifica los niveles de tension de los cables izquierdo o derecho
        segun la accion tomada por el agente. Tambien aplica una atenuacion o
        friccion constante que reduce la tension de forma natural en cada paso.
        """
        if action == 0:
            self.left_tension = min(MAX_TENSION, self.left_tension + TENSION_STEP)
            self.right_tension = max(0.0, self.right_tension - TENSION_STEP / 2)
        elif action == 1:
            self.right_tension = min(MAX_TENSION, self.right_tension + TENSION_STEP)
            self.left_tension = max(0.0, self.left_tension - TENSION_STEP / 2)

        self.left_tension = max(0.0, self.left_tension - TENSION_STEP / 8)
        self.right_tension = max(0.0, self.right_tension - TENSION_STEP / 8)

    def update_angles(self):
        """
        Calcula la diferencia de tensiones entre cables para modificar el angulo de
        cada segmento de forma acumulativa, simulando la flexibilidad del mecanismo.
        Mantiene los limites de rotacion para evitar deformaciones no realistas.
        """
        SMOOTH_FORCE = CABLE_FORCE_FACTOR * 0.25
        diff = self.right_tension - self.left_tension

        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * SMOOTH_FORCE
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        """
        Calcula la posicion cartesiana X e Y de cada articulacion en el espacio
        utilizando trigonometria directa acumulada desde la base fija.
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
        Dibuja los segmentos poligonales verdes en la pantalla y genera el
        contorno de los cables laterales de tension, cambiando la intensidad
        del color azul o rojo segun el esfuerzo aplicado en el tendon.
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
        if self.left_tension > 0:
            pygame.draw.lines(surface, BLUE, False, left_points, 2)
        else:
            pygame.draw.lines(surface, (50, 50, 200), False, left_points, 1)
        if self.right_tension > 0:
            pygame.draw.lines(surface, RED, False, right_points, 2)
        else:
            pygame.draw.lines(surface, (200, 50, 50), False, right_points, 1)
        return positions

    def get_tip(self):
        """
        Obtiene la posicion espacial del ultimo extremo libre del tentaculo.
        """
        return self.compute_positions()[-1]

def get_tip_angle(t):
    """
    Calcula la orientacion angular absoluta de la punta del tentaculo
    sumando las desviaciones parciales de todos sus componentes interiores.
    """
    return BASE_ANGLE + sum(seg.angle for seg in t.segments)

def discretize_angle(angle):
    """
    Normaliza y divide un angulo continuo en una de las 8 direcciones posibles
    para simplificar la lectura del espacio de estados del algoritmo de IA.
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return int(((angle + np.pi) / (2 * np.pi)) * 8) % 8

def discretize_tension(T):
    """
    Reduce un valor continuo de tension mecanica a un entero discreto entre
    0 y 2 para optimizar las dimensiones y busquedas en la tabla Q.
    """
    return int(np.clip(T / (MAX_TENSION / 3), 0, 2))

def spawn_object():
    """
    Genera una posicion aleatoria para el objetivo dentro del rango visualizable
    y los limites fisicos de estiramiento del brazo roboticoflexible.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.7)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y])

def get_state(t, obj_pos):
    """
    Extrae la representacion compacta y codificada de la situacion actual del entorno.
    Incluye distancia al objetivo, direccion relativa del objetivo respecto a la punta,
    angulo de la punta y tensiones en ambos cables para indexar la tabla Q.
    """
    pos = t.get_tip()
    dx, dy = obj_pos - pos
    dist = np.hypot(dx, dy)
    dist_bin = int(np.clip(dist / (MAX_REACH / 5), 0, 4))
    tip_angle = get_tip_angle(t)
    ang_obj = math.atan2(dy, dx)
    diff = (ang_obj - tip_angle + np.pi) % (2 * np.pi) - np.pi
    dir_bin = int(((diff + np.pi) / (2 * np.pi)) * 8) % 8
    angle_bin = discretize_angle(tip_angle)
    tl = discretize_tension(t.left_tension)
    tr = discretize_tension(t.right_tension)
    return (dist_bin, dir_bin, angle_bin, tl, tr)

# Bloque de lectura del archivo serializado que contiene el cerebro entrenado por la IA
try:
    with open("tentacle_q.pkl", "rb") as f:
        Q = pickle.load(f)
    print("Q-table cargada.")
except:
    print("No se encontró tentacle_q.pkl — entrena primero.")
    Q = {}

def main():
    """
    Bucle principal de la interfaz interactiva. Ejecuta la lectura del modelo de IA,
    comprueba los eventos del sistema, calcula colisiones fisicas contra el objetivo
    y gestiona los contadores de rendimiento en pantalla junto con los reinicios por tiempo.
    """
    tentacle = CableTentacle()
    target = spawn_object()
    clock = pygame.time.Clock()

    captured_count = 0
    missed_count = 0

    start_time = time.time()   # Registra el momento exacto de inicio del intento actual

    running = True
    while running:
        screen.fill(BLACK)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Comprobacion de limite de tiempo por intento (15 segundos)
        if time.time() - start_time >= 15:
            missed_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()

        # Seleccion de la mejor accion disponible en base a la politica del agente IA
        state = get_state(tentacle, target)
        if state in Q:
            action = int(np.argmax(Q[state]))
        else:
            action = random.choice([0, 1])

        # Actualizacion fisica de los segmentos
        tentacle.apply_action(action)
        tentacle.update_angles()

        # Dibujado del modelo y del objetivo estatico
        positions = tentacle.draw(screen)
        pygame.draw.circle(screen, YELLOW, target.astype(int), OBJECT_RADIUS)

        # Calculo geometrico avanzado de colision por segmentos de linea recta
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

        # Si el tentaculo toca el circulo amarillo se marca captura exitosa y se reubica el objetivo
        if collision:
            pygame.draw.circle(screen, ORANGE, closest.astype(int), 6)
            captured_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()   # Reiniciar temporizador de seguridad

        time_left = max(0, 15 - int(time.time() - start_time))

        # Texto informativo estructurado para visualizacion del usuario
        info = [
            f"Capturados: {captured_count}",
            f"No capturados: {missed_count}",
            f"Tiempo restante: {time_left}s",
            f"Accion: {['Izquierda', 'Derecha'][action]}",
            "Cerrar ventana para salir."
        ]

        for i, txt in enumerate(info):
            screen.blit(font.render(txt, True, WHITE), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # Redundancia estructural o duplicacion del bucle de simulacion sin temporizador estricto
    tentacle = CableTentacle()
    target = spawn_object()
    clock = pygame.time.Clock()
    captured_count = 0
    running = True
    while running:
        screen.fill(BLACK)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        state = get_state(tentacle, target)
        if state in Q:
            action = int(np.argmax(Q[state]))
        else:
            action = random.choice([0, 1])
        tentacle.apply_action(action)
        tentacle.update_angles()
        positions = tentacle.draw(screen)
        pygame.draw.circle(screen, YELLOW, target.astype(int), OBJECT_RADIUS)
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
        if collision:
            pygame.draw.circle(screen, ORANGE, closest.astype(int), 6)
            captured_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
        info = [
            f"Objetos capturados: {captured_count}",
            f"Accion: {['Izquierda', 'Derecha'][action]}",
            "Cerrar ventana para salir."
        ]
        for i, txt in enumerate(info):
            screen.blit(font.render(txt, True, WHITE), (20, 20 + i * 25))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()