import numpy as np
import random
import pickle
import math

# Configuracion estructural base identica para mantener correspondencia dinamica
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

# Constantes del modelo de cables y del espacio de maniobra
MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

# Hiperparametros del algoritmo de Aprendizaje por Refuerzo Q-Learning
LEARNING_RATE = 0.18
DISCOUNT = 0.97
EPISODES = 1000

# Parametros para la estrategia de exploracion Epsilon-Greedy
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.02


class Segment:
    def __init__(self, length, width):
        """
        Define el componente basico del tentaculo para calculos kinematicos del modelo sin graficos.
        """
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    def __init__(self):
        """
        Instancia los arreglos de dimensionamiento y las variables de tension interna para el aprendizaje.
        """
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]
        self.left_tension = 0.0
        self.right_tension = 0.0

    def update_angles(self):
        """
        Modifica la orientacion angular de cada pieza segun el estres neto del sistema.
        """
        SMOOTH_FORCE = CABLE_FORCE_FACTOR * 0.25
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * SMOOTH_FORCE
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        """
        Transforma los angulos internos a coordenadas cartesianas globales de forma secuencial.
        """
        positions = [BASE_POS.astype(float)]
        current_angle = BASE_ANGLE
        for seg in self.segments:
            current_angle += seg.angle
            dx = seg.length * np.cos(current_angle)
            dy = seg.length * np.sin(current_angle)
            positions.append(positions[-1] + np.array([dx, dy]))
        return positions

    def apply_action(self, action):
        """
        Ejecuta las modificaciones de tension e inercia asociadas con las opciones discretas de control.
        """
        if action == 0:
            self.left_tension = min(MAX_TENSION, self.left_tension + TENSION_STEP)
            self.right_tension = max(0.0, self.right_tension - TENSION_STEP / 2)
        elif action == 1:
            self.right_tension = min(MAX_TENSION, self.right_tension + TENSION_STEP)
            self.left_tension = max(0.0, self.left_tension - TENSION_STEP / 2)

        self.left_tension = max(0.0, self.left_tension - TENSION_STEP / 8)
        self.right_tension = max(0.0, self.right_tension - TENSION_STEP / 8)


def spawn_object():
    """
    Genera posiciones de objetivos de entrenamiento distribuidos aleatoriamente dentro del area util.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.8)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y], dtype=float)


def tip_distance(tentacle, obj_pos):
    """
    Mide la distancia lineal directa euclidiana entre la punta del brazo y el objetivo.
    """
    return np.linalg.norm(tentacle.compute_positions()[-1] - obj_pos)


def get_tip_angle(tentacle):
    """
    Calcula el angulo acumulativo absoluto en radianes que presenta el extremo final.
    """
    return BASE_ANGLE + sum(seg.angle for seg in tentacle.segments)


def discretize_angle(angle):
    """
    Agrupa orientaciones continuas en indices discretos fijos para el modelado de estados.
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return int(((angle + np.pi) / (2 * np.pi)) * 8) % 8


def discretize_tension(T):
    """
    Cuantiza los niveles de tension de los tensores para indexacion estructurada.
    """
    return int(np.clip(T / (MAX_TENSION / 3), 0, 2))


def get_state(tentacle, obj_pos):
    """
    Construye la tupla descriptiva que actua como clave del diccionario en la Q-Table.
    """
    tip = tentacle.compute_positions()[-1]
    dx, dy = obj_pos - tip
    dist = np.hypot(dx, dy)
    dist_bin = int(np.clip(dist / (MAX_REACH / 5), 0, 4))

    tip_angle = get_tip_angle(tentacle)
    target_angle = math.atan2(dy, dx)
    diff = (target_angle - tip_angle + np.pi) % (2 * np.pi) - np.pi
    dir_bin = int(((diff + np.pi) / (2 * np.pi)) * 8) % 8

    angle_bin = discretize_angle(tip_angle)
    tl = discretize_tension(tentacle.left_tension)
    tr = discretize_tension(tentacle.right_tension)

    return (dist_bin, dir_bin, angle_bin, tl, tr)


def train():
    """
    Proceso central de optimizacion iterativa por refuerzo.
    Simula miles de pasos distribuidos en episodios donde premia los acercamientos directos al
    objetivo, penaliza la inactividad motora o alejamientos, y actualiza la ecuacion de Bellman discreta.
    """
    q = {}
    epsilon = EPSILON

    for ep in range(EPISODES):
        tentacle = CableTentacle()
        obj = spawn_object()

        state = get_state(tentacle, obj)
        prev_dist = tip_distance(tentacle, obj)
        total_reward = 0

        for step in range(450):
            if state not in q:
                q[state] = np.zeros(2)

            # Seleccion exploratoria o de explotacion basada en la tasa Epsilon actual
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                action = int(np.argmax(q[state]))

            tentacle.apply_action(action)
            tentacle.update_angles()

            dist = tip_distance(tentacle, obj)

            # Calculo adaptativo del sistema de recompensas ponderadas
            reward = (prev_dist - dist) * 80
            reward -= 0.2

            if dist >= prev_dist:
                reward -= 2

            if tentacle.left_tension < 0.05 and tentacle.right_tension < 0.05:
                reward -= 1

            tip_angle = get_tip_angle(tentacle)
            target_angle = math.atan2(
                obj[1] - tentacle.compute_positions()[-1][1],
                obj[0] - tentacle.compute_positions()[-1][0]
            )
            ang_diff = abs(((target_angle - tip_angle + np.pi) % (2 * np.pi)) - np.pi)
            reward += (np.pi - ang_diff) * 0.25

            # Condicion de exito si el extremo intercepta los limites radiales del objetivo
            if dist < OBJECT_RADIUS + 10:
                reward += 3500
                done = True
            else:
                done = False

            next_state = get_state(tentacle, obj)
            if next_state not in q:
                q[next_state] = np.zeros(2)

            # Aplicacion de la formula fundamental de actualizacion de Q-Learning
            q[state][action] += LEARNING_RATE * (
                reward + DISCOUNT * np.max(q[next_state]) - q[state][action]
            )

            prev_dist = dist
            total_reward += reward
            state = next_state

            if done:
                break

        # Reduccion logaritmica del parametro de exploracion
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if ep % 200 == 0:
            print(f"Ep {ep} | R {total_reward:.1f} | eps {epsilon:.3f}")

    # Exportacion del conocimiento consolidado a un archivo Pickle binario permanente
    with open("tentacle_q.pkl", "wb") as f:
        pickle.dump(q, f)

    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    train()