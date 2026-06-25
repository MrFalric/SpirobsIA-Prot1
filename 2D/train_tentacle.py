import numpy as np
import random
import pickle
import math

# Configuracion de parametros globales del entorno mecanico
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

# Variables fisicas que determinan el control de tension de los cables artificiales
MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

# Tamaño del objetivo y limite maximo teorico de extension del brazo robotico
OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

# Hiperparametros del algoritmo de Aprendizaje por Refuerzo (Q-Learning)
LEARNING_RATE = 0.18
DISCOUNT = 0.97
EPISODES = 1000

# Configuracion de la estrategia de exploracion Epsilon-Greedy
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.02


class Segment:
    """
    Representa una articulacion o modulo individual dentro del tentaculo.
    Almacena dimensiones base y el angulo de rotacion actual en radianes.
    """
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    """
    Simula la estructura cinematica del tentaculo controlado por cables.
    Administra los arreglos de eslabones y calcula el efecto de las fuerzas de tension.
    """
    def __init__(self):
        # Genera dimensiones de segmentos que se reducen de forma lineal hacia el extremo
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]
        self.left_tension = 0.0
        self.right_tension = 0.0

    def update_angles(self):
        """
        Calcula y actualiza los angulos acumulativos de cada seccion del robot.
        La fuerza ejercida por la diferencia de tension afecta mas a los eslabones distales.
        """
        SMOOTH_FORCE = CABLE_FORCE_FACTOR * 0.25
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * SMOOTH_FORCE
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        """
        Resuelve la cinematica directa calculando las coordenadas espaciales X e Y
        de todos los vertices de los segmentos a partir de la base.
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
        Modifica los valores continuos de tension de los cables segun el comando de la IA.
        Aplica disminuciones simultaneas para simular perdida dinamica por friccion o soltado.
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
    Determina de manera aleatoria una posicion espacial valida dentro del area de alcance
    para situar el objeto que el tentaculo debe aprender a tocar.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.8)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y], dtype=float)


def tip_distance(tentacle, obj_pos):
    """
    Calcula la distancia euclidiana en pixeles desde el extremo final del brazo hasta el objetivo.
    """
    return np.linalg.norm(tentacle.compute_positions()[-1] - obj_pos)


def get_tip_angle(tentacle):
    """
    Determina la orientacion final acumulada en la punta del tentaculo en radianes.
    """
    return BASE_ANGLE + sum(seg.angle for seg in tentacle.segments)


def discretize_angle(angle):
    """
    Ajusta y divide un angulo continuo en una de las 8 zonas discretas posibles.
    Reduce la complejidad del entorno para facilitar el mapeo de estados del agente.
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return int(((angle + np.pi) / (2 * np.pi)) * 8) % 8


def discretize_tension(T):
    """
    Transforma el valor decimal de tension de un cable en 3 niveles discretos fijos.
    """
    return int(np.clip(T / (MAX_TENSION / 3), 0, 2))


def get_state(tentacle, obj_pos):
    """
    Construye la representacion discreta o firma del estado actual del sistema.
    Combina rangos de distancia, direccion relativa hacia el objeto, orientacion
    de la punta del brazo y las tensiones estimadas de ambos lados.
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
    Ejecuta el ciclo global de optimizacion por Q-Learning.
    Itera a traves de multiples episodios donde el agente interactua con el entorno,
    recibe castigos o incentivos numericos y actualiza la tabla de valores Q.
    Guarda la estructura resultante en un archivo serializado al concluir.
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
            # Inicializacion dinamica de los estados no registrados previamente en la tabla Q
            if state not in q:
                q[state] = np.zeros(2)

            # Seleccion de accion basada en politica Epsilon-Greedy (Exploracion vs Explotacion)
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                action = int(np.argmax(q[state]))

            tentacle.apply_action(action)
            tentacle.update_angles()

            dist = tip_distance(tentacle, obj)

            # Sistema de recompensas basado en la aproximacion geometrica al objetivo
            reward = (prev_dist - dist) * 80
            reward -= 0.2  # Costo por paso temporal transcurrido

            # Penalizacion si el movimiento provoco que se alejara de la meta
            if dist >= prev_dist:
                reward -= 2

            # Penalizacion por inactividad o flacidez extrema en la estructura de cables
            if tentacle.left_tension < 0.05 and tentacle.right_tension < 0.05:
                reward -= 1

            # Incentivo por alineacion angular correcta de la punta apuntando al objetivo
            tip_angle = get_tip_angle(tentacle)
            target_angle = math.atan2(
                obj[1] - tentacle.compute_positions()[-1][1],
                obj[0] - tentacle.compute_positions()[-1][0]
            )
            ang_diff = abs(((target_angle - tip_angle + np.pi) % (2 * np.pi)) - np.pi)
            reward += (np.pi - ang_diff) * 0.25

            # Condicion de exito: el tentaculo logro hacer contacto con el radio del objeto
            if dist < OBJECT_RADIUS + 10:
                reward += 3500
                done = True
            else:
                done = False

            next_state = get_state(tentacle, obj)
            if next_state not in q:
                q[next_state] = np.zeros(2)

            # Aplicacion de la formula de actualizacion de Bellman para diferencias temporales
            q[state][action] += LEARNING_RATE * (
                reward + DISCOUNT * np.max(q[next_state]) - q[state][action]
            )

            prev_dist = dist
            total_reward += reward
            state = next_state

            if done:
                break

        # Reduccion progresiva del valor Epsilon para estabilizar el aprendizaje
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Muestra metricas de monitoreo en terminal cada 200 episodios completados
        if ep % 200 == 0:
            print(f"Ep {ep} | R {total_reward:.1f} | eps {epsilon:.3f}")

    # Exportacion de la base de conocimiento entrenada para su posterior uso en simulaciones
    with open("tentacle_q.pkl", "wb") as f:
        pickle.dump(q, f)

    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    train()