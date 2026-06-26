import numpy as np
import random
import math
import tensorflow as tf
from collections import deque

# ====== PARAMETROS ======
# Configuracion estructural del brazo articulado y dimensiones de la pantalla virtual
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

# Valores limites de control de tensiones para simular los tendones artificiales
MAX_TENSION = 1.5
TENSION_STEP = 0.04
CABLE_FORCE_FACTOR = 0.04

# Variables para determinar el radio del objetivo y el rango limite de estiramiento
OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

# Hiperparametros del entrenamiento DQN Deep Q-Network
EPISODES = 25  # Numero de ciclos completos de ejecucion para el agente

# Parametros de optimizacion de la red y factores de descuento de recompensa temporal
GAMMA = 0.97
LR = 0.00025
BATCH_SIZE = 64
MEMORY_SIZE = 30000
TRAIN_EVERY = 4

# Ajustes para la reduccion progresiva de la tasa de exploracion aleatoria
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

# Dimensiones de la matriz de imagen procesada y cantidad de cuadros apilados en memoria
FRAME_STACK = 4
IMG_SIZE = 84

# ====== ENTORNO ======
class Segment:
    def __init__(self, length, width):
        """
        Representa una seccion rigida individual dentro del tentaculo.
        Guarda las dimensiones fisicas de longitud, grosor y el angulo de giro.
        """
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    def __init__(self):
        """
        Inicializa el cuerpo del tentaculo creando una transicion de eslabones
        que disminuyen su tamano hacia la punta. Establece tensiones base estables.
        """
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]
        self.left_tension = 1.0   # Base constante para dar consistencia estructural
        self.right_tension = 1.0

    def update_angles(self):
        """
        Aplica los cambios en los angulos relacionales de cada tramo calculando
        el diferencial de fuerza mecanica entre ambos tendones y aplicando limites.
        """
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * 0.01
            seg.angle = np.clip(seg.angle, -np.pi/5, np.pi/5)

    def compute_positions(self):
        """
        Calcula las coordenadas geometricas cartesianas consecutivas de todas las
        articulaciones basandose en la sumatoria de angulos trigonometricos.
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
        Modifica el estado de los tensores reduciendo o incrementando sus fuerzas.
        Restringe los valores numericos para mantener la tension en rangos seguros.
        """
        if action == 0:
            self.left_tension += TENSION_STEP
            self.right_tension -= TENSION_STEP / 2
        elif action == 1:
            self.right_tension += TENSION_STEP
            self.left_tension -= TENSION_STEP / 2

        # Acota el estres de los cables para mantener la base cercana a uno
        self.left_tension = np.clip(self.left_tension, 0.5, MAX_TENSION)
        self.right_tension = np.clip(self.right_tension, 0.5, MAX_TENSION)


# ====== UTILIDADES ======
def spawn_object():
    """
    Genera un vector bidimensional aleatorio para situar el objetivo amarillo
    dentro del arco util y del radio de operacion mecanica del brazo.
    """
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.8)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y], dtype=float)


def tip_distance(tentacle, obj):
    """
    Obtiene el modulo o distancia geometrica en linea recta entre la punta libre
    del robot flexible y la ubicacion espacial de la meta.
    """
    return np.linalg.norm(tentacle.compute_positions()[-1] - obj)


# ====== RENDER ======
def render_image(tentacle, obj):
    """
    Crea una matriz binaria simplificada que emula una camara de baja resolucion.
    Dibuja los segmentos del robot como pixeles encendidos y crea una caja de impacto
    para representar visualmente la ubicacion del objetivo en el mapa matricial.
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def scale(p):
        """
        Interpola de forma lineal las coordenadas del lienzo grande para ajustarlas
        al tamano estandarizado de entrada de la red neuronal convolucional.
        """
        x = int(p[0] / WIDTH * IMG_SIZE)
        y = int(p[1] / HEIGHT * IMG_SIZE)
        x = np.clip(x, 0, IMG_SIZE - 1)
        y = np.clip(y, 0, IMG_SIZE - 1)
        return np.array([x, y])

    positions = tentacle.compute_positions()

    # Trazado de lineas de conexion entre las articulaciones escaladas
    for i in range(len(positions) - 1):
        p1 = scale(positions[i])
        p2 = scale(positions[i + 1])

        xs = np.linspace(p1[0], p2[0], 10).astype(int)
        xs = np.linspace(p1[0], p2[0], 10).astype(int)
        ys = np.linspace(p1[1], p2[1], 10).astype(int)

        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = 1.0

    # Dibujo del recuadro del objetivo dentro de la matriz de imagen
    obj_p = scale(obj)
    x, y = obj_p
    img[max(0,y-2):min(IMG_SIZE,y+2), max(0,x-2):min(IMG_SIZE,x+2)] = 1.0

    return img


def stack_frames(frames, new_frame):
    """
    Gestiona la cola de imagenes secuenciales para proveer informacion temporal del movimiento.
    Si la cola no esta llena, replica el cuadro inicial para completar las dimensiones requeridas.
    """
    frames.append(new_frame)
    if len(frames) < FRAME_STACK:
        while len(frames) < FRAME_STACK:
            frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MODELO MULTI-INPUT ======
def build_model():
    """
    Construye una arquitectura de red neuronal profunda con multiples entradas.
    Procesa las imagenes secuenciales a traves de capas convolucionales y combina
    esta representacion geometrica con las lecturas directas de los sensores de tension.
    """
    # Rama de procesamiento espacial para analisis de imagenes de mapas de pixeles
    img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, FRAME_STACK))
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(img_input)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Rama de procesamiento escalar para la medicion de estados de fuerza muscular
    tension_input = tf.keras.layers.Input(shape=(2,))
    t = tf.keras.layers.Dense(32, activation='relu')(tension_input)

    # Capas densas de fusion para la estimacion de los valores Q de las acciones
    combined = tf.keras.layers.concatenate([x, t])
    combined = tf.keras.layers.Dense(512, activation='relu')(combined)
    output = tf.keras.layers.Dense(2)(combined)

    model = tf.keras.Model(inputs=[img_input, tension_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mse'
    )
    return model


# ====== MEMORIA ======
# Estructura circular de almacenamiento de experiencias pasadas del agente de IA
memory = deque(maxlen=MEMORY_SIZE)


def train_step(model):
    """
    Realiza una iteracion de ajuste de pesos de la red usando repeticion de experiencia.
    Extrae una muestra aleatoria, calcula los objetivos temporales aplicando la ecuacion
    de Bellman y optimiza los coeficientes del modelo para reducir el error cuadratico medio.
    """
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    imgs, tens, targets = [], [], []

    for state, action, reward, next_state, done in batch:
        img, ten = state
        next_img, next_ten = next_state

        q_vals = model.predict([img[np.newaxis], ten[np.newaxis]], verbose=0)[0]

        if done:
            q_vals[action] = reward
        else:
            future_q = np.max(model.predict([next_img[np.newaxis], next_ten[np.newaxis]], verbose=0)[0])
            q_vals[action] = reward + GAMMA * future_q

        imgs.append(img)
        tens.append(ten)
        targets.append(q_vals)

    model.fit([np.array(imgs), np.array(tens)], np.array(targets), verbose=0)


# ====== ENTRENAMIENTO ======
def train():
    """
    Funcion principal de control de la sesion de aprendizaje. Inicializa el modelo,
    gestiona el bucle de episodios, evalua la funcion de recompensa compuesta por acercamiento
    y contacto fisico, y guarda la red neuronal optimizada en el almacenamiento local.
    """
    model = build_model()
    epsilon = EPSILON
    step_global = 0

    for ep in range(EPISODES):
        tentacle = CableTentacle()
        obj = spawn_object()

        frames = deque(maxlen=FRAME_STACK)
        img = stack_frames(frames, render_image(tentacle, obj))

        tension_state = np.array([
            tentacle.left_tension / MAX_TENSION,
            tentacle.right_tension / MAX_TENSION
        ])

        state = (img, tension_state)

        prev_dist = tip_distance(tentacle, obj)
        total_reward = 0

        for step in range(400):

            # Mecanismo de seleccion de acciones mediante politica Epsilon-Greedy
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                q_vals = model.predict([img[np.newaxis], tension_state[np.newaxis]], verbose=0)
                action = int(np.argmax(q_vals[0]))

            tentacle.apply_action(action)
            tentacle.update_angles()

            dist = tip_distance(tentacle, obj)

            # ====== DETECCION DE CONTACTO ======
            contact = dist < OBJECT_RADIUS + 10

            # Incremento de tension al tocar para simular la fuerza de sujecion mecanica
            if contact:
                tentacle.left_tension = MAX_TENSION
                tentacle.right_tension = MAX_TENSION

            # ====== REWARD ======
            # Formula de asignacion de puntaje de mejora en funcion del cambio de distancia
            reward = (prev_dist - dist) * 80 - 0.2
            if dist >= prev_dist:
                reward -= 2

            if contact:
                reward += 3000
                done = True
            else:
                done = False

            next_img = stack_frames(frames, render_image(tentacle, obj))
            next_tension = np.array([
                tentacle.left_tension / MAX_TENSION,
                tentacle.right_tension / MAX_TENSION
            ])

            next_state = (next_img, next_tension)

            # Insercion de la transicion de aprendizaje observada en el bucle
            memory.append((state, action, reward, next_state, done))

            # Disparo periodico de los algoritmos de optimizacion de pesos
            if step_global % TRAIN_EVERY == 0:
                train_step(model)

            state = next_state
            img = next_img
            tension_state = next_tension

            prev_dist = dist
            total_reward += reward
            step_global += 1

            if done:
                break

        # Disminucion progresiva de la probabilidad de exploracion aleatoria
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if ep % 50 == 0:
            print(f"Ep {ep} | Reward {total_reward:.1f} | eps {epsilon:.3f}")

    # Guardado del archivo binario con los parametros optimizados de la red neuronal
    model.save("tentacle_visual_tension_dqn.h5")
    print("Entrenamiento listo")


if __name__ == "__main__":
    train()