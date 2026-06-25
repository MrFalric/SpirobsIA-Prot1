import numpy as np
import random
import math
import tensorflow as tf
from collections import deque

# ====== PARAMETROS ======
NUM_SEGMENTS = 19
BASE_LENGTH = 50
WIDTH = 1000
HEIGHT = 700
BASE_POS = np.array([WIDTH // 2, HEIGHT - 20])
BASE_ANGLE = -np.pi / 2

MAX_TENSION = 1.5
TENSION_STEP = 0.04

# ¡Corregido para coincidir con la simulación!
OBJECT_RADIUS = 20 
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

EPISODES = 1000  # Subido ligeramente para dar más tiempo de convergencia

GAMMA = 0.97
LR = 0.00025
BATCH_SIZE = 64
MEMORY_SIZE = 20000
TRAIN_EVERY = 4

EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

FRAME_STACK = 4
IMG_SIZE = 84

# ====== ENTORNO ======
class Segment:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.angle = 0.0


class CableTentacle:
    def __init__(self):
        lengths = np.linspace(BASE_LENGTH, BASE_LENGTH * 0.4, NUM_SEGMENTS)
        widths = np.linspace(30, 8, NUM_SEGMENTS)
        self.segments = [Segment(l, w) for l, w in zip(lengths, widths)]

        self.left_tension = 1.0
        self.right_tension = 1.0

    def apply_action(self, action):
        if action == 0:
            self.left_tension += TENSION_STEP
            self.right_tension -= TENSION_STEP / 2
        elif action == 1:
            self.right_tension += TENSION_STEP
            self.left_tension -= TENSION_STEP / 2

        self.left_tension = np.clip(self.left_tension, 0.5, MAX_TENSION)
        self.right_tension = np.clip(self.right_tension, 0.5, MAX_TENSION)

    def update_angles(self):
        diff = self.right_tension - self.left_tension
        for i, seg in enumerate(self.segments):
            seg.angle += diff * (i + 1) * 0.01
            seg.angle = np.clip(seg.angle, -np.pi/5, np.pi/5)

    def compute_positions(self):
        positions = [BASE_POS.astype(float)]
        current_angle = BASE_ANGLE
        for seg in self.segments:
            current_angle += seg.angle
            dx = seg.length * np.cos(current_angle)
            dy = seg.length * np.sin(current_angle)
            positions.append(positions[-1] + np.array([dx, dy]))
        return positions


# ====== OBJETO ======
def spawn_object():
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.8)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y], dtype=float)


# ====== FISICA OBJETO ======
def apply_object_physics(positions, obj_pos, obj_vel, grip_strength):
    total_force = np.zeros(2)
    contact_count = 0

    for p in positions:
        offset = obj_pos - p
        dist = np.linalg.norm(offset)

        if 0 < dist < OBJECT_RADIUS:
            direction = offset / dist
            penetration = OBJECT_RADIUS - dist
            force = direction * penetration * 0.5

            total_force += force
            contact_count += 1

    if contact_count > 0:
        avg_force = total_force / contact_count
        obj_vel += avg_force * (0.8 + grip_strength)
        obj_vel *= 0.85
    else:
        obj_vel *= 0.92

    obj_pos += obj_vel
    return obj_pos, obj_vel


# ====== DETECCIONES ======
def check_collision(positions, obj):
    for p in positions:
        if np.linalg.norm(p - obj) <= OBJECT_RADIUS:
            return True
    return False


def check_enclosure(positions, obj):
    close = [p for p in positions if np.linalg.norm(p - obj) <= OBJECT_RADIUS * 1.2]
    return len(close) >= 3


# ====== RENDER ======
def render_image(tentacle, obj):
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
        ys = np.linspace(p1[1], p2[1], 10).astype(int)

        valid = (xs >= 0) & (xs < IMG_SIZE) & (ys >= 0) & (ys < IMG_SIZE)
        img[ys[valid], xs[valid]] = 1.0

    obj_p = scale(obj)
    x, y = obj_p
    img[max(0,y-3):min(IMG_SIZE,y+3), max(0,x-3):min(IMG_SIZE,x+3)] = 1.0

    return img


def stack_frames(frames, new_frame):
    frames.append(new_frame)
    if len(frames) < FRAME_STACK:
        while len(frames) < FRAME_STACK:
            frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MODELO ======
def build_model():
    img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, FRAME_STACK))
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(img_input)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    tension_input = tf.keras.layers.Input(shape=(2,))
    t = tf.keras.layers.Dense(32, activation='relu')(tension_input)

    combined = tf.keras.layers.concatenate([x, t])
    combined = tf.keras.layers.Dense(512, activation='relu')(combined)
    output = tf.keras.layers.Dense(2)(combined)

    model = tf.keras.Model(inputs=[img_input, tension_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
    return model


# ====== MEMORIA ======
memory = deque(maxlen=MEMORY_SIZE)


def train_step(model):
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    # Separar datos del batch de manera eficiente
    states_img = np.array([b[0][0] for b in batch])
    states_ten = np.array([b[0][1] for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.array([b[2] for b in batch])
    next_states_img = np.array([b[3][0] for b in batch])
    next_states_ten = np.array([b[3][1] for b in batch])
    dones = np.array([b[4] for b in batch])

    # ¡CORRECCIÓN CRÍTICA!: Predecir todo el batch de golpe (100x más rápido)
    current_q_batch = model([states_img, states_ten], training=False).numpy()
    next_q_batch = model([next_states_img, next_states_ten], training=False).numpy()

    for i in range(BATCH_SIZE):
        if dones[i]:
            current_q_batch[i, actions[i]] = rewards[i]
        else:
            current_q_batch[i, actions[i]] = rewards[i] + GAMMA * np.max(next_q_batch[i])

    model.fit([states_img, states_ten], current_q_batch, verbose=0, batch_size=BATCH_SIZE)


# ====== ENTRENAMIENTO ======
def train():
    model = build_model()
    epsilon = EPSILON
    step_global = 0

    for ep in range(EPISODES):
        tentacle = CableTentacle()
        obj = spawn_object()
        obj_vel = np.zeros(2)
        grip_strength = 0.0

        frames = deque(maxlen=FRAME_STACK)
        img = stack_frames(frames, render_image(tentacle, obj))

        tension_state = np.array([
            tentacle.left_tension / MAX_TENSION,
            tentacle.right_tension / MAX_TENSION
        ])

        prev_dist = np.linalg.norm(tentacle.compute_positions()[-1] - obj)
        total_reward = 0

        for step in range(400):
            # Selección de acción
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                # Uso directo del modelo para evitar la lentitud de predict() en bucles vivos
                q_vals = model([img[np.newaxis], tension_state[np.newaxis]], training=False).numpy()
                action = int(np.argmax(q_vals[0]))

            tentacle.apply_action(action)
            tentacle.update_angles()

            positions = tentacle.compute_positions()

            collision = check_collision(positions, obj)
            enclosed = check_enclosure(positions, obj)

            if enclosed:
                grip_strength = min(1.0, grip_strength + 0.05)
            else:
                grip_strength = max(0.0, grip_strength - 0.05)

            obj, obj_vel = apply_object_physics(positions, obj, obj_vel, grip_strength)
            dist = np.linalg.norm(positions[-1] - obj)

            # ====== REWARD ======
            reward = (prev_dist - dist) * 60

            if collision:
                reward += 5
            if enclosed:
                reward += 20

            reward += grip_strength * 50

            if grip_strength > 0.8:
                reward += 500
                done = True
            else:
                done = False

            next_img = stack_frames(frames, render_image(tentacle, obj))
            next_tension = np.array([
                tentacle.left_tension / MAX_TENSION,
                tentacle.right_tension / MAX_TENSION
            ])

            memory.append(
                ((img, tension_state), action, reward, (next_img, next_tension), done)
            )

            if step_global % TRAIN_EVERY == 0:
                train_step(model)

            img = next_img
            tension_state = next_tension
            prev_dist = dist

            total_reward += reward
            step_global += 1

            if done:
                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        print(f"Ep {ep} | Reward {total_reward:.1f} | eps {epsilon:.3f}")

    # Unificado el nombre del archivo para que coincida con el simulador
    model.save("tentacle_visual_tension_dqn.h5")
    print("Entrenamiento listo y modelo guardado.")


if __name__ == "__main__":
    train()