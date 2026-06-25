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
CABLE_FORCE_FACTOR = 0.04

OBJECT_RADIUS = 20
MAX_REACH = NUM_SEGMENTS * BASE_LENGTH * 0.55

EPISODES = 25  # subir esto despues

GAMMA = 0.97
LR = 0.00025
BATCH_SIZE = 64
MEMORY_SIZE = 30000
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
        self.left_tension = 1.0   # 👈 base constante
        self.right_tension = 1.0

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

    def apply_action(self, action):
        if action == 0:
            self.left_tension += TENSION_STEP
            self.right_tension -= TENSION_STEP / 2
        elif action == 1:
            self.right_tension += TENSION_STEP
            self.left_tension -= TENSION_STEP / 2

        # mantener base ~ 1
        self.left_tension = np.clip(self.left_tension, 0.5, MAX_TENSION)
        self.right_tension = np.clip(self.right_tension, 0.5, MAX_TENSION)


# ====== UTILIDADES ======
def spawn_object():
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.8)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y], dtype=float)


def tip_distance(tentacle, obj):
    return np.linalg.norm(tentacle.compute_positions()[-1] - obj)


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
    img[max(0,y-2):min(IMG_SIZE,y+2), max(0,x-2):min(IMG_SIZE,x+2)] = 1.0

    return img


def stack_frames(frames, new_frame):
    frames.append(new_frame)
    if len(frames) < FRAME_STACK:
        while len(frames) < FRAME_STACK:
            frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MODELO MULTI-INPUT ======
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mse'
    )
    return model


# ====== MEMORIA ======
memory = deque(maxlen=MEMORY_SIZE)


def train_step(model):
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

            # 👇 aumento de tensión al tocar
            if contact:
                tentacle.left_tension = MAX_TENSION
                tentacle.right_tension = MAX_TENSION

            # ====== REWARD ======
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

            memory.append((state, action, reward, next_state, done))

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

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if ep % 50 == 0:
            print(f"Ep {ep} | Reward {total_reward:.1f} | eps {epsilon:.3f}")

    model.save("tentacle_visual_tension_dqn.h5")
    print("Entrenamiento listo")


if __name__ == "__main__":
    train()