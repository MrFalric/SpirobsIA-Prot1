import pygame
import numpy as np
import random
import math
import time
import tensorflow as tf
from collections import deque

# ====== PARÁMETROS ======
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
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (80, 80, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
ORANGE = (255, 165, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación Tentáculo - IA Visual + Tensión")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# ====== MODELO ======
model = tf.keras.models.load_model("tentacle_visual_tension_dqn.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
    loss='mse'
)

print("Modelo visual + tensión cargado correctamente")

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

        # 👇 igual que entrenamiento
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
            seg.angle = np.clip(seg.angle, -np.pi / 5, np.pi / 5)

    def compute_positions(self):
        positions = [BASE_POS.copy()]
        current_angle = BASE_ANGLE
        for seg in self.segments:
            current_angle += seg.angle
            dx = seg.length * np.cos(current_angle)
            dy = seg.length * np.sin(current_angle)
            positions.append(positions[-1] + np.array([dx, dy]))
        return positions

    def draw(self, surface):
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
        return self.compute_positions()[-1]


# ====== FUNCIONES ======
def spawn_object():
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.7)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y])


# ====== RENDER IA ======
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


# ====== MAIN ======
def main():
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

        if time.time() - start_time >= 15:
            missed_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()

        # ====== INPUT MODELO ======
        tension_state = np.array([
            tentacle.left_tension / MAX_TENSION,
            tentacle.right_tension / MAX_TENSION
        ])

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

            # 👇 igual que entrenamiento
            tentacle.left_tension = MAX_TENSION
            tentacle.right_tension = MAX_TENSION

            captured_count += 1
            tentacle = CableTentacle()
            target = spawn_object()
            start_time = time.time()

        # ====== NUEVO FRAME ======
        img = stack_frames(frames, render_image(tentacle, target))

        # ====== UI ======
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