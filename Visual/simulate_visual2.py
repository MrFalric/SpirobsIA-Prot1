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

MAX_TIME = 15

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
pygame.display.set_caption("Tentáculo IA - AGARRE REAL")
font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

# ====== MODELO ======
# Cargamos el archivo correcto unificado
model = tf.keras.models.load_model("tentacle_visual_tension_dqn.h5", compile=False)

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
        left_points, right_points = [], []

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

            # Manteniendo tu estructura original de renderizado intacta
            pygame.draw.polygon(surface, GREEN, points)

        pygame.draw.lines(surface, BLUE, False, left_points, 2)
        pygame.draw.lines(surface, RED, False, right_points, 2)

        return positions


# ====== AGARRE ======
def check_enclosure(positions, obj_center, obj_radius):
    count = sum(np.linalg.norm(p - obj_center) < obj_radius * 1.2 for p in positions)
    return count >= 3


def spawn_object():
    angle = random.uniform(-np.pi / 3, np.pi / 3)
    distance = random.uniform(MAX_REACH * 0.3, MAX_REACH * 0.7)
    x = BASE_POS[0] + distance * np.sin(angle)
    y = BASE_POS[1] - distance * np.cos(angle)
    return np.array([x, y])


# ====== IA RENDER ======
def render_image(tentacle, obj):
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def scale(p):
        x = int(p[0] / WIDTH * IMG_SIZE)
        y = int(p[1] / HEIGHT * IMG_SIZE)
        return np.clip([x, y], 0, IMG_SIZE - 1)

    positions = tentacle.compute_positions()

    for i in range(len(positions) - 1):
        p1 = scale(positions[i])
        p2 = scale(positions[i + 1])
        xs = np.linspace(p1[0], p2[0], 10).astype(int)
        ys = np.linspace(p1[1], p2[1], 10).astype(int)
        img[ys, xs] = 1.0

    p = scale(obj)
    img[p[1]-2:p[1]+2, p[0]-2:p[0]+2] = 1.0

    return img


def stack_frames(frames, new_frame):
    frames.append(new_frame)
    while len(frames) < FRAME_STACK:
        frames.append(new_frame)
    return np.stack(frames, axis=-1)


# ====== MAIN ======
def main():
    tentacle = CableTentacle()
    obj = spawn_object()

    obj_vel = np.zeros(2)
    grip_strength = 0.0

    frames = deque(maxlen=FRAME_STACK)
    img = stack_frames(frames, render_image(tentacle, obj))

    captured = 0
    missed = 0
    start_time = time.time()

    running = True
    while running:
        screen.fill(BLACK)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # TIEMPO LÍMITE
        elapsed = time.time() - start_time
        if elapsed >= MAX_TIME:
            missed += 1
            tentacle = CableTentacle()
            obj = spawn_object()
            obj_vel *= 0
            grip_strength = 0
            start_time = time.time()

        # ====== IA ======
        tension_state = np.array([
            tentacle.left_tension / MAX_TENSION,
            tentacle.right_tension / MAX_TENSION
        ])

        # Usar llamada directa en lugar de predict() para evitar micro-stuttering en Pygame
        q_vals = model([img[np.newaxis], tension_state[np.newaxis]], training=False).numpy()
        action = int(np.argmax(q_vals[0]))

        tentacle.apply_action(action)
        tentacle.update_angles()

        positions = tentacle.draw(screen)

        # ====== AGARRE ======
        enclosed = check_enclosure(positions, obj, OBJECT_RADIUS)

        if enclosed:
            grip_strength = min(1.0, grip_strength + 0.05)
        else:
            grip_strength = max(0.0, grip_strength - 0.05)

        # ====== FÍSICA ======
        total_force = np.zeros(2)
        contacts = 0

        for p in positions:
            d = np.linalg.norm(obj - p)
            if 0 < d < OBJECT_RADIUS:
                direction = (obj - p) / d
                total_force += direction * (OBJECT_RADIUS - d)
                contacts += 1

        if contacts > 0:
            obj_vel += (total_force / contacts) * (0.5 + grip_strength)

        obj_vel *= 0.9
        obj += obj_vel

        # ====== CAPTURA REAL ======
        if grip_strength > 0.9:
            captured += 1
            tentacle = CableTentacle()
            obj = spawn_object()
            obj_vel *= 0
            grip_strength = 0
            start_time = time.time()

        # ====== VISUAL OBJETO ======
        color = (int(255*(1-grip_strength)), int(255*grip_strength), 0)
        pygame.draw.circle(screen, color, obj.astype(int), OBJECT_RADIUS)

        # ====== FRAME IA ======
        img = stack_frames(frames, render_image(tentacle, obj))

        # ====== UI ======
        time_left = max(0, int(MAX_TIME - elapsed))

        info = [
            f"Capturados: {captured}",
            f"Fallados: {missed}",
            f"Tiempo: {time_left}s",
            f"Grip: {grip_strength:.2f}"
        ]

        for i, txt in enumerate(info):
            screen.blit(font.render(txt, True, WHITE), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()