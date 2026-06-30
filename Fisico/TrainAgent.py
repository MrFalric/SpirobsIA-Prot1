# =========================================================
# SCRIPT DE ENTRENAMIENTO LOCAL COMPLETO (TrainAgent.py)
# =========================================================
import sys
import types

# Simulacion del modulo imp para mantener compatibilidad con dependencias antiguas
imp_mock = types.ModuleType('imp')
def fake_find_module(name, path=None): return (None, None, (None, None, None))
imp_mock.find_module = fake_find_module
sys.modules['imp'] = imp_mock

import os
import time
import random
from collections import deque
import numpy as np
import cv2
from smbus2 import SMBus
from gpiozero import Motor, RotaryEncoder

# Optimizacion de hilos para la ejecucion de TensorFlow en hardware embebido
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import tensorflow as tf

# HARDWARE: CONFIGURACIÓN LATERAL DE CABLES
# Inicializacion de motores DC y codificadores rotativos para el control de tension
motor1 = Motor(forward=23, backward=24, enable=18)
motor2 = Motor(forward=17, backward=27, enable=22)

encoder1 = RotaryEncoder(5, 6)   
encoder2 = RotaryEncoder(20, 21) 

MOVE_TIME = 0.35  

def acc_curvar_izquierda():
    """
    Activa los motores para generar una curvatura hacia el lado izquierdo.
    """
    motor1.forward(1)
    motor2.backward(1)
    time.sleep(MOVE_TIME)

def acc_curvar_derecha():
    """
    Activa los motores para generar una curvatura hacia el lado derecho.
    """
    motor2.forward(1)
    motor1.backward(1)
    time.sleep(MOVE_TIME)

def acc_abrir_tentaculo():
    """
    Libera tension de ambos cables para abrir o relajar el tentaculo.
    """
    motor1.backward(1)
    motor2.backward(1)
    time.sleep(MOVE_TIME)

def acc_stop():
    """
    Detiene el movimiento de todos los motores de forma inmediata.
    """
    motor1.stop()
    motor2.stop()
    time.sleep(0.1)

# Diccionario de mapeo de acciones discretas del agente de inteligencia artificial
ACTIONS = {
    0: acc_curvar_izquierda, 
    1: acc_curvar_derecha, 
    2: acc_abrir_tentaculo, 
    3: acc_stop
}

# Parametros de configuracion para los sensores de corriente INA238 por bus I2C
INA238_1_ADDR = 0x41  
INA238_2_ADDR = 0x40  
REG_CURRENT = 0x04

def read_register(bus, addr, reg):
    """
    Lee un registro de 16 bits desde el bus I2C y ajusta el orden de los bytes.
    """
    data = bus.read_word_data(addr, reg)
    return ((data & 0xFF) << 8) | (data >> 8)

def read_current(bus, addr):
    """
    Obtiene la lectura de corriente absoluta convertida a Amperios.
    """
    raw = read_register(bus, addr, REG_CURRENT)
    if raw > 32767: raw -= 65536
    return abs(raw * 0.001)

# Configuracion de la camara mediante Video4Linux2 para captura rapida
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
IMG_SIZE = 64

def get_camera_frame():
    """
    Captura un cuadro de la camara, limpia el buffer, convierte a escala
    de grises, redimensiona y normaliza los valores de los pixeles.
    """
    for _ in range(2): cap.grab()
    ret, frame = cap.read()
    if not ret: return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=-1)

def get_dark_ratio(frame64):
    """
    Calcula la proporcion de pixeles oscuros para detectar la presencia de objetos.
    """
    return float(np.mean(frame64 < 0.4))


# AGENTE DQN (6 ENTRADAS SENSORIALES LOCALES)
class DQNAgent:
    def __init__(self, weights_file="/home/felipe/progreso_ia_tentaculo.weights.h5"):
        """
        Inicializa las variables de aprendizaje por refuerzo, define la estructura
        de la memoria de experiencia y gestiona la persistencia de archivos.
        """
        self.weights_file = weights_file
        self.epsilon_file = "/home/felipe/config_epsilon_tentaculo.txt"
        
        self.alpha = 0.0005
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        self.batch_size = 32
        self.memory = deque(maxlen=4000)
        self.model = self._build_model()
        self.load_menu()

    def _build_model(self):
        """
        Construye el modelo de red neuronal multimodal con una rama convolucional
        para imagenes y una rama densa para datos vectoriales de sensores.
        """
        img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="img_in")
        x = tf.keras.layers.Conv2D(16, 5, strides=2, activation='relu')(img_input)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)

        sensor_input = tf.keras.layers.Input(shape=(6,), name="sensor_in")
        s = tf.keras.layers.Dense(32, activation='relu')(sensor_input)

        combined = tf.keras.layers.concatenate([x, s])
        combined = tf.keras.layers.Dense(64, activation='relu')(combined)
        output = tf.keras.layers.Dense(4, activation='linear')(combined)

        model = tf.keras.Model(inputs=[img_input, sensor_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.alpha), loss='mse')
        return model

    def get_action(self, frame, sensors):
        """
        Selecciona una accion usando la estrategia Epsilon-Greedy para
        balancear la exploracion aleatoria y la explotacion del conocimiento.
        """
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        q_values = self.model([frame[np.newaxis], sensors[np.newaxis]], training=False).numpy()
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una transicion de estado en el buffer circular de memoria.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Extrae un lote de experiencias pasadas para entrenar la red neuronal,
        calcula los valores Q objetivo y decrementa el valor de epsilon.
        """
        if len(memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        states_img = np.array([m[0][0] for m in minibatch])
        states_sen = np.array([m[0][1] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states_img = np.array([m[3][0] for m in minibatch])
        next_states_sen = np.array([m[3][1] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        current_q = self.model([states_img, states_sen], training=False).numpy()
        next_q = self.model([next_states_img, next_states_sen], training=False).numpy()

        for i in range(self.batch_size):
            if dones[i]: current_q[i, actions[i]] = rewards[i]
            else: current_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        self.model.fit([states_img, states_sen], current_q, batch_size=self.batch_size, verbose=0)
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

    def save_agent(self):
        """
        Guarda los pesos de la red neuronal y el estado de epsilon en el disco.
        """
        self.model.save_weights(self.weights_file)
        with open(self.epsilon_file, "w") as f: 
            f.write(str(self.epsilon))

    def load_menu(self):
        """
        Despliega un menu interactivo por consola para seleccionar el modo de carga
        de datos o inicializacion del agente desde cero.
        """
        if os.path.exists(self.weights_file):
            print("\n1) CARGAR ENTRENAMIENTO PREVIO\n2) Resetear exploracion\n3) Iniciar de cero")
            opc = input("Opcion: ")
            if opc == "1":
                self.model.load_weights(self.weights_file)
                if os.path.exists(self.epsilon_file):
                    with open(self.epsilon_file, "r") as f: 
                        self.epsilon = float(f.read().strip())
            elif opc == "2":
                self.model.load_weights(self.weights_file)
                self.epsilon = 1.0
            elif opc == "3":
                print("Iniciando red neuronal desde cero...")


# RECOMPENSA FÍSICA ASIMÉTRICA CON FILTROS ANTI-FALSO POSITIVO
def compute_reward(action, c1, c2, enc1, enc2, dark_ratio, step):
    """
    Calcula la recompensa del agente aplicando penalizaciones por inaccion,
    sobretensiones, y asignando un gran estimulo si se valida un contacto real.
    """
    if action == 3: 
        return -0.2, False  # Castigo de STOP suavizado

    reward, done = 0.1, False
    
    if c1 >= 0.380 and c2 >= 0.380:
        reward -= 0.5

    if step > 1:
        # Validacion de movimiento real en bloques discretos
        brazo_en_movimiento = (abs(enc1) + abs(enc2)) >= 24
        
        # Filtros de direccion coherente + validacion de presencia visual de la taza
        contacto_m1 = (c1 >= 0.390 and c2 < 0.320 and brazo_en_movimiento and dark_ratio > 0.12 and enc1 > 0)
        contacto_m2 = (c2 >= 0.390 and c1 < 0.320 and brazo_en_movimiento and dark_ratio > 0.12 and enc2 > 0)

        if contacto_m1 or contacto_m2:
            print(f"\n[CONTACTO REAL ASISTIDO DETECTADO]")
            print(f"-> Coherencia: M1={c1:.3f}A ({enc1} enc) | M2={c2:.3f}A ({enc2} enc) | Vista={dark_ratio:.3f}")
            reward += 100.0
            done = True

    if c1 > 0.650 or c2 > 0.650:
        print("\n[EMERGENCIA] Corriente critica superada!")
        reward -= 20.0
        done = True

    return reward, done


def main():
    """
    Funcion principal que gestiona el bucle de episodios de entrenamiento fisico.
    Lee los sensores, ejecuta acciones de control en tiempo real y guarda el progreso.
    """
    agent = DQNAgent()
    STEPS_PER_EPISODE = 30 
    
    with SMBus(1) as bus:
        try:
            for episode in range(1, 1001):
                print(f"\n--- Episodio {episode} | Epsilon: {agent.epsilon:.4f} ---")
                frame = get_camera_frame()
                d_ratio = get_dark_ratio(frame)
                
                try: c1 = read_current(bus, INA238_1_ADDR); c2 = read_current(bus, INA238_2_ADDR)
                except: c1 = c2 = 0
                
                sensor_state = np.array([c1, c2, abs(c1 - c2), 0.0, 0.0, d_ratio], dtype=np.float32)
                ep_reward = 0

                for step in range(1, STEPS_PER_EPISODE + 1):
                    p1_inc, p2_inc = encoder1.steps, encoder2.steps
                    action = agent.get_action(frame, sensor_state)
                    ACTIONS[action]()  
                    mov_m1 = encoder1.steps - p1_inc; mov_m2 = encoder2.steps - p2_inc
                    
                    next_frame = get_camera_frame()
                    nd_ratio = get_dark_ratio(next_frame)
                    
                    try: nc1 = read_current(bus, INA238_1_ADDR); nc2 = read_current(bus, INA238_2_ADDR)
                    except: nc1 = nc2 = 0
                        
                    next_sensor_state = np.array([nc1, nc2, abs(nc1 - nc2), float(mov_m1), float(mov_m2), nd_ratio], dtype=np.float32)
                    
                    reward, done = compute_reward(action, nc1, nc2, mov_m1, mov_m2, nd_ratio, step)
                    ep_reward += reward
                    
                    agent.remember((frame, sensor_state), action, reward, (next_frame, next_sensor_state), done)
                    agent.replay()
                    
                    frame, sensor_state = next_frame, next_sensor_state
                    if step % 5 == 0 or done:
                        print(f"Paso: {step:2d} | Acc: {action} | Encoders: [{mov_m1}/{mov_m2}] | Dark_Ratio: {nd_ratio:.3f} | M1:{nc1:.3f}A M2:{nc2:.3f}A")
                    if done: acc_stop(); break
                        
                print(f"Fin Episodio {episode}. Reward: {ep_reward:.2f}")
                if episode % 5 == 0: agent.save_agent()
                    
        except KeyboardInterrupt: 
            print("\nCierre solicitado por teclado.")
        finally: 
            acc_stop()
            cap.release()
            agent.save_agent()
            print("Pesos y progreso guardados localmente.")

if __name__ == "__main__": 
    main()