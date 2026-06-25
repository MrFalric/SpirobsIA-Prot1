# =========================================================
# SCRIPT DE ENTRENAMIENTO LOCAL COMPLETO (TrainAgent.py)
# =========================================================
import sys
import types

# Simulacion del modulo imp para mantener compatibilidad con dependencias antiguas
# Evita errores de importacion si librerias de terceros buscan este modulo obsoleto
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

# Optimizacion de hilos para TensorFlow en procesadores embebidos como Raspberry Pi
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import tensorflow as tf

# HARDWARE: CONFIGURACION LATERAL DE CABLES
# Inicializacion de los motores especificando los pines GPIO de avance, retroceso y activacion (enable)
motor1 = Motor(forward=23, backward=24, enable=18)
motor2 = Motor(forward=17, backward=27, enable=22)

# Inicializacion de los encoders rotativos para medir el movimiento real de cada motor
encoder1 = RotaryEncoder(5, 6)   
encoder2 = RotaryEncoder(20, 21) 

# Tiempo en segundos que dura cada accion de movimiento de los motores
MOVE_TIME = 0.35  

def acc_curvar_izquierda():
    """
    Activa los motores para realizar una curva hacia la izquierda.
    El motor 1 avanza y el motor 2 retrocede durante el tiempo establecido.
    """
    motor1.forward(1)
    motor2.backward(1)
    time.sleep(MOVE_TIME)

def acc_curvar_derecha():
    """
    Activa los motores para realizar una curva hacia la derecha.
    El motor 2 avanza y el motor 1 retrocede durante el tiempo establecido.
    """
    motor2.forward(1)
    motor1.backward(1)
    time.sleep(MOVE_TIME)

def acc_abrir_tentaculo():
    """
    Activa ambos motores en retroceso simultaneo.
    Se utiliza para abrir o extender el mecanismo del tentaculo.
    """
    motor1.backward(1)
    motor2.backward(1)
    time.sleep(MOVE_TIME)

def acc_stop():
    """
    Detiene inmediatamente ambos motores y aplica una breve pausa de estabilizacion.
    """
    motor1.stop()
    motor2.stop()
    time.sleep(0.1)

# Diccionario que mapea indices numericos con sus respectivas funciones de accion
ACTIONS = {
    0: acc_curvar_izquierda, 
    1: acc_curvar_derecha, 
    2: acc_abrir_tentaculo, 
    3: acc_stop
}

# Direcciones I2C de los sensores de corriente INA238 y registro de lectura
INA238_1_ADDR = 0x41  
INA238_2_ADDR = 0x40  
REG_CURRENT = 0x04

def read_register(bus, addr, reg):
    """
    Lee un dato de 16 bits desde un registro I2C especifico.
    Realiza la conversion de formato Big-Endian a Little-Endian para procesar la informacion adecuadamente.
    """
    data = bus.read_word_data(addr, reg)
    return ((data & 0xFF) << 8) | (data >> 8)

def read_current(bus, addr):
    """
    Lee el registro de corriente del sensor INA238.
    Convierte el valor binario con signo a un valor absoluto en Amperios.
    """
    raw = read_register(bus, addr, REG_CURRENT)
    if raw > 32767: raw -= 65536
    return abs(raw * 0.001)

# Configuracion de la camara mediante la interfaz de video V4L2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimiza el retraso manteniendo solo el cuadro mas reciente
IMG_SIZE = 64

def get_camera_frame():
    """
    Captura una imagen de la camara, la convierte a escala de grises y reduce su tamaño a 64x64.
    Normaliza los valores de los pixeles entre 0.0 y 1.0 para la red neuronal.
    Si la camara falla, devuelve una matriz vacia de ceros.
    """
    for _ in range(2): cap.grab() # Descarta cuadros antiguos acumulados en el buffer
    ret, frame = cap.read()
    if not ret: return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=-1)

def get_dark_ratio(frame64):
    """
    Calcula el porcentaje de pixeles oscuros en la imagen.
    Se utiliza para determinar visualmente la proximidad u obstruccion de un objeto como una taza.
    """
    return float(np.mean(frame64 < 0.4))

# AGENTE DQN (6 ENTRADAS SENSORIALES LOCALES)
class DQNAgent:
    """
    Clase que gestiona la red neuronal de Aprendizaje por Refuerzo Profundo (DQN).
    Se encarga de tomar decisiones, almacenar la memoria de experiencias y entrenar el modelo.
    """
    def __init__(self, weights_file="/home/felipe/progreso_ia_tentaculo.weights.h5"):
        """
        Inicializa los hiperparametros del agente, las rutas de guardado,
        construye la arquitectura de la red y despliega el menu de inicio.
        """
        self.weights_file = weights_file
        self.epsilon_file = "/home/felipe/config_epsilon_tentaculo.txt"
        
        self.alpha = 0.0005          # Tasa de aprendizaje de la red neuronal
        self.gamma = 0.95            # Factor de descuento para recompensas futuras
        self.epsilon = 1.0           # Probabilidad inicial de exploracion aleatoria
        self.epsilon_decay = 0.995   # Factor de reduccion de la exploracion por episodio
        self.min_epsilon = 0.1       # Limite minimo de exploracion permitido
        self.batch_size = 32         # Tamaño del grupo de datos para cada paso de entrenamiento
        self.memory = deque(maxlen=4000) # Memoria circular para almacenar experiencias pasadas
        self.model = self._build_model()
        self.load_menu()

    def _build_model(self):
        """
        Construye una red neuronal multimodal de tipo funcional.
        Procesa imagenes mediante capas convolucionales y datos de sensores mediante capas densas,
        combinando ambas fuentes para predecir el valor Q de las 4 acciones posibles.
        """
        # Entrada y procesamiento de la imagen de la camara
        img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="img_in")
        x = tf.keras.layers.Conv2D(16, 5, strides=2, activation='relu')(img_input)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)

        # Entrada y procesamiento de las lecturas fisicas (6 datos)
        sensor_input = tf.keras.layers.Input(shape=(6,), name="sensor_in")
        s = tf.keras.layers.Dense(32, activation='relu')(sensor_input)

        # Fusion de caracteristicas visuales y sensoriales
        combined = tf.keras.layers.concatenate([x, s])
        combined = tf.keras.layers.Dense(64, activation='relu')(combined)
        output = tf.keras.layers.Dense(4, activation='linear')(combined)

        model = tf.keras.Model(inputs=[img_input, sensor_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.alpha), loss='mse')
        return model

    def get_action(self, frame, sensors):
        """
        Selecciona una accion basandose en la estrategia Epsilon-Greedy.
        Con probabilidad Epsilon elige una accion al azar (exploracion),
        de lo contrario utiliza la red neuronal para predecir la mejor accion (explotacion).
        """
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        q_values = self.model([frame[np.newaxis], sensors[np.newaxis]], training=False).numpy()
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Guarda una transicion de experiencia en la memoria de repeticion.
        Almacena el estado actual, accion ejecutada, recompensa recibida, estado siguiente y estado de finalizacion.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Entrena la red neuronal extrayendo un lote aleatorio de experiencias previas.
        Calcula los valores Q objetivos utilizando la ecuacion de Bellman y ajusta los pesos de la red.
        Reduce gradualmente el valor de Epsilon para disminuir la exploracion a lo largo del tiempo.
        """
        if len(self.memory) < self.batch_size: return
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
        Guarda los pesos actuales de la red neuronal y el valor de Epsilon en el almacenamiento local.
        Asegura que el progreso no se pierda al interrumpir el programa.
        """
        self.model.save_weights(self.weights_file)
        with open(self.epsilon_file, "w") as f: 
            f.write(str(self.epsilon))

    def load_menu(self):
        """
        Despliega un menu interactivo en la terminal si se detecta un entrenamiento guardado.
        Permite al usuario elegir entre continuar la sesion anterior, reiniciar la exploracion o empezar de cero.
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

# RECOMPENSA FISICA ASIMETRICA CON FILTROS ANTI-FALSO POSITIVO
def compute_reward(action, c1, c2, enc1, enc2, dark_ratio, step):
    """
    Calcula la recompensa numerica basada en la interaccion fisica del robot con el entorno.
    Valida corrientes, movimientos mecanicos detectados por encoders y señales de la camara
    para evitar falsos positivos y otorgar la maxima puntuacion al hacer contacto correcto con el objetivo.
    Termina el episodio si se logra el objetivo o si se excede un limite de seguridad por corriente.
    """
    if action == 3: 
        return -0.2, False  # Penalizacion suave por quedarse estatico

    reward, done = 0.1, False
    
    # Penalizacion si ambos motores consumen corriente elevada de forma simultanea sin sentido
    if c1 >= 0.380 and c2 >= 0.380:
        reward -= 0.5

    if step > 1:
        # Verifica si el brazo se movio fisicamente una cantidad minima de pasos en los encoders
        brazo_en_movimiento = (abs(enc1) + abs(enc2)) >= 24
        
        # Filtros estrictos que determinan si hay un contacto lateral real con el objeto objetivo (la taza)
        contacto_m1 = (c1 >= 0.390 and c2 < 0.320 and brazo_en_movimiento and dark_ratio > 0.12 and enc1 > 0)
        contacto_m2 = (c2 >= 0.390 and c1 < 0.320 and brazo_en_movimiento and dark_ratio > 0.12 and enc2 > 0)

        if contacto_m1 or contacto_m2:
            print(f"\n[CONTACTO REAL ASISTIDO DETECTADO]")
            print(f"-> Coherencia: M1={c1:.3f}A ({enc1} enc) | M2={c2:.3f}A ({enc2} enc) | Vista={dark_ratio:.3f}")
            reward += 100.0
            done = True

    # Parada de emergencia por seguridad si la corriente supera los limites criticos del motor
    if c1 > 0.650 or c2 > 0.650:
        print("\n[EMERGENCIA] Corriente critica superada!")
        reward -= 20.0
        done = True

    return reward, done

def main():
    """
    Funcion principal que coordina el ciclo de entrenamiento global.
    Establece la conexion I2C, gestiona la estructura de episodios, interactua con el hardware,
    recolecta los estados sensoriales y ejecuta periodicamente el guardado de datos.
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
                
                # Construccion del estado inicial compuesto por lecturas fisicas y visuales
                sensor_state = np.array([c1, c2, abs(c1 - c2), 0.0, 0.0, d_ratio], dtype=np.float32)
                ep_reward = 0

                for step in range(1, STEPS_PER_EPISODE + 1):
                    # Registro de la posicion previa de los encoders
                    p1_inc, p2_inc = encoder1.steps, encoder2.steps
                    
                    # El agente decide la accion y esta se ejecuta en los motores fisicos
                    action = agent.get_action(frame, sensor_state)
                    ACTIONS[action]()  
                    
                    # Medicion del desplazamiento real generado tras ejecutar la accion
                    mov_m1 = encoder1.steps - p1_inc; mov_m2 = encoder2.steps - p2_inc
                    
                    # Captura del nuevo estado del entorno despues del movimiento
                    next_frame = get_camera_frame()
                    nd_ratio = get_dark_ratio(next_frame)
                    
                    try: nc1 = read_current(bus, INA238_1_ADDR); nc2 = read_current(bus, INA238_2_ADDR)
                    except: nc1 = nc2 = 0
                        
                    next_sensor_state = np.array([nc1, nc2, abs(nc1 - nc2), float(mov_m1), float(mov_m2), nd_ratio], dtype=np.float32)
                    
                    # Evaluacion del desempeño para obtener la recompensa y verificar si concluyo el intento
                    reward, done = compute_reward(action, nc1, nc2, mov_m1, mov_m2, nd_ratio, step)
                    ep_reward += reward
                    
                    # Almacenamiento en memoria y ejecucion de un paso de optimizacion de la red neuronal
                    agent.remember((frame, sensor_state), action, reward, (next_frame, next_sensor_state), done)
                    agent.replay()
                    
                    frame, sensor_state = next_frame, next_sensor_state
                    
                    # Impresion de telemetria en consola cada 5 pasos o al finalizar el episodio
                    if step % 5 == 0 or done:
                        print(f"Paso: {step:2d} | Acc: {action} | Encoders: [{mov_m1}/{mov_m2}] | Dark_Ratio: {nd_ratio:.3f} | M1:{nc1:.3f}A M2:{nc2:.3f}A")
                    if done: acc_stop(); break
                        
                print(f"Fin Episodio {episode}. Reward: {ep_reward:.2f}")
                # Guarda los pesos de seguridad de la IA de manera automatica cada 5 episodios completos
                if episode % 5 == 0: agent.save_agent()
                    
        except KeyboardInterrupt: 
            print("\nCierre solicitado por teclado.")
        finally: 
            # Asegura el apagado de motores y la liberacion de recursos de hardware al salir
            acc_stop()
            cap.release()
            agent.save_agent()
            print("Pesos y progreso guardados localmente.")

if __name__ == "__main__": 
    main()