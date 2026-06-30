# Control de Tentáculo Flexible mediante Aprendizaje por Refuerzo (Q-Learning y DQN)

Este proyecto implementa diferentes enfoques de Inteligencia Artificial para controlar un brazo robótico flexible utilizando el modelo del spirobs accionado por un sistema de cables o tendones artificiales. El objetivo principal es que el tentáculo aprenda de forma autónoma a curvarse y estirarse para tocar un objetivo en el espacio.

El repositorio se divide en tres fases evolutivas distribuidas en **5 scripts principales**:
1. **Fase Q-Learning:** `train_tentacle.py` y `simulate_tentacle.py`
2. **Fase DQN:** `train_visual.py` y `simulate_visual.py`
3. **Fase Hardware Real:** `TrainAgent.py`

---

## Estructura del Proyecto y Scripts

### 1. Q-Learning Clásico (Basado en Tabla Q)
* **`train_tentacle.py`**: Entrena al agente utilizando un diccionario de Python como Tabla Q. Discretiza las distancias, ángulos y tensiones en "bloques" numéricos y aplica la Ecuación de Bellman clásica de forma matemática pura (sin gráficos) para exportar el cerebro en el archivo binario `tentacle_q.pkl`.
* **`simulate_tentacle.py`**: Interfaz gráfica en 2D desarrollada con Pygame que carga el archivo `tentacle_q.pkl`. Muestra la deformación física del tentáculo usando geometría vectorial y líneas de tensión de colores (azul y rojo) mientras busca capturar el objetivo amarillo.

### 2. Deep Q-Network (DQN Multimodal Virtual)
* **`train_visual.py`**: Eleva el control a redes neuronales. Entrena un modelo multimodal en TensorFlow que recibe simultáneamente una matriz de píxeles (visión artificial simulada del entorno) y lecturas numéricas continuas. Guarda los pesos optimizados en `tentacle_visual_tension_dqn.h5`.
* **`simulate_visual.py`**: Renderiza el entorno visual dinámico y evalúa el comportamiento del agente neuronal utilizando predicciones de la red en tiempo real.

### 3. Transferencia a Hardware Real
* **`TrainAgent.py`**: Script optimizado exclusivamente para ejecutarse en una **Raspberry Pi**. Sustituye las funciones físicas virtuales por interacciones directas con componentes electrónicos reales (Motores DC, encoders rotativos y sensores de corriente por bus I2C).

---

## Requisitos e Instalación

Para instalar todas las librerías necesarias del proyecto en tu entorno, abre una terminal y ejecuta el siguiente comando:

```bash
pip install numpy pygame opencv-python tensorflow gpiozero smbus2```