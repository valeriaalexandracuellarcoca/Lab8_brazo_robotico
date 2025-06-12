import gymnasium as gym
import robotic_arm_env
import mujoco
from mujoco import viewer
import numpy as np
import os

# Verificar que el archivo XML existe
xml_path = "universal_robots_ur5e/ur5e.xml"
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"No se encontró el archivo XML en: {xml_path}")

# Registrar el entorno
gym.envs.register(
    id="robotic_arm_env/RoboticArmEnv-v0",
    entry_point="robotic_arm_env:RoboticArmEnv",
)

# Crear el entorno
try:
    env = gym.make("robotic_arm_env/RoboticArmEnv-v0", xml_path=xml_path)
except Exception as e:
    print(f"Error al crear el entorno: {e}")
    exit(1)

# Inicializar el visualizador
try:
    mj_viewer = viewer.launch_passive(env.model, env.data)  # Usar launch_passive para control manual
except Exception as e:
    print(f"Error al inicializar el visualizador: {e}")
    exit(1)

# Ejecutar un episodio de prueba con acciones sinusoidales
try:
    obs, _ = env.reset()
    for t in range(2000):  # Más pasos para observar movimiento
        # Acciones sinusoidales para movimiento visible
        action = np.sin(t * 0.1) * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        print(f"Paso: {t}, Ángulos del brazo: {obs[:6]}, Posición vaso: {obs[6:9]}, Orientación vaso: {obs[9:13]}, Recompensa: {reward}, Done: {done}, Truncated: {truncated}")
        
        # Sincronizar el visor con los datos actuales
        mj_viewer.sync()
        
        # Verificar si el vaso responde a la gravedad
        glass_z = env.data.qpos[8]  # Posición z del vaso
        print(f"Posición z del vaso: {glass_z}")
        
        if done or truncated:
            print("Episodio terminado. Reiniciando...")
            obs, _ = env.reset()
except Exception as e:
    print(f"Error durante la simulación: {e}")
finally:
    env.close()
    mj_viewer.close()