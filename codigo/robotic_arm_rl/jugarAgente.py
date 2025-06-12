import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robotic_arm_env import RoboticArmEnv
import time

# Configurar matplotlib para modo interactivo
plt.ion()
fig, ax = plt.subplots()

# Cargar el modelo entrenado
model = PPO.load("E:/1-2025/IA/Labs/Lab8/Lab8_brazo_robotico/codigo/robotic_arm_rl/ppo_robotic_arm.zip")

# Crear el entorno
env = RoboticArmEnv()

# Bucle infinito para ejecutar episodios continuamente
while True:
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        # Predecir la acción basada en la observación actual
        action, _states = model.predict(obs)
        
        # Ejecutar la acción en el entorno
        obs, reward, done, truncated, info = env.step(action)
        
        # Renderizar el entorno como rgb_array
        img = env.render(mode='rgb_array')
        
        # Mostrar la imagen con matplotlib
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        plt.draw()
        plt.pause(0.01)  # Pequeño retraso para visualización fluida
        
        # Acumular la recompensa
        total_reward += reward
    
    # Imprimir la recompensa total al final de cada episodio
    print(f"Episodio terminado. Recompensa total: {total_reward}")

# Cerrar el entorno al finalizar
env.close()
plt.close()